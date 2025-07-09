import asyncio
import inspect
from uuid import uuid4
from dataclasses import dataclass
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Type,
    Union,
    cast,
    get_args,
)

from pydantic import BaseModel, Field
from agno.media import Media
from agno.agent import Agent
from agno.team.team import Team
from agno.workflow import Workflow
from agno.memory.v2.memory import Memory
from agno.utils.log import log_debug, logger
from agno.run.messages import RunMessages
from agno.run.team import TeamRunResponseEvent
from agno.run.workflow import WorkflowRunResponseEvent
from agno.run.response import RunResponse, RunResponseEvent
from agno.memory.workflow import WorkflowMemory, WorkflowRun
from agno.models.base import Model
from agno.models.message import Message
from agno.models.response import ModelResponse
from agno.utils.log import (
    log_debug,
    log_error,
    log_exception,
    log_info,
    log_warning,
    set_log_level_to_debug,
    set_log_level_to_info,
)

from ..web import get_current_workflow


class Artifact(Media):
    ...


@dataclass
class MindmatrixRunResponse(RunResponse):
    response_artifacts: Optional[List[Artifact]] = Field(default_factory=list)


@dataclass(init=False)
class BaseAgent(Agent):

    async def _arun(
        self,
        run_response: RunResponse,
        run_messages: RunMessages,
        session_id: str,
        user_id: Optional[str] = None,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        message: Optional[Union[str, List, Dict, Message]] = None,
        messages: Optional[Sequence[Union[Dict, Message]]] = None,
    ) -> RunResponse:
        """Run the Agent and yield the RunResponse.

        Steps:
        1. Reason about the task if reasoning is enabled
        2. Generate a response from the Model (includes running function calls)
        3. Add the run to memory
        4. Update Agent Memory
        5. Calculate session metrics
        6. Save session to storage
        7. Save output to file if save_response_to_file is set
        """
        log_debug(f"Agent Run Start: {run_response.run_id}", center=True)

        self.model = cast(Model, self.model)
        # 1. Reason about the task if reasoning is enabled
        await self._ahandle_reasoning(run_messages=run_messages)

        # Get the index of the last "user" message in messages_for_run
        # We track this so we can add messages after this index to the RunResponse and Memory
        index_of_last_user_message = len(run_messages.messages)

        # 2. Generate a response from the Model (includes running function calls)
        model_response: ModelResponse = await self.model.aresponse(
            messages=run_messages.messages,
            tools=self._tools_for_model,
            functions=self._functions_for_model,
            tool_choice=self.tool_choice,
            tool_call_limit=self.tool_call_limit,
            response_format=response_format,
        )

        # If a parser model is provided, structure the response separately
        if self.parser_model is not None:
            if self.response_model is not None:
                parser_response_format = self._get_response_format(self.parser_model)
                messages_for_parser_model = self.get_messages_for_parser_model(model_response, parser_response_format)
                parser_model_response: ModelResponse = await self.parser_model.aresponse(
                    messages=messages_for_parser_model,
                    response_format=parser_response_format,
                )
                parser_model_response_message: Optional[Message] = None
                for message in reversed(messages_for_parser_model):
                    if message.role == "assistant":
                        parser_model_response_message = message
                        break
                if parser_model_response_message is not None:
                    run_messages.messages.append(parser_model_response_message)
                    model_response.parsed = parser_model_response.parsed
                    model_response.content = parser_model_response.content
                else:
                    log_warning("Unable to parse response with parser model")
            else:
                log_warning("A response model is required to parse the response with a parser model")

        self._update_run_response(model_response=model_response, run_response=run_response, run_messages=run_messages)

        # 3. Add the run to memory
        self._add_run_to_memory(
            run_response=run_response,
            run_messages=run_messages,
            session_id=session_id,
            messages=messages,
            index_of_last_user_message=index_of_last_user_message,
        )

        # We should break out of the run function
        if any(tool_call.is_paused for tool_call in run_response.tools or []):
            return self._handle_agent_run_paused(
                run_response=run_response, session_id=session_id, user_id=user_id, message=message
            )

        # 4. Update Agent Memory (后台协程异步执行，不阻塞主流程)
        asyncio.create_task(self._aupdate_memory_background(
            run_messages=run_messages,
            session_id=session_id,
            user_id=user_id,
            messages=messages,
        ))

        # 5. Calculate session metrics
        self._set_session_metrics(run_messages)

        # 6. Save session to storage
        self.write_to_storage(user_id=user_id, session_id=session_id)

        # 7. Save output to file if save_response_to_file is set
        self.save_run_response_to_file(message=message, session_id=session_id)

        # Log Agent Run
        await self._alog_agent_run(user_id=user_id, session_id=session_id)

        # Convert the response to the structured format if needed
        self._convert_response_to_structured_format(run_response)

        log_debug(f"Agent Run End: {run_response.run_id}", center=True, symbol="*")

        return run_response

    @property
    def custom_session_state(self) -> Dict[str, Any]:
        """
        获取当前工作流的 session_state，如果当前工作流不存在，则返回当前 Agent 的 session_state
        """
        if hasattr(self, "workflow_id"):
            workflow = get_current_workflow()
            assert self.workflow_id == workflow.workflow_id, (
                f"workflow_id mismatch, agent.workflow_id = {self.workflow_id}, "
                f"workflow.workflow_id = {workflow.workflow_id}"
            )
            return workflow.session_state
        return self.session_state

    def convert_documents_to_string(self, docs: List[Any]) -> str:
        if docs is None or len(docs) == 0:
            return ""

        # 递归将所有 Pydantic BaseModel 转为 dict
        def to_dict(obj):
            if hasattr(obj, "model_dump"):  # Pydantic v2
                return obj.model_dump()
            elif hasattr(obj, "dict"):      # Pydantic v1
                return obj.dict()
            elif isinstance(obj, list):
                return [to_dict(i) for i in obj]
            elif isinstance(obj, dict):
                return {k: to_dict(v) for k, v in obj.items()}
            else:
                return obj

        docs_dict = to_dict(docs)

        if getattr(self, "references_format", None) == "yaml":
            import yaml
            return yaml.dump(docs_dict, allow_unicode=True)

        import json
        return json.dumps(docs_dict, indent=2, ensure_ascii=False)

    async def _aupdate_memory_background(
        self,
        run_messages: RunMessages,
        session_id: str,
        user_id: Optional[str] = None,
        messages: Optional[Sequence[Union[Dict, Message]]] = None,
    ) -> None:
        """后台异步更新内存，不阻塞主流程"""
        try:
            async for _ in self._aupdate_memory(
                run_messages=run_messages,
                session_id=session_id,
                user_id=user_id,
                messages=messages,
            ):
                pass
        except Exception as e:
            logger.error(f"后台内存更新失败: {e}")
            # 不抛出异常，避免影响主流程
    

class BaseWorkflow(Workflow):
    ...

    # def update_agent_session_ids(self):
    #     super().update_agent_session_ids()
    #     # 遍历所属类的属性并更新Agent实例的session_id
    #     for _, value in self.__class__.__dict__.items():
    #         if isinstance(value, Agent):
    #             value.workflow = self
    #             value.session_id = self.session_id

    async def arun_workflow(self, **kwargs: Any):
        """Run the Workflow asynchronously"""

        # Set mode, debug, workflow_id, session_id, initialize memory
        self.set_storage_mode()
        self.set_debug()
        self.set_monitoring()
        self.set_workflow_id()  # Ensure workflow_id is set
        self.set_session_id()
        self.initialize_memory()

        # Update workflow_id for all agents before registration
        for field_name, value in self.__class__.__dict__.items():
            if isinstance(value, Agent):
                value.initialize_agent()
                value.workflow_id = self.workflow_id

            if isinstance(value, Team):
                value.initialize_team()
                value.workflow_id = self.workflow_id

        # Register the workflow, which will also register agents and teams
        await self.aregister_workflow()

        # Create a run_id
        self.run_id = str(uuid4())

        # Set run_input, run_response
        self.run_input = kwargs
        self.run_response = RunResponse(run_id=self.run_id, session_id=self.session_id, workflow_id=self.workflow_id)

        # Read existing session from storage
        self.read_from_storage()

        # Update the session_id for all Agent instances
        self.update_agent_session_ids()

        log_debug(f"Workflow Run Start: {self.run_id}", center=True)
        try:
            self._subclass_run = cast(Callable, self._subclass_run)
            result = self._subclass_run(**kwargs)
            # 如果 result 是异步的，则等待结果
            if inspect.isawaitable(result):
                result = await result
        except Exception as e:
            logger.error(f"Workflow.arun() failed: {e}")
            raise e

        # Handle async iterator results
        if isinstance(result, (AsyncIterator, AsyncGenerator)):
            # Initialize the run_response content
            self.run_response.content = ""

            async def result_generator():
                self.run_response = cast(RunResponse, self.run_response)
                if isinstance(self.memory, WorkflowMemory):
                    self.memory = cast(WorkflowMemory, self.memory)
                elif isinstance(self.memory, Memory):
                    self.memory = cast(Memory, self.memory)

                async for item in result:
                    if (
                        isinstance(item, tuple(get_args(RunResponseEvent)))
                        or isinstance(item, tuple(get_args(TeamRunResponseEvent)))
                        or isinstance(item, tuple(get_args(WorkflowRunResponseEvent)))
                    ):
                        # Update the run_id, session_id and workflow_id of the RunResponseEvent
                        item.run_id = self.run_id
                        item.session_id = self.session_id
                        item.workflow_id = self.workflow_id

                        # Update the run_response with the content from the result
                        if hasattr(item, "content") and item.content is not None and isinstance(item.content, str):
                            self.run_response.content += item.content
                    else:
                        logger.warning(f"Workflow.arun() should only yield RunResponseEvent objects, got: {type(item)}")
                    yield item

                # Add the run to the memory
                if isinstance(self.memory, WorkflowMemory):
                    self.memory.add_run(WorkflowRun(input=self.run_input, response=self.run_response))
                elif isinstance(self.memory, Memory):
                    self.memory.add_run(session_id=self.session_id, run=self.run_response)  # type: ignore
                # Write this run to the database
                self.write_to_storage()
                log_debug(f"Workflow Run End: {self.run_id}", center=True)

            return result_generator()
        # Handle single RunResponse result
        elif isinstance(result, RunResponse):
            # Update the result with the run_id, session_id and workflow_id of the workflow run
            result.run_id = self.run_id
            result.session_id = self.session_id
            result.workflow_id = self.workflow_id

            # Update the run_response with the content from the result
            if result.content is not None and isinstance(result.content, str):
                self.run_response.content = result.content

            # Add the run to the memory
            if isinstance(self.memory, WorkflowMemory):
                self.memory.add_run(WorkflowRun(input=self.run_input, response=self.run_response))
            elif isinstance(self.memory, Memory):
                self.memory.add_run(session_id=self.session_id, run=self.run_response)  # type: ignore
            # Write this run to the database
            self.write_to_storage()
            log_debug(f"Workflow Run End: {self.run_id}", center=True)
            return result
        else:
            logger.warning(f"Workflow.arun() should only return RunResponse objects, got: {type(result)}")
            return None
