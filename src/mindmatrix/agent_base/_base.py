import asyncio
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Type,
    Union,
    cast,
)

from pydantic import BaseModel, Field
from agno.media import Media
from agno.agent import Agent
from agno.models.base import Model
from agno.models.message import Message
from agno.run.messages import RunMessages
from agno.run.response import RunResponse
from agno.models.response import ModelResponse
from agno.workflow.v2 import Workflow as WorkflowV2, Step as StepV2, StepInput as StepInputV2, StepOutput as StepOutputV2
from agno.utils.log import (
    logger,
    log_debug,
    log_warning,
)


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
            index_of_last_user_message=index_of_last_user_message,
        )

        # We should break out of the run function
        if any(tool_call.is_paused for tool_call in run_response.tools or []):
            return self._handle_agent_run_paused(
                run_response=run_response, run_messages=run_messages, session_id=session_id, user_id=user_id
            )

        # 4. Update Agent Memory (后台协程异步执行，不阻塞主流程)
        asyncio.create_task(self._aupdate_memory_background(
            run_messages=run_messages,
            session_id=session_id,
            user_id=user_id,
        ))

        # 5. Calculate session metrics
        self._set_session_metrics(run_messages)

        # 6. Save session to storage
        self.write_to_storage(user_id=user_id, session_id=session_id)

        # 7. Save output to file if save_response_to_file is set
        self.save_run_response_to_file(message=run_messages.user_message, session_id=session_id)

        # Log Agent Run
        await self._alog_agent_run(user_id=user_id, session_id=session_id)

        # Convert the response to the structured format if needed
        self._convert_response_to_structured_format(run_response)

        log_debug(f"Agent Run End: {run_response.run_id}", center=True, symbol="*")

        return run_response

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
    ) -> None:
        """后台异步更新内存，不阻塞主流程"""
        try:
            async for _ in self._aupdate_memory(
                run_messages=run_messages,
                session_id=session_id,
                user_id=user_id,
            ):
                pass
        except Exception as e:
            logger.error(f"后台内存更新失败: {e}")
            # 不抛出异常，避免影响主流程
    

@dataclass
class BaseWorkflow(WorkflowV2):
    pass


@dataclass
class Step(StepV2):
    pass


@dataclass
class StepInput(StepInputV2):
    pass


@dataclass
class StepOutput(StepOutputV2):
    pass
