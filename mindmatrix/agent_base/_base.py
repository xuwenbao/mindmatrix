import inspect
from uuid import uuid4
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Callable, cast, AsyncIterator, AsyncGenerator, get_args

from pydantic import Field
from agno.media import Media
from agno.agent import Agent
from agno.team.team import Team
from agno.workflow import Workflow
from agno.memory.v2.memory import Memory
from agno.utils.log import log_debug, logger
from agno.run.team import TeamRunResponseEvent
from agno.run.workflow import WorkflowRunResponseEvent
from agno.run.response import RunResponse, RunResponseEvent
from agno.memory.workflow import WorkflowMemory, WorkflowRun

from ..web import get_current_workflow


class Artifact(Media):
    ...


@dataclass
class MindmatrixRunResponse(RunResponse):
    response_artifacts: Optional[List[Artifact]] = Field(default_factory=list)


@dataclass(init=False)
class BaseAgent(Agent):

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
