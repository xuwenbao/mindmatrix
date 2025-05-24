import collections.abc
from uuid import uuid4
from types import GeneratorType
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Callable, cast

from pydantic import Field
from agno.media import Media
from agno.agent import Agent
from agno.workflow import Workflow
from agno.run.response import RunResponse
from agno.utils.log import log_debug, logger
from agno.memory.v2.memory import Memory
from agno.memory.workflow import WorkflowMemory, WorkflowRun


class Artifact(Media):
    ...


@dataclass
class CustomRunResponse(RunResponse):
    response_artifacts: Optional[List[Artifact]] = Field(default_factory=list)


@dataclass(init=False)
class BaseAgent(Agent):

    workflow: Optional[Workflow] = None

    @property
    def custom_session_state(self) -> Dict[str, Any]:
        session_state = self.workflow.session_state \
            if hasattr(self, "workflow") and self.workflow is not None \
            else self.session_state
        return session_state

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

    def update_agent_session_ids(self):
        super().update_agent_session_ids()
        # 遍历所属类的属性并更新Agent实例的session_id
        for _, value in self.__class__.__dict__.items():
            if isinstance(value, Agent):
                value.workflow = self
                value.session_id = self.session_id

    def run_workflow(self, **kwargs: Any):
        """Run the Workflow"""

        # Set mode, debug, workflow_id, session_id, initialize memory
        self.set_storage_mode()
        self.set_debug()
        self.set_workflow_id()
        self.set_session_id()
        self.initialize_memory()

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
        except Exception as e:
            logger.error(f"Workflow.run() failed: {e}")
            raise e

        # The run_workflow() method handles both Iterator[RunResponse] and RunResponse
        # 临时添加异步迭代器类型处理
        if isinstance(result, collections.abc.AsyncIterator):
            # Initialize the run_response content
            self.run_response.content = ""

            async def async_result_generator():
                self.run_response = cast(RunResponse, self.run_response)
                if isinstance(self.memory, WorkflowMemory):
                    self.memory = cast(WorkflowMemory, self.memory)
                elif isinstance(self.memory, Memory):
                    self.memory = cast(Memory, self.memory)

                async for item in result:
                    if isinstance(item, RunResponse):
                        # Update the run_id, session_id and workflow_id of the RunResponse
                        item.run_id = self.run_id
                        item.session_id = self.session_id
                        item.workflow_id = self.workflow_id

                        # Update the run_response with the content from the result
                        if item.content is not None and isinstance(item.content, str):
                            self.run_response.content += item.content
                    else:
                        logger.warning(f"Workflow.run() should only yield RunResponse objects, got: {type(item)}")
                    yield item

                # Add the run to the memory
                if isinstance(self.memory, WorkflowMemory):
                    self.memory.add_run(WorkflowRun(input=self.run_input, response=self.run_response))
                elif isinstance(self.memory, Memory):
                    self.memory.add_run(session_id=self.session_id, run=self.run_response)  # type: ignore
                # Write this run to the database
                self.write_to_storage()
                log_debug(f"Workflow Run End: {self.run_id}", center=True)

            return async_result_generator
        # Case 1: The run method returns an Iterator[RunResponse]
        elif isinstance(result, (GeneratorType, collections.abc.Iterator)):
            # Initialize the run_response content
            self.run_response.content = ""

            def result_generator():
                self.run_response = cast(RunResponse, self.run_response)
                if isinstance(self.memory, WorkflowMemory):
                    self.memory = cast(WorkflowMemory, self.memory)
                elif isinstance(self.memory, Memory):
                    self.memory = cast(Memory, self.memory)

                for item in result:
                    if isinstance(item, RunResponse):
                        # Update the run_id, session_id and workflow_id of the RunResponse
                        item.run_id = self.run_id
                        item.session_id = self.session_id
                        item.workflow_id = self.workflow_id

                        # Update the run_response with the content from the result
                        if item.content is not None and isinstance(item.content, str):
                            self.run_response.content += item.content
                    else:
                        logger.warning(f"Workflow.run() should only yield RunResponse objects, got: {type(item)}")
                    yield item

                # Add the run to the memory
                if isinstance(self.memory, WorkflowMemory):
                    self.memory.add_run(WorkflowRun(input=self.run_input, response=self.run_response))
                elif isinstance(self.memory, Memory):
                    self.memory.add_run(session_id=self.session_id, run=self.run_response)  # type: ignore
                # Write this run to the database
                self.write_to_storage()
                log_debug(f"Workflow Run End: {self.run_id}", center=True)

            return result_generator
        # Case 2: The run method returns a RunResponse
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
            logger.warning(f"Workflow.run() should only return RunResponse objects, got: {type(result)}")
            return None