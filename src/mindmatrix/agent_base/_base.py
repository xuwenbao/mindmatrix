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
)


class Artifact(Media):
    ...


@dataclass(init=False)
class BaseAgent(Agent):

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
    

@dataclass(init=False)
class BaseWorkflow(WorkflowV2):
    pass


@dataclass(init=False)
class Step(StepV2):
    pass


@dataclass
class StepInput(StepInputV2):
    pass


@dataclass
class StepOutput(StepOutputV2):
    pass
