from dataclasses import dataclass
from agno.models.openai.like import OpenAILike


@dataclass
class ZhipuAI(OpenAILike):
    supports_structured_outputs: bool = False