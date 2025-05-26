from dataclasses import dataclass
from agno.models.openai.like import OpenAILike as OpenAILike_


@dataclass
class OpenAILike(OpenAILike_):
    ...


@dataclass
class ZhipuAI(OpenAILike_):
    supports_structured_outputs: bool = False