import uuid
from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse
from typing import Literal

from loguru import logger

from ._sse_adapter import SSEAdapter, ChatCompletionRequest as SSEChatCompletionRequest
from ._openai_adapter import OpenAIAdapter, ChatCompletionRequest as OpenAIChatCompletionRequest

router = APIRouter(prefix='/mm/v1')


@router.post("/agent/chat/completions")
async def chat_completions(
    input: OpenAIChatCompletionRequest,
):
    agent = router.agent_provider(input.model)
    return await OpenAIAdapter.handle_chat_request(agent, input)


@router.post("/workflow/chat/completions")
async def workflow_chat_completions(
    input: OpenAIChatCompletionRequest,
):
    workflow = router.agent_provider(input.model, type_="workflow")
    return await OpenAIAdapter.handle_workflow_chat_request(workflow, input)


@router.post("/sse/{type}/chat/completions")
async def sse_chat_completions(
    request: Request,
    input: SSEChatCompletionRequest,
    type: Literal["agent", "workflow"],
):
    logger.debug(f"Received request: {input}")

    # 如果session_id为空，则生成一个新的session_id
    session_id = input.session_id or str(uuid.uuid4())

    # 根据type参数获取对应的agent或workflow
    handler = router.agent_provider(
        input.model,
        type_=type,
        user_id=input.user_id,
        session_id=session_id,
    )
    sse_event_generator = SSEAdapter.handle_chat_request(request, handler, input, type=type)
    
    return EventSourceResponse(sse_event_generator)
