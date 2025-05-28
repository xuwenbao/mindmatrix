from fastapi import APIRouter

from ._openai_adapter import OpenAIAdapter, ChatCompletionRequest

router = APIRouter(prefix='/mm/v1')


@router.post("/agent/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
):
    agent = router.agent_provider(request.model)
    return await OpenAIAdapter.handle_chat_request(agent, request)


@router.post("/workflow/chat/completions")
async def workflow_chat_completions(
    request: ChatCompletionRequest,
):
    workflow = router.agent_provider(request.model, type_="workflow")
    return await OpenAIAdapter.handle_workflow_chat_request(workflow, request)
