import uuid
from typing import Literal, Optional

from loguru import logger
from pydantic import BaseModel, Field
from fastapi import APIRouter, Request
from agno.memory.v2.schema import UserMemory
from sse_starlette.sse import EventSourceResponse

from ._sse_adapter import SSEAdapter, ChatCompletionRequest as SSEChatCompletionRequest
from ._openai_adapter import OpenAIAdapter, ChatCompletionRequest as OpenAIChatCompletionRequest


router = APIRouter(prefix='/mm/v1')


class MemoryCreateRequest(BaseModel):
    memory: str = Field(description="记忆内容")
    topics: Optional[list[str]] = Field(default=[], description="记忆主题")


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
    """
    智能聊天接口，支持流式响应和会话管理。

    ## 功能描述
    此接口实现了Coordinator智能体路由和对话功能，支持流式响应和会话管理。

    ## 请求参数

    ### Headers
    暂无

    ### Query
    - type: 可选，类型，值为 "agent" 或 "workflow"，默认值为 "agent"

    ### Body
    ```json
    {
        "messages": [
            {
                "role": "string",  # 消息发送者角色，如 "user" 或 "assistant"
                "content": "string"  # 消息内容
            }
        ],
        "session_id": "string",  # 可选的会话ID，用于继续已有对话
        "stream": boolean  # 是否启用流式响应
    }
    ```

    ## 响应说明
    ### 成功响应 (200 OK)
    - 流式响应格式 (SSE):
        - 事件类型: "stream" - 实时文本响应
          事件内容示例:
          ```json
            {
                "event": "stream",
                "data": "{\"delta\": \"你好，这里是AI的回复内容...\"}"
            }
          ```
        - 事件类型: "artifact" - 结构化数据响应
          事件内容示例:
          ```json
            {
                "event": "artifact",
                "data": "{\"type\": \"agents\", \"content\": [{\"id\": \"agent1\", \"name\": \"智能体A\"}, {\"id\": \"agent2\", \"name\": \"智能体B\"}]}"
            }
          ```

    ### 错误响应
    - 400 Bad Request: 请求参数无效
    - 401 Unauthorized: API密钥无效或 Bearer token 缺失

    ### 调用示例代码
    ```python
    import json
    import httpx
    from sse_starlette.sse import aconnect_sse


    url = f"<server endpoint>/mm/v1/sse/workflow/chat/completions"
    payload = {
        "model": "agent_router",
        "messages": [{"role": "user", "content": <query>}],
        "session_id": <session_id>,
        "stream": True,
    }
    headers = {
        "Content-Type": "application/json",
    }

    logger.debug(f"payload: {payload}")
    
    # 使用httpx-sse库进行异步SSE连接
    async with httpx.AsyncClient(timeout=None) as client:
        async with aconnect_sse(client, "POST", url, json=payload, headers=headers) as event_source:
            async for sse in event_source.aiter_sse():
                logger.info(f"SSE {sse.event} 事件(type: {type(sse.data)}): {sse.data}")
                
                try:
                    if sse.event == "stream":
                        chunk = json.loads(sse.data)
                        ... # 处理流式响应
                    elif sse.event == "artifact":
                        ... # 处理结构化数据响应
                except json.JSONDecodeError as e:
                    logger.warning(f"无法解析JSON数据: {sse.data}")
                    logger.exception(e)
                    ... # 处理错误响应
    ```
    """
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


@router.get("/memory/{user_id}/memories")
async def get_memories(
    user_id: str,
):
    memory = router.memory_provider()
    assert memory is not None, "Memory is not set"
    return memory.get_user_memories(user_id)


@router.post("/memory/{user_id}/memories")
async def add_memory(
    user_id: str,
    input: MemoryCreateRequest,
):
    memory = router.memory_provider()
    assert memory is not None, "Memory is not set"
    return memory.add_user_memory(
        user_id=user_id, 
        memory=UserMemory(
            memory=input.memory,
            topics=input.topics,
        ),
    )


@router.delete("/memory/{user_id}/memories/{memory_id}")
async def delete_memory(
    user_id: str,
    memory_id: str,
):
    """
    删除指定用户的指定记忆
    
    ## 功能描述
    此接口用于删除指定用户的特定记忆记录。
    
    ## 请求参数
    
    ### Path Parameters
    - user_id: 用户ID
    - memory_id: 记忆ID
    
    ## 响应说明
    ### 成功响应 (200 OK)
    返回删除操作的结果
    
    ### 错误响应
    - 404 Not Found: 记忆不存在
    - 500 Internal Server Error: 服务器内部错误
    
    ## 调用示例
    ```bash
    DELETE /mm/v1/memory/jane_doe@example.com/memories/memory_123
    ```
    """
    memory = router.memory_provider()
    assert memory is not None, "Memory is not set"
    return memory.delete_user_memory(user_id=user_id, memory_id=memory_id)
