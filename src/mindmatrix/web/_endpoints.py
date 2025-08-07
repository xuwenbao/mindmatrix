import uuid
from typing import Literal, Optional

from loguru import logger
from pydantic import BaseModel, Field
from fastapi import APIRouter, Request
from agno.memory.v2.schema import UserMemory
from sse_starlette.sse import EventSourceResponse

from ._contextvars import set_current_workflow
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
    """
    智能体聊天接口，支持标准聊天对话。

    ## 功能描述
    此接口实现了智能体聊天功能，支持与单个智能体进行对话交互。

    ## 请求参数

    ### Headers
    暂无

    ### Body
    ```json
    {
        "model": "string",  # 智能体模型名称
        "messages": [
            {
                "role": "string",  # 消息发送者角色，如 "user" 或 "assistant"
                "content": "string"  # 消息内容
            }
        ],
        "session_id": "string",  # 可选的会话ID，用于继续已有对话
        "user_id": "string"  # 可选的用户ID
    }
    ```

    ## 响应说明
    ### 成功响应 (200 OK)
    ```json
    {
        "id": "string",  # 响应ID
        "object": "chat.completion",
        "created": 1234567890,
        "model": "string",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "智能体的回复内容"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    }
    ```

    ### 错误响应
    - 400 Bad Request: 请求参数无效
    - 401 Unauthorized: API密钥无效或 Bearer token 缺失
    - 500 Internal Server Error: 服务器内部错误

    ### 调用示例代码
    ```python
    import httpx

    url = "<server endpoint>/mm/v1/agent/chat/completions"
    payload = {
        "model": "agent_router",
        "messages": [{"role": "user", "content": "你好，请介绍一下你自己"}],
        "session_id": "session_123",
        "user_id": "user_456"
    }
    headers = {
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers=headers)
        result = response.json()
        print(result["choices"][0]["message"]["content"])
    ```
    """
    agent = router.agent_provider(input.model)
    return await OpenAIAdapter.handle_chat_request(agent, input)


@router.post("/workflow/chat/completions")
async def workflow_chat_completions(
    input: OpenAIChatCompletionRequest,
):
    """
    工作流聊天接口，支持复杂工作流对话。

    ## 功能描述
    此接口实现了工作流聊天功能，支持多步骤、复杂的对话工作流处理。

    ## 请求参数

    ### Headers
    暂无

    ### Body
    ```json
    {
        "model": "string",  # 工作流模型名称
        "messages": [
            {
                "role": "string",  # 消息发送者角色，如 "user" 或 "assistant"
                "content": "string"  # 消息内容
            }
        ],
        "session_id": "string",  # 可选的会话ID，用于继续已有对话
        "user_id": "string"  # 可选的用户ID
    }
    ```

    ## 响应说明
    ### 成功响应 (200 OK)
    ```json
    {
        "id": "string",  # 响应ID
        "object": "chat.completion",
        "created": 1234567890,
        "model": "string",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "工作流的回复内容"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 25,
            "total_tokens": 40
        }
    }
    ```

    ### 错误响应
    - 400 Bad Request: 请求参数无效
    - 401 Unauthorized: API密钥无效或 Bearer token 缺失
    - 500 Internal Server Error: 服务器内部错误

    ### 调用示例代码
    ```python
    import httpx

    url = "<server endpoint>/mm/v1/workflow/chat/completions"
    payload = {
        "model": "workflow_router",
        "messages": [{"role": "user", "content": "请帮我分析这个数据"}],
        "session_id": "workflow_session_123",
        "user_id": "user_456"
    }
    headers = {
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers=headers)
        result = response.json()
        print(result["choices"][0]["message"]["content"])
    ```
    """
    workflow = router.agent_provider(input.model, type_="workflow")
    set_current_workflow(workflow)
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

    if type == "workflow":
        set_current_workflow(handler)

    sse_event_generator = SSEAdapter.handle_chat_request(request, handler, input, type=type)
    
    return EventSourceResponse(sse_event_generator)


@router.get("/memory/{user_id}/memories")
async def get_memories(
    user_id: str,
):
    """
    获取用户记忆列表接口

    ## 功能描述
    此接口用于获取指定用户的所有记忆记录

    ## 请求参数

    ### Path Parameters
    - user_id: 用户ID，用于标识特定用户

    ### Headers
    暂无

    ### Query Parameters
    暂无

    ## 响应说明
    ### 成功响应 (200 OK)
    ```json
    [
        {
            "memory": "用户喜欢打篮球",
            "topics": [
                "hobbies"
            ],
            "input": "我喜欢打篮球 /nothink",
            "last_updated": "2025-07-23T08:08:07.973604",
            "memory_id": "cdb1fa19-edf5-45f7-85c6-4bb0d9bf31cd"
        },
        {
            "memory": "用户是一名后端工程师",
            "topics": [],
            "input": null,
            "last_updated": "2025-07-23T14:59:17.362716",
            "memory_id": "a716b643-9eac-4ce2-a20d-d40883c1943a"
        },
        {
            "memory": "用户在天府绛溪实验室工作",
            "topics": [],
            "input": null,
            "last_updated": "2025-07-23T14:59:07.110299",
            "memory_id": "a70c7faf-cf21-495b-b41b-34acc928bbff"
        }
    ]
    ```

    ### 错误响应
    - 400 Bad Request: 请求参数无效
    - 404 Not Found: 用户不存在
    - 500 Internal Server Error: 服务器内部错误

    ### 调用示例代码
    ```python
    import httpx

    user_id = "cd65ed5f-29ad-4ac2-ac15-785c251141c0"
    url = f"<server endpoint>/mm/v1/memory/{user_id}/memories"
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        memories = response.json()
        for memory in memories:
            print(f"记忆ID: {memory['id']}")
            print(f"内容: {memory['memory']}")
            print(f"主题: {memory['topics']}")
    ```
    """
    memory = router.memory_provider()
    assert memory is not None, "Memory is not set"
    return memory.get_user_memories(user_id=user_id)


@router.post("/memory/{user_id}/memories")
async def add_memory(
    user_id: str,
    input: MemoryCreateRequest,
):
    """
    添加用户记忆接口。

    ## 功能描述
    此接口用于为指定用户添加新的记忆记录，支持记忆存储和管理。

    ## 请求参数

    ### Path Parameters
    - user_id: 用户ID，用于标识特定用户

    ### Headers
    - Content-Type: application/json

    ### Body
    ```json
    {
        "memory": "string",  # 记忆内容，必填
        "topics": ["string"]  # 记忆主题列表，可选，默认为空数组
    }
    ```

    ## 响应说明
    ### 成功响应 (200 OK)
    ```json
    {
        "memory_id": "cdb1fa19-edf5-45f7-85c6-4bb0d9bf31cd",
        "memory": "用户今天学习了React框架",
        "topics": ["前端开发", "React"],
    }
    ```

    ### 错误响应
    - 400 Bad Request: 请求参数无效
    - 401 Unauthorized: API密钥无效或 Bearer token 缺失
    - 500 Internal Server Error: 服务器内部错误

    ### 调用示例代码
    ```python
    import httpx

    user_id = "cd65ed5f-29ad-4ac2-ac15-785c251141c0"
    url = f"<server endpoint>/mm/v1/memory/{user_id}/memories"
    payload = {
        "memory": "用户今天学习了React框架",
        "topics": ["前端开发", "React"]
    }
    headers = {
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers=headers)
        result = response.json()
        print(f"新增记忆ID: {result['id']}")
        print(f"记忆内容: {result['memory']}")
    ```
    """
    memory = router.memory_provider()
    assert memory is not None, "Memory is not set"
    memory_id = memory.add_user_memory(
        user_id=user_id, 
        memory=UserMemory(
            memory=input.memory,
            topics=input.topics,
        ),
    )
    return {
        "memory_id": memory_id,
        "memory": input.memory,
        "topics": input.topics,
    }

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
