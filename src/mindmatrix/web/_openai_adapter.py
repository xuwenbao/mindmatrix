import json
import asyncio
from typing import AsyncGenerator, AsyncIterator,List, Optional, Union

from loguru import logger
from pydantic import BaseModel, Field
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from agno.agent import Agent, RunResponse
from agno.workflow.v2.workflow import Workflow
from agno.run.response import RunResponseContentEvent
from agno.run.v2.workflow import WorkflowRunResponseEvent


class Message(BaseModel):
    role: str
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="deepseek-v3")
    messages: List[Message]
    temperature: Optional[float] = Field(default=0.7)
    stream: Optional[bool] = Field(default=False)
    max_tokens: Optional[int] = Field(default=None)

    
class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Optional[Message] = None
    delta: Optional[DeltaMessage] = None
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]


class OpenAIAdapter:
    """
    Agno到OpenAI API的适配器，提供OpenAI兼容的接口
    """
    
    @staticmethod
    def extract_user_message(request: ChatCompletionRequest) -> str:
        """从OpenAI请求中提取用户消息"""
        messages = request.messages
        if not messages:
            raise HTTPException(status_code=400, detail="No messages provided")
        
        # 获取用户的最后一条消息
        user_message = None
        for msg in reversed(messages):
            if msg.role == "user":
                user_message = msg.content
                break
        
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")
            
        return user_message

    @staticmethod
    def create_completion_response(response: RunResponse, model: str) -> ChatCompletionResponse:
        """创建非流式响应"""
        return ChatCompletionResponse(
            id=f"chatcmpl-{id(response)}",
            object="chat.completion",
            created=int(asyncio.get_event_loop().time()),
            model=model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=Message(
                        role="assistant",
                        content=response.get_content_as_string(by_alias=True)
                    ),
                    finish_reason="stop"
                )
            ]
        )

    @staticmethod
    async def stream_response(agent: Agent, message: str) -> AsyncGenerator[str, None]:
        """生成与OpenAI兼容的流式响应"""
        logger.debug(f"Starting stream response for message: {message}")

        try:
            # 使用agent生成流式响应
            async_response = await agent.arun(message, stream=True)
            logger.debug("Agent response obtained, starting streaming")
            
            # 生成与OpenAI兼容的事件流
            chunk_id = f"chatcmpl-{id(async_response)}"
            created_time = int(asyncio.get_event_loop().time())
            
            # 发送开始事件
            first_chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": "deepseek-v3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None
                    }
                ]
            }
            yield f"data: {json.dumps(first_chunk)}\n\n"
            
            # 迭代流式响应内容
            async for response_delta in async_response:
                content = response_delta.content if hasattr(response_delta, 'content') else ""
                try:
                    logger.debug(f"Processing delta content: {content}")
                except UnicodeEncodeError:
                    # 对于可能包含emoji的内容，使用更安全的记录方式
                    logger.debug(f"Processing delta content: {repr(content)}")
                
                if content:
                    chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": "deepseek-v3",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": content},
                                "finish_reason": None
                            }
                        ]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
            
            # 发送结束事件
            last_chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": "deepseek-v3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }
                ]
            }
            yield f"data: {json.dumps(last_chunk)}\n\n"
            
            yield 'data: [DONE]\n\n'
        
        except Exception as e:
            logger.error(f"Error in stream_response: {str(e)}")
            # 返回错误信息
            error_chunk = {
                "id": f"chatcmpl-error",
                "object": "chat.completion.chunk",
                "created": int(asyncio.get_event_loop().time()),
                "model": "deepseek-v3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": f"Error: {str(e)}"},
                        "finish_reason": "error"
                    }
                ]
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield 'data: [DONE]\n\n'

    @staticmethod
    async def stream_workflow_response(workflow: Workflow, message: str) -> AsyncGenerator[str, None]:
        """生成与OpenAI兼容的流式响应"""
        logger.debug(f"Starting stream response for message: {message}")

        try:
            # 使用workflow生成流式响应
            async_response: AsyncIterator[WorkflowRunResponseEvent] = await workflow.arun(message, stream=True)
            logger.debug("Workflow response obtained, starting streaming")
            
            # 生成与OpenAI兼容的事件流
            chunk_id = f"chatcmpl-{id(async_response)}"
            created_time = int(asyncio.get_event_loop().time())
            
            # 发送开始事件
            first_chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": "deepseek-v3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None
                    }
                ]
            }
            yield f"data: {json.dumps(first_chunk)}\n\n"
            
            # 迭代流式响应内容
            async for event in async_response:
                if event.event == RunResponseContentEvent.event:
                    content = event.content
                    try:
                        logger.debug(f"Processing delta content: {content}")
                    except UnicodeEncodeError:
                        # 对于可能包含emoji的内容，使用更安全的记录方式
                        logger.debug(f"Processing delta content: {repr(content)}")
                    
                    if content:
                        chunk = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": "deepseek-v3",
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": content},
                                    "finish_reason": None
                                }
                            ]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
            
            # 发送结束事件
            last_chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": "deepseek-v3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }
                ]
            }
            yield f"data: {json.dumps(last_chunk)}\n\n"
            
            yield 'data: [DONE]\n\n'
        
        except Exception as e:
            logger.error(f"Error in stream_response: {str(e)}")
            # 返回错误信息
            error_chunk = {
                "id": f"chatcmpl-error",
                "object": "chat.completion.chunk",
                "created": int(asyncio.get_event_loop().time()),
                "model": "deepseek-v3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": f"Error: {str(e)}"},
                        "finish_reason": "error"
                    }
                ]
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield 'data: [DONE]\n\n'
            
    @staticmethod
    async def handle_chat_request(agent: Agent, request: ChatCompletionRequest) -> Union[ChatCompletionResponse, StreamingResponse]:
        """处理聊天请求，返回适当的响应类型"""
        # 提取用户消息
        user_message = OpenAIAdapter.extract_user_message(request)
        
        # 如果请求流式输出
        if request.stream:
            return StreamingResponse(
                OpenAIAdapter.stream_response(agent, user_message),
                media_type="text/event-stream"
            )
        
        # 非流式输出
        response = await agent.arun(user_message)
        return OpenAIAdapter.create_completion_response(response, request.model)
    
    @staticmethod
    async def handle_workflow_chat_request(workflow: Workflow, request: ChatCompletionRequest) -> Union[ChatCompletionResponse, StreamingResponse]:
        """处理聊天请求，返回适当的响应类型"""
        # 提取用户消息
        user_message = OpenAIAdapter.extract_user_message(request)
        
        # 如果请求流式输出
        if request.stream:
            return StreamingResponse(
                OpenAIAdapter.stream_workflow_response(workflow, user_message),
                media_type="text/event-stream"
            )
        
        # 非流式输出 - 使用异步方法获取RunResponse
        response = await workflow.arun(user_message)
        return OpenAIAdapter.create_completion_response(response, request.model)