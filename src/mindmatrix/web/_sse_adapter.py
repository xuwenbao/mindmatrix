import json
from typing import List, Optional, Dict, Any, AsyncGenerator, Union, Literal, AsyncIterator

from loguru import logger
from pydantic import BaseModel, Field
from fastapi import HTTPException, Request
from agno.agent import Agent
from agno.workflow.v2.workflow import Workflow
from agno.run.response import RunResponseContentEvent
from agno.run.v2.workflow import WorkflowRunResponseEvent


class Message(BaseModel):
    role: str = Field(..., description="消息发送者的角色，例如 'user' 或 'assistant'")
    content: str = Field(..., description="消息的内容")
    name: Optional[str] = Field(None, description="消息发送者的名称")


class ChatCompletionRequest(BaseModel):
    user_id: Optional[str] = Field(None, description="用户ID")
    session_id: Optional[str] = Field(None, description="可选的会话ID, 传入已有会话ID则继续对话, 否则生成新的会话ID")
    model: str = Field(..., description="模型名称")
    messages: List[Message] = Field(..., description="聊天消息的列表")
    stream: Optional[bool] = Field(False, description="可选的布尔值，指示是否流式传输响应")


class SSEAdapter:

    @staticmethod
    def extract_user_message(request: ChatCompletionRequest) -> str:
        """从请求中提取用户消息"""
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
    async def handle_chat_request(
        request: Request,
        handler: Union[Agent, Workflow],
        input: ChatCompletionRequest,
        type: Literal["agent", "workflow"] = "agent",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """处理聊天请求，返回适当的响应类型
        
        Args:
            request: FastAPI请求对象
            handler: Agent或Workflow实例
            input: 聊天请求对象
            type: 处理类型，"agent"或"workflow"
        """
        # 提取用户消息
        user_message = SSEAdapter.extract_user_message(input)
        logger.debug(f"Starting stream response for message: {user_message}")

        # 根据type选择对应的处理方法
        if type == "agent":
            async_gen = await handler.arun(user_message, stream=input.stream)

            async for chunk in async_gen:
                if await request.is_disconnected():
                    logger.warning("客户端已断开连接...")
                    break

                yield {
                    "event": "stream",
                    "data": json.dumps({"delta": chunk.content}, ensure_ascii=False),
                }
        else:
            response: AsyncIterator[WorkflowRunResponseEvent] = await handler.arun(message=user_message, stream=input.stream)

            async for event in response:
                if await request.is_disconnected():
                    logger.warning("客户端已断开连接...")
                    break

                if event.event == RunResponseContentEvent.event:
                    if event.extra_data and "artifacts" in event.extra_data:
                        for artifact in event.extra_data["artifacts"]:
                            yield {
                                "event": "artifact",
                                "data": artifact.model_dump_json(),
                            }
                    else:
                        yield {
                            "event": "stream",
                            "data": json.dumps({"delta": event.content}, ensure_ascii=False),
                        }
