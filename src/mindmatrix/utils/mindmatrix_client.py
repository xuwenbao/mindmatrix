from typing import Dict, Any, List, Optional, Literal
import json
import uuid

from loguru import logger

from .http_client import AsyncHttpClient, SyncHttpClient


class MindMatrixError(Exception):
    """MindMatrix客户端错误"""
    pass


class AsyncMindMatrixClient(AsyncHttpClient):
    def __init__(
        self, 
        base_url: str = "http://localhost:9527", 
        api_key: str = None,
    ):
        super().__init__(base_url=base_url)
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json"
        }
        if api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

    async def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: str = "agent_router",
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        stream: bool = False,
        type_: Literal["agent", "workflow"] = "agent"
    ) -> Dict[str, Any]:
        """
        调用MindMatrix聊天完成接口
        
        Args:
            messages: 消息列表，格式为[{"role": "user", "content": "消息内容"}]
            model: 使用的模型名称
            session_id: 会话ID，可选
            user_id: 用户ID，可选
            stream: 是否启用流式响应
            type_: 类型，"agent" 或 "workflow"
            
        Returns:
            响应数据
        """
        if not session_id:
            session_id = str(uuid.uuid4())
            
        payload = {
            "model": model,
            "messages": messages,
            "session_id": session_id,
            "stream": stream
        }
        
        if user_id:
            payload["user_id"] = user_id
            
        path = f"/mm/v1/sse/{type_}/chat/completions"
        
        if stream:
            # 流式响应处理
            response = await self._post(path, json=payload, headers=self.headers)
            logger.debug(f"stream chat response: {response}")
            return response
        else:
            # 非流式响应
            path = f"/mm/v1/{type_}/chat/completions"
            response = await self._post(path, json=payload, headers=self.headers)
            logger.debug(f"chat response: {response}")
            
            if response.get("status") != 200:
                raise MindMatrixError(f"Failed to get chat completion: ({response.get('status')}) {response.get('error')}")
            
            return response.get("data", {})

    async def get_memories(self, user_id: str) -> List[Dict[str, Any]]:
        """
        获取用户记忆
        
        Args:
            user_id: 用户ID
            
        Returns:
            用户记忆列表
        """
        path = f"/mm/v1/memory/{user_id}/memories"
        response = await self._get(path, headers=self.headers)
        logger.debug(f"get memories response: {response}")
        
        if response.get("status") != 200:
            raise MindMatrixError(f"Failed to get memories: ({response.get('status')}) {response.get('error')}")
        
        return response.get("data", [])

    async def add_memory(
        self, 
        user_id: str, 
        memory: str, 
        topics: List[str] = None
    ) -> Dict[str, Any]:
        """
        添加用户记忆
        
        Args:
            user_id: 用户ID
            memory: 记忆内容
            topics: 标签列表，可选
            
        Returns:
            添加结果
        """
        path = f"/mm/v1/memory/{user_id}/memories"
        payload = {
            "memory": memory,
            "topics": topics or []
        }
        
        response = await self._post(path, json=payload, headers=self.headers)
        logger.debug(f"add memory response: {response}")
        
        if response.get("status") != 200:
            raise MindMatrixError(f"Failed to add memory: ({response.get('status')}) {response.get('error')}")
        
        return response.get("data", {})

    async def delete_memory(
        self, 
        user_id: str, 
        memory_id: str
    ) -> Dict[str, Any]:
        """
        删除用户记忆
        
        Args:
            user_id: 用户ID
            memory_id: 记忆ID
            
        Returns:
            删除结果
        """
        path = f"/mm/v1/memory/{user_id}/memories/{memory_id}"
        
        response = await self._delete(path, headers=self.headers)
        logger.debug(f"delete memory response: {response}")
        
        if response.get("status") != 200:
            raise MindMatrixError(f"Failed to delete memory: ({response.get('status')}) {response.get('error')}")
        
        return response.get("data", {})


class MindMatrixClient(SyncHttpClient):
    def __init__(
        self, 
        base_url: str = "http://localhost:9527", 
        api_key: str = None,
    ):
        super().__init__(base_url=base_url)
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json"
        }
        if api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: str = "agent_router",
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        stream: bool = False,
        type_: Literal["agent", "workflow"] = "agent"
    ) -> Dict[str, Any]:
        """
        调用MindMatrix聊天完成接口（同步版本）
        
        Args:
            messages: 消息列表，格式为[{"role": "user", "content": "消息内容"}]
            model: 使用的模型名称
            session_id: 会话ID，可选
            user_id: 用户ID，可选
            stream: 是否启用流式响应
            type_: 类型，"agent" 或 "workflow"
            
        Returns:
            响应数据
        """
        if not session_id:
            session_id = str(uuid.uuid4())
            
        payload = {
            "model": model,
            "messages": messages,
            "session_id": session_id,
            "stream": stream
        }
        
        if user_id:
            payload["user_id"] = user_id
            
        path = f"/mm/v1/sse/{type_}/chat/completions"
        
        if stream:
            # 流式响应处理
            response = self._post(path, json=payload, headers=self.headers)
            logger.debug(f"stream chat response: {response}")
            return response
        else:
            # 非流式响应
            path = f"/mm/v1/{type_}/chat/completions"
            response = self._post(path, json=payload, headers=self.headers)
            logger.debug(f"chat response: {response}")
            
            if response.get("status") != 200:
                raise MindMatrixError(f"Failed to get chat completion: ({response.get('status')}) {response.get('error')}")
            
            return response.get("data", {})

    def get_memories(self, user_id: str) -> List[Dict[str, Any]]:
        """
        获取用户记忆（同步版本）
        
        Args:
            user_id: 用户ID
            
        Returns:
            用户记忆列表
        """
        path = f"/mm/v1/memory/{user_id}/memories"
        response = self._get(path, headers=self.headers)
        logger.debug(f"get memories response: {response}")
        
        if response.get("status") != 200:
            raise MindMatrixError(f"Failed to get memories: ({response.get('status')}) {response.get('error')}")
        
        return response.get("data", [])

    def add_memory(
        self, 
        user_id: str, 
        memory: str, 
        topics: List[str] = None
    ) -> Dict[str, Any]:
        """
        添加用户记忆（同步版本）
        
        Args:
            user_id: 用户ID
            memory: 记忆内容
            topics: 标签列表，可选
            
        Returns:
            添加结果
        """
        path = f"/mm/v1/memory/{user_id}/memories"
        payload = {
            "memory": memory,
            "topics": topics or []
        }
        
        response = self._post(path, json=payload, headers=self.headers)
        logger.debug(f"add memory response: {response}")
        
        if response.get("status") != 200:
            raise MindMatrixError(f"Failed to add memory: ({response.get('status')}) {response.get('error')}")
        
        return response.get("data", {})

    def delete_memory(
        self, 
        user_id: str, 
        memory_id: str
    ) -> Dict[str, Any]:
        """
        删除用户记忆（同步版本）
        
        Args:
            user_id: 用户ID
            memory_id: 记忆ID
            
        Returns:
            删除结果
        """
        path = f"/mm/v1/memory/{user_id}/memories/{memory_id}"
        
        response = self._delete(path, headers=self.headers)
        logger.debug(f"delete memory response: {response}")
        
        if response.get("status") != 200:
            raise MindMatrixError(f"Failed to delete memory: ({response.get('status')}) {response.get('error')}")
        
        return response.get("data", {})


if __name__ == "__main__":
    from pprint import pprint
    
    async def main():
        # 异步用法示例
        async with AsyncMindMatrixClient() as client:
            # 聊天完成
            response = await client.chat_completion(
                messages=[{"role": "user", "content": "你好，请介绍一下你自己"}],
                model="agent_router",
                type_="agent"
            )
            pprint(response)
            
            # 获取记忆
            memories = await client.get_memories("user123")
            pprint(memories)
            
            # 添加记忆
            result = await client.add_memory(
                user_id="user123",
                memory="用户喜欢喝咖啡",
                topics=["偏好", "饮品"]
            )
            pprint(result)
            
            # 删除记忆
            delete_result = await client.delete_memory(
                user_id="user123",
                memory_id="memory_123"
            )
            pprint(delete_result)

    import asyncio
    asyncio.run(main())

    # 同步用法示例
    with MindMatrixClient() as client:
        # 聊天完成
        response = client.chat_completion(
            messages=[{"role": "user", "content": "你好，请介绍一下你自己"}],
            model="agent_router",
            type_="agent"
        )
        pprint(response)
        
        # 获取记忆
        memories = client.get_memories("user123")
        pprint(memories)
        
        # 添加记忆
        result = client.add_memory(
            user_id="user123",
            memory="用户喜欢喝咖啡",
            topics=["偏好", "饮品"]
        )
        pprint(result)
        
        # 删除记忆
        delete_result = client.delete_memory(
            user_id="user123",
            memory_id="memory_123"
        )
        pprint(delete_result) 