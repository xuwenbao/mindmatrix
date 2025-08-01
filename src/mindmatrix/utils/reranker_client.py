from typing import Dict, Any, List

from loguru import logger

from .http_client import AsyncHttpClient, SyncHttpClient


class RerankerError(Exception):
    ...


class AsyncRerankerClient(AsyncHttpClient):
    def __init__(
        self, 
        base_url: str = "https://api.siliconflow.cn/v1", 
        model: str = None, 
        api_key: str = None,
    ):
        super().__init__(base_url=base_url)
        self.model = model
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    async def score(self, instruction: str, queries: List[str], documents: List[str]) -> List[Dict[str, Any]]:
        prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

        query_template = "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
        document_template = "<Document>: {doc}{suffix}"

        queries = [
            query_template.format(prefix=prefix, instruction=instruction, query=query)
            for query in queries
        ]
        documents = [
            document_template.format(doc=doc, suffix=suffix) for doc in documents
        ]

        path = "/score"
        payload = {
            "model": self.model,
            "text_1": queries,
            "text_2": documents,
            "truncate_prompt_tokens": -1,
        }
        response = await self._post(path, json=payload, headers=self.headers)
        logger.debug(f"score response: {response}")
        return response["data"]["data"] # TODO: 处理HTTP Error

    async def rerank(self, query: str, documents: List[str]) -> List[Dict[str, Any]]:
        """
        对文档进行重排序
        
        Args:
            query: 查询文本
            documents: 待重排序的文档列表
            model: 使用的重排序模型
            
        Returns:
            重排序后的文档列表，包含分数和排名信息
        """
        path = "/rerank"
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents
        }
        
        response = await self._post(path, json=payload, headers=self.headers)
        if response["status"] != 200:
            raise RerankerError(f"Failed to rerank documents: ({response['status']}) {response['error']}")
        
        return response["data"]["results"]


class RerankerClient(SyncHttpClient):
    def __init__(
        self, 
        base_url: str = "https://api.siliconflow.cn/v1", 
        model: str = None, 
        api_key: str = None,
    ):
        super().__init__(base_url=base_url)
        self.model = model
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def rerank(self, query: str, documents: List[str]) -> List[Dict[str, Any]]:
        """
        对文档进行重排序
        
        Args:
            query: 查询文本
            documents: 待重排序的文档列表
            model: 使用的重排序模型
            
        Returns:
            重排序后的文档列表，包含分数和排名信息
        """
        path = "/rerank"
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents
        }
        
        response = self._post(path, json=payload, headers=self.headers)
        logger.debug(f"rerank response: {response}")
        
        if response["status"] != 200:
            raise RerankerError(f"Failed to rerank documents: ({response['status']}) {response['error']}")
        
        return response["results"]


if __name__ == "__main__":
    from pprint import pprint
    
    async def main():
        async with AsyncRerankerClient() as client:
            response = await client.rerank(
                query="Apple",
                documents=["apple", "banana", "fruit", "vegetable"]
            )
            pprint(response)

    import asyncio
    asyncio.run(main())

    # 同步用法示例
    with RerankerClient() as client:
        response = client.rerank(
            query="Apple",
            documents=["apple", "banana", "fruit", "vegetable"]
        )
        pprint(response) 