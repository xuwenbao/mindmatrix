from pprint import pformat
from typing import List, Optional

from loguru import logger
from agno.embedder.base import Embedder

from .reranker_client import AsyncRerankerClient

try:
    from pymilvus import MilvusClient
except ImportError:
    logger.error("pymilvus is not installed, please install it with: pip install pymilvus")


class MilvusAnnotatedResponseMixin:
    """
    标注回复
    """
    @classmethod
    async def annotated_response(
        cls, 
        query: str,
        background_info: str,
        embedder: Embedder,
        milvus: MilvusClient,
        collection_name: str,
        anns_field: str,
        output_fields: List[str],
        content_field: str,
        *,
        use_reranker: bool = False,
        reranker: Optional[AsyncRerankerClient] = None,
        similarity_threshold: Optional[float] = None,
        metric_type: str = "COSINE",
        filter: str = "",
        limit: int = 5,
    ) -> List[str]:
        query = query[-1]["content"] if isinstance(query, list) else query

        docs = await cls._retrieve_documents(
            f"Instruct: 根据用户的指令，召回能够完成该指令任务的智能体。\nQuery:{query}",
            embedder,
            milvus,
            collection_name,
            anns_field,
            output_fields,
            similarity_threshold=similarity_threshold,
            metric_type=metric_type,
            filter=filter,
            limit=limit,
        )

        if use_reranker and len(docs) > 1:
            rerank_docs = [item[content_field] for item in docs]
            rerank_results = await cls._rerank_documents(
                query=query,
                instruction=f"{background_info}。根据以上背景信息(如年龄、职业、兴趣爱好等)，结合用户查询，召回对应的智能体。" \
                    if background_info else \
                    f"根据用户的指令，召回能够完成该指令任务的智能体。",
                documents=rerank_docs,
                reranker_client=reranker,
            )
            docs = [docs[item["index"]] for item in rerank_results]
            logger.debug(f"reranked docs: {docs}")
        
        return docs

    @classmethod
    async def _retrieve_documents(
        cls,
        query: str,
        embedder: Embedder,
        milvus: MilvusClient,
        collection_name: str,
        anns_field: str,
        output_fields: List[str],
        *,
        similarity_threshold: float = None,
        metric_type: str = "COSINE",
        filter: str = "",
        limit: int = 5,
    ) -> List[dict]:
        """
        检索文档
        """
        logger.info(f"embedding query:\n{query}")
        embeddings = embedder.get_embedding(query)
        logger.debug(f"get query embedding: {embeddings[:10]}, len: {len(embeddings)}")

        res = milvus.search(
            collection_name=collection_name, 
            anns_field=anns_field,
            data=[embeddings],
            limit=limit,
            filter=filter,
            search_params={"metric_type": metric_type},
            output_fields=output_fields,
        )
        if res and len(res) > 0:
            logger.info(f"Found {len(res[0])} search results:")
            for i, item in enumerate(res[0]):
                logger.debug(f"Result {i}: {pformat(item)}")

            if similarity_threshold is not None:
                logger.info(f"filtering docs by similarity threshold ({similarity_threshold})")
                results = [
                    {key: item["entity"][key] for key in output_fields}
                    for item in res[0]
                    if item["distance"] >= similarity_threshold
                ]
                logger.info(f"Found {len(results)} search results after filtering by similarity threshold ({similarity_threshold}):")
                for i, item in enumerate(results):
                    logger.debug(f"Result {i}: {pformat(item)}")

                return results
            return [{key: item["entity"][key] for key in output_fields} for item in res[0]]
        else:
            logger.debug("No search results found")
            return []

    @classmethod
    async def _rerank_documents(
        cls,
        query: str,
        instruction: str,
        reranker_client: AsyncRerankerClient,
        documents: List[str],
    ) -> List[dict]:
        """
        对文档进行重排序
        
        Args:
            query: 查询文本
            documents: 待重排序的文档列表
            model: 使用的重排序模型
            api_key: API密钥
            base_url: API基础URL
            
        Returns:
            重排序后的文档列表，包含分数和排名信息
        """
        try:
            logger.info(f"Starting rerank for query: '{query}' with {len(documents)} documents")
            
            results = await reranker_client.score(
                instruction=instruction, 
                queries=[query] * len(documents), 
                documents=documents,
            )
            if results and len(results) > 0:
                # 按照score字段降序排序
                results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)
                logger.info(f"Found {len(results)} rerank results:")
                for i, item in enumerate(results):
                    logger.debug(f"Rerank Result {i}: {pformat(item)}, doc: 「{documents[item['index']]}」")

            return results
        except Exception as e:
            logger.error(f"Failed to rerank documents: {e}")
            raise e