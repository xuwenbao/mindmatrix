from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Literal

from loguru import logger
from agno.reranker.base import Reranker
from agno.vectordb.search import SearchType
from agno.document import Document as Document_
from agno.vectordb.milvus import Milvus as Milvus_
from agno.embedder.openai import OpenAIEmbedder as OpenAIEmbedder_


@dataclass
class Document(Document_):
    ...


@dataclass
class OpenAIEmbedder(OpenAIEmbedder_):
    dimensions: int = 1024


@dataclass
class VectorDb:
    ...


class VectorDbProvider:
    def __init__(self, mindmatrix):
        self.mindmatrix = mindmatrix

    def __call__(self, vectordb_name: str) -> VectorDb:
        return self.mindmatrix.get_vectordb(vectordb_name)


class Milvus(VectorDb):
    
    def __init__(
        self, 
        embedder: OpenAIEmbedder, 
        uri: str = "http://localhost:19530",
        *,
        token: Optional[str] = None, 
        search_type: SearchType = SearchType.vector, 
        reranker: Optional[Reranker] = None, 
        sparse_vector_dimensions: int = 10000, 
        **kwargs
    ):
        self._embedder = embedder
        self._uri = uri
        self._token = token
        self._search_type = search_type
        self._reranker = reranker
        self._sparse_vector_dimensions = sparse_vector_dimensions
        self._kwargs = kwargs

    def _get_client(self, collection: str) -> Milvus_:
        logger.debug(f"get client for collection: {collection}")
        client = Milvus_(
            collection=collection,
            embedder=self._embedder,
            uri=self._uri,
            token=self._token,
            search_type=self._search_type,
            reranker=self._reranker,
            sparse_vector_dimensions=self._sparse_vector_dimensions,
            **self._kwargs,
        )
        client.create()
        return client

    async def _async_get_client(self, collection: str) -> Milvus_:
        logger.debug(f"get client for collection: {collection}")
        client = Milvus_(
            collection=collection,
            embedder=self._embedder,
            uri=self._uri,
            token=self._token,
            search_type=self._search_type,
            reranker=self._reranker,
            sparse_vector_dimensions=self._sparse_vector_dimensions,
            **self._kwargs,
        )
        await client.async_create()
        return client
    
    def insert(
        self, 
        collection: str, 
        documents: List[Document], 
        filters: Optional[Dict[str, Any]] = None
    ) -> None:
        self._get_client(collection).insert(documents, filters)

    async def async_insert(
        self, 
        collection: str, 
        documents: List[Document], 
        filters: Optional[Dict[str, Any]] = None
    ) -> None:
        await self._async_get_client(collection).async_insert(documents, filters)

    def upsert(
        self, 
        collection: str, 
        documents: List[Document], 
        filters: Optional[Dict[str, Any]] = None
    ) -> None:
        self._get_client(collection).upsert(documents, filters)
    
    async def async_upsert(
        self, 
        collection: str, 
        documents: List[Document], 
        filters: Optional[Dict[str, Any]] = None,
        batch_size: int = 100
    ) -> None:
        logger.debug(f"async upsert {len(documents)} to collection[{collection}] with batch_size={batch_size}")
        client = await self._async_get_client(collection)
        
        # 批量处理 documents
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            logger.debug(f"processing batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size} with {len(batch)} documents")
            await client.async_upsert(batch, filters)

    async def async_search(
        self,
        collection: str,
        query: str, 
        limit: int = 5, 
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        logger.debug(f"async search query: {query} on collection[{collection}]")
        client = await self._async_get_client(collection)
        return await client.async_search(query, limit, filters)
