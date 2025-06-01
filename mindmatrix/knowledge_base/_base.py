from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Literal

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
    ...


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
        return Milvus_(
            collection=collection,
            embedder=self._embedder,
            uri=self._uri,
            token=self._token,
            search_type=self._search_type,
            reranker=self._reranker,
            sparse_vector_dimensions=self._sparse_vector_dimensions,
            **self._kwargs,
        )
    
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
        await self._get_client(collection).async_insert(documents, filters)

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
        filters: Optional[Dict[str, Any]] = None
    ) -> None:
        await self._get_client(collection).async_upsert(documents, filters)
