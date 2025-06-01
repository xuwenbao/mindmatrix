from typing import List

from agno.document import Document
from prefect import task, get_run_logger

from ..knowledge_base import VectorDbProvider


@task(log_prints=True)
async def embed_documents(
    vectordb_name: str,
    collection_name: str,
    documents: List[Document],
    *,
    vectordb_provider: VectorDbProvider = None, # TODO: 优化依赖注入参数机制
) -> None:
    logger = get_run_logger()
    logger.info(f"* using collection: {vectordb_name}")

    vectordb = vectordb_provider(vectordb_name)
    await vectordb.async_upsert(collection_name, documents)