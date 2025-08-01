from typing import Literal

from loguru import logger
from fastapi.responses import JSONResponse
from fastapi import FastAPI, Request, Query
from fastapi.middleware.cors import CORSMiddleware

from ._endpoints import router as api_router


class AgentProvider:
    def __init__(self, mindmatrix):
        self.mindmatrix = mindmatrix

    def __call__(
        self,
        agent_name: str = Query(...),
        type_: Literal["agent", "workflow"] = "agent",
        **kwargs,
    ):
        if type_ == "agent":
            return self.mindmatrix.get_agent(agent_name, **kwargs)
        elif type_ == "workflow":
            return self.mindmatrix.get_workflow(agent_name, **kwargs)
        else:
            raise ValueError(f"Invalid type: {type_}, must be 'agent' or 'workflow'")
        

class MemoryProvider:
    def __init__(self, mindmatrix):
        self.mindmatrix = mindmatrix

    def __call__(
        self,
        **kwargs,
    ):
        return self.mindmatrix.memory


def create_app(agent_provider: AgentProvider, memory_provider: MemoryProvider):
    """Create the FastAPI app and include the router."""
    app = FastAPI(
        title="mindmatrix",
        description="A FastAPI app for mindmatrix",
        version="0.0.1",
    )

    origins = [
        '*',
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=['*'],
        allow_headers=['*'],
    )

    @app.exception_handler(Exception)
    async def catch_exceptions_middleware(_request: Request, exc: Exception):
        # logger.exception(exc)
        return JSONResponse(
            content={"error": f"An unexpected error occurred: {str(exc)}"}, status_code=500
        )

    @app.get('/health')
    def get_health():
        return JSONResponse(content={"status": "OK"})
    
    logger.debug(f"setting router agent_provider: {agent_provider}")
    logger.debug(f"setting router memory_provider: {memory_provider}")
    api_router.agent_provider = agent_provider
    api_router.memory_provider = memory_provider
    app.include_router(router=api_router)

    return app
