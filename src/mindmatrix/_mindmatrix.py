import inspect
import traceback
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Union, Callable, List, Dict, Any, Optional

from loguru import logger
from fastapi import FastAPI
from prefect.tasks import Task
from agno.agent import Agent
from agno.workflow.v2 import Workflow
from agno.memory.v2 import Memory
from agno.models.base import Model
from agno.vectordb.base import VectorDb
from agno.embedder.openai import OpenAIEmbedder

from .builtins_.tasks import embed_documents
from .knowledge_base import VectorDb, VectorDbProvider
from .web import (
    create_app,
    AgentProvider,
    MemoryProvider,
    get_current_workflow,
    set_current_workflow,
)


@dataclass(kw_only=True, frozen=True)
class VectorDbRegistration:
    """A registration of a vector database with its name and factory."""
    name: str
    vector_db: VectorDb


@dataclass(kw_only=True, frozen=True)
class AgentRegistration:
    """A registration of a agent with its name and factory."""
    name: str
    agent_factory: Callable
    agent_config: Dict[str, Any]


@dataclass(kw_only=True, frozen=True)
class WorkflowRegistration:
    """A registration of a workflow with its name and factory."""
    name: str
    workflow_factory: Callable
    workflow_config: Dict[str, Any]


@dataclass(kw_only=True, frozen=True)
class TaskRegistration:
    """A registration of a task with its name and factory."""
    name: str
    task: Task


_plugins: Union[None, List[Any]] = None  # If None, plugins have not been loaded yet.


def _load_plugins() -> Union[None, List[Any]]:
    """Lazy load plugins, exiting early if already loaded."""
    global _plugins

    # Skip if we've already loaded plugins
    if _plugins is not None:
        return _plugins

    # Load plugins
    _plugins = []
    for entry_point in entry_points(group="mindmatrix.plugin"):
        try:
            logger.info(f"load plugin: {entry_point.name}")
            _plugins.append(entry_point.load())
        except Exception:
            tb = traceback.format_exc()
            logger.warning(f"Plugin '{entry_point.name}' failed to load ... skipping:\n{tb}")

    return _plugins


class MindMatrix:
    def __init__(
        self,
        *,
        llm: Union[Model, None] = None,
        memory: Union[Memory, None] = None,
        embedder: Union[OpenAIEmbedder, None] = None,
        vectordb: Union[VectorDb, None] = None,
        enable_builtins: Union[None, bool] = None,
        enable_plugins: Union[None, bool] = None,
        **kwargs,
    ):
        self._builtins_enabled = False
        self._plugins_enabled = False

        self._app: FastAPI = None

        self._llm = llm
        self._memory = memory
        self._embedder = embedder
        self._vectordb = vectordb

        # Register the converters
        self._agent_factories: List[AgentRegistration] = []
        self._workflow_factories: List[WorkflowRegistration] = []
        self._vectordbs: List[VectorDbRegistration] = []
        self._tasks: List[TaskRegistration] = []

        if (
            enable_builtins is None or enable_builtins
        ):  # Default to True when not specified
            self.enable_builtins(**kwargs)

        if enable_plugins:
            self.enable_plugins(**kwargs)

    @property
    def current_workflow(self) -> Optional[Workflow]:
        workflow = get_current_workflow()
        if workflow is None:
            logger.warning("current workflow is None, please set current workflow first")
        return workflow
    
    @current_workflow.setter
    def current_workflow(self, workflow: Workflow) -> None:
        assert workflow is not None, "can't set current workflow to None"
        
        logger.debug(f"setting current workflow: {workflow}")
        set_current_workflow(workflow)
    
    @property
    def app(self) -> FastAPI:
        if self._app is None:
            self._app = create_app(
                agent_provider=AgentProvider(self),
                memory_provider=MemoryProvider(self),
            )
        return self._app
    
    @property
    def llm(self) -> Model:
        return self._llm

    @property
    def memory(self) -> Memory:
        return self._memory

    @property
    def embedder(self) -> OpenAIEmbedder:
        return self._embedder

    def enable_builtins(self, **kwargs) -> None:
        self.register_task(task_name="embed_documents", task=embed_documents)

    def enable_plugins(self, **kwargs) -> None:
        if not self._plugins_enabled:
            # Load plugins
            plugins = _load_plugins()
            assert plugins is not None
            for plugin in plugins:
                try:
                    plugin.register_plugin(self, **kwargs)
                except Exception:
                    tb = traceback.format_exc()
                    logger.warning(f"Plugin '{plugin}' failed to register plugin:\n{tb}")
            self._plugins_enabled = True
        else:
            logger.warning("Plugins are already enabled.")

    def has_vectordb(self, vectordb_name: str) -> bool:
        return any(registration.name == vectordb_name for registration in self._vectordbs)
    
    def has_agent(self, agent_name: str) -> bool:
        return any(registration.name == agent_name for registration in self._agent_factories)
    
    def has_workflow(self, workflow_name: str) -> bool:
        return any(registration.name == workflow_name for registration in self._workflow_factories)

    def register_vectordb(
        self,
        vectordb_name: str,
        vectordb: VectorDb,
    ) -> None:
        if self.has_vectordb(vectordb_name):
            logger.warning(f"Vector database for {vectordb_name} already registered.")

        self._vectordbs.append(VectorDbRegistration(name=vectordb_name, vector_db=vectordb))

    def register_agent_factory(
        self,
        agent_name: str,
        agent_factory: Callable,
        agent_config: Dict[str, Any],
    ) -> None:
        if self.has_agent(agent_name):
            logger.warning(f"Agent factory for {agent_name} already registered.")
            
        self._agent_factories.append(
            AgentRegistration(name=agent_name, agent_factory=agent_factory, agent_config=agent_config)
        )

    def register_workflow_factory(
        self,
        workflow_name: str,
        workflow_factory: Callable,
        workflow_config: Dict[str, Any] = None,
    ) -> None:
        if self.has_workflow(workflow_name):
            logger.warning(f"Workflow factory for {workflow_name} already registered.")
            
        if workflow_config is None:
            workflow_config = {}
        
        self._workflow_factories.append(
            WorkflowRegistration(name=workflow_name, workflow_factory=workflow_factory, workflow_config=workflow_config)
        )

    def register_task(
        self,
        task_name: str,
        task: Task,
    ) -> None:
        self._tasks.append(TaskRegistration(name=task_name, task=task))

    def get_vectordb(
        self,
        vectordb_name: str,
    ) -> VectorDb:
        for registration in self._vectordbs:
            if registration.name == vectordb_name:
                return registration.vector_db
        raise ValueError(f"Vector database for {registration.name} not found")

    def get_agent(
        self,
        agent_name: str,
        **kwargs,
    ) -> Agent:
        for registration in self._agent_factories:
            if registration.name == agent_name:
                return registration.agent_factory(self, **registration.agent_config, **kwargs) # TODO: 动态注入
        raise ValueError(f"Agent factory for {agent_name} not found")
    
    def get_workflow(
        self,
        workflow_name: str,
        **kwargs,
    ) -> Workflow:
        for registration in self._workflow_factories:
            if registration.name == workflow_name:
                return registration.workflow_factory(self, **registration.workflow_config, **kwargs) # TODO: 动态注入
        raise ValueError(f"Workflow factory for {workflow_name} not found")

    def get_task(
        self,
        task_name: str,
    ) -> Task:
        for registration in self._tasks:
            if registration.name == task_name:
                return registration.task
        raise ValueError(f"Task for {task_name} not found")
    
    # @flow
    async def async_run_task(
        self,
        task_name: str,
        *args,
        **kwargs,
    ) -> Any:
        task = self.get_task(task_name)
        logger.debug(f"* Running task: {task_name}")
        # 获取task函数的参数信息
        sig = inspect.signature(task.fn)
        
        # 检查参数中是否包含vectordb_provider
        if 'vectordb_provider' in sig.parameters:
            kwargs['vectordb_provider'] = VectorDbProvider(self)

        if 'agent_provider' in sig.parameters:
            kwargs['agent_provider'] = AgentProvider(self)
        
        return await task.fn(*args, **kwargs)

    def start_web_server(
        self,
        host: str = "127.0.0.1",
        port: int = 9527,
        log_level: str = "debug",
        **kwargs,
    ) -> None:
        try:
            import uvicorn
        except ImportError:
            logger.error("uvicorn 未安装，请运行: pip install uvicorn")
            return

        logger.info(f"启动 MindMatrix Web 服务器...")
        logger.info(f"服务器地址: http://{host}:{port}")
        logger.info(f"API 文档: http://{host}:{port}/docs")
        logger.info(f"交互式文档: http://{host}:{port}/redoc")
        
        try:
            uvicorn.run(self.app, host=host, port=port, log_level=log_level, **kwargs)
        except KeyboardInterrupt:
            logger.info("服务器已停止")
        except Exception as e:
            logger.error(f"启动服务器时发生错误: {e}")
            raise