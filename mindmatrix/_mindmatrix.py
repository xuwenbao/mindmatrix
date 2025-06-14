import inspect
import traceback
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Union, Callable, List, Dict, Any

from loguru import logger
from fastapi import FastAPI
from prefect import flow
from prefect.tasks import Task
from agno.agent import Agent
from agno.workflow import Workflow

from .tasks import embed_documents
from .web import create_app, AgentProvider
from .knowledge_base import VectorDb, VectorDbProvider


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
        enable_builtins: Union[None, bool] = None,
        enable_plugins: Union[None, bool] = None,
        **kwargs,
    ):
        self._builtins_enabled = False
        self._plugins_enabled = False

        self._app: FastAPI = None

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
    def app(self) -> FastAPI:
        if self._app is None:
            self._app = create_app(
                agent_provider=AgentProvider(self)
            )
        return self._app

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

    def register_vectordb(
        self,
        vectordb_name: str,
        vectordb: VectorDb,
    ) -> None:
        self._vectordbs.append(VectorDbRegistration(name=vectordb_name, vector_db=vectordb))

    def register_agent_factory(
        self,
        agent_name: str,
        agent_factory: Callable,
        agent_config: Dict[str, Any],
    ) -> None:
        self._agent_factories.append(
            AgentRegistration(name=agent_name, agent_factory=agent_factory, agent_config=agent_config)
        )

    def register_workflow_factory(
        self,
        workflow_name: str,
        workflow_factory: Callable,
        workflow_config: Dict[str, Any] = None,
    ) -> None:
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
                return registration.agent_factory(**registration.agent_config, **kwargs)
        raise ValueError(f"Agent factory for {agent_name} not found")
    
    def get_workflow(
        self,
        workflow_name: str,
        **kwargs,
    ) -> Workflow:
        for registration in self._workflow_factories:
            if registration.name == workflow_name:
                return registration.workflow_factory(**registration.workflow_config)
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
            # 创建VectorDbProvider实例
            vectordb_provider = VectorDbProvider(self)
            kwargs['vectordb_provider'] = vectordb_provider
        
        return await task.fn(*args, **kwargs)