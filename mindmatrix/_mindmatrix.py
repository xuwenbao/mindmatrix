from dataclasses import dataclass
from typing import Union, Callable, List, Dict, Any

from fastapi import FastAPI
from agno.agent import Agent
from agno.models.base import Model
from agno.storage.base import Storage

from .web import create_app, AgentProvider


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


class MindMatrix:
    def __init__(
        self,
        *,
        # llm_client: Model = None,
        enable_builtins: Union[None, bool] = None,
        enable_plugins: Union[None, bool] = None,
        **kwargs,
    ):
        self._builtins_enabled = False
        self._plugins_enabled = False

        self._app: FastAPI = None
        # self._llm_client: Model = llm_client

        # Register the converters
        self._agent_factories: List[AgentRegistration] = []
        self._workflow_factories: List[WorkflowRegistration] = []

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
        ...

    def enable_plugins(self, **kwargs) -> None:
        ...

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
        workflow_config: Dict[str, Any],
    ) -> None:
        self._workflow_factories.append(
            WorkflowRegistration(name=workflow_name, workflow_factory=workflow_factory, workflow_config=workflow_config)
        )

    def get_agent(
        self,
        agent_name: str,
        **kwargs,
    ) -> Agent:
        for factory in self._agent_factories:
            if factory.name == agent_name:
                return factory.agent_factory(**factory.agent_config, **kwargs)
        raise ValueError(f"Agent factory for {agent_name} not found")
    
    def get_workflow(self, workflow_name: str) -> Callable:
        # for factory in self._workflow_factories:
        #     if factory.name == workflow_name:
        #         return factory.workflow_factory
        # raise ValueError(f"Workflow factory for {workflow_name} not found")
        ...
