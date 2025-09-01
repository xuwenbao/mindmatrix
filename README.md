# MindMatrix(灵枢)

> 此项目目前还处于开发测试中，请等待项目正式发布 ...

## MindMatrix 是什么？

MindMatrix 是一个基于 Agno 的轻量级智能体"应用"框架，专为快速构建具有RAG、记忆和可访问能力的生产级多智能体系统而设计。

## 快速开始

TODO

## 为什么使用 MindMatrix？

MindMatrix 将帮助您构建一流的、高性能的智能体系统，节省数小时的研究和样板代码时间。以下是 MindMatrix 的关键特性：

* **智能体插件系统**：智能体即插即用，以插件形式注册智能体（Agent）、工作流（Workflow）
* **智能体接口**：插件注册后即可访问，通过 FastAPI 实现，预置 SSE、OpenAI 兼容等形式的智能体接口
* **RAG 功能增强**：内置 RAG 功能增强代码，更好中文场景、中文模型支持
* **长期记忆功能增强**：内置长期记忆功能增强代码，更好中文场景长期记忆提取

## 安装

```bash
pip install -U mindmatrix
```

## 示例 - 实现一个简单智能体

让我们构建一个智能体来感受 MindMatrix 的功能。MindMatrix智能体是基于[Agno Agent](https://docs.agno.com/agents/introduction)实现的。

完整代码可访问：[simple_agent.py](examples/simple_agent.py)

```python:examples/simple_agent.py
import os
from typing import List, Callable

from mindmatrix import MindMatrix, BaseAgent, OpenAILike


# 智能体的角色
CHATTER_DESCRIPTION = """
你是一个智能协作助手，名字叫做Chatter，是用户请求的最终回复者。
"""


# 智能体的目标(任务)
CHATTER_GOAL = """
当用户的请求没有匹配到其他协作者时，你会接收到这个请求，你需要回复用户。
"""


# 智能体的指令（约束）
CHATTER_INSTRUCTIONS = [
    "语言精炼，不要啰嗦", 
    "回答符合人类交流的语气，不能包含表情",
]


def create_chatter(
    mm: MindMatrix,
    model: OpenAILike,
    name: str = "闲聊",
    description: str = None,
    goal: str = None,
    instructions: List[str] = None,
    tools: List[Callable] = None,
    show_tool_calls: bool = False,
    **kwargs
) -> BaseAgent:
    if description is None:
        description = CHATTER_DESCRIPTION
    if goal is None:
        goal = CHATTER_GOAL
    if instructions is None:
        instructions = CHATTER_INSTRUCTIONS
    
    return BaseAgent(
        model=model,
        name=name,
        description=description,
        goal=goal,
        instructions=instructions,
        tools=tools,
        show_tool_calls=show_tool_calls,
        **kwargs
    )


if __name__ == "__main__":
    # 请确保已设置以下环境变量，否则 OpenAILike 无法正常工作：
    # OPENAI_API_MODEL: 你的模型ID，例如 "gpt-4o-mini"
    # OPENAI_API_KEY: 你的OpenAI API密钥
    # OPENAI_API_BASEURL: OpenAI API的Base URL（如使用官方API可不设置，第三方代理需设置）
    # 
    # 设置方法（Linux/macOS）:
    # export OPENAI_API_MODEL="gpt-4o-mini"
    # export OPENAI_API_KEY="sk-xxxx"
    # export OPENAI_API_BASEURL="https://api.openai.com/v1"
    #
    # 设置方法（Windows CMD）:
    # set OPENAI_API_MODEL=gpt-4o-mini
    # set OPENAI_API_KEY=sk-xxxx
    # set OPENAI_API_BASEURL=https://api.openai.com/v1

    llm = OpenAILike(
        id=os.getenv("OPENAI_API_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASEURL", "https://api.openai.com/v1"),
    )
    mm = MindMatrix(llm=llm)
    mm.register_agent_factory(
        agent_name="chatter",
        agent_factory=create_chatter,
        agent_config={
            "name": "chatter",
            "model": mm.llm,
            "debug_mode": True,
        },
    )

    mm.start_web_server(host="127.0.0.1", port=9527)
```

然后创建虚拟环境，安装依赖并运行智能体：

```bash
uv venv --python 3.12
source .venv/bin/activate

uv pip install mindmatrix

export OPENAI_API_MODEL="gpt-4o-mini"
export OPENAI_API_KEY="sk-xxxx"

python simple_agent.py
```

## 示例 - 实现一个简单工作流

让我们基于上面的chatter闲聊智能体，继续构建一个简单的工作流(Workflow)。MindMatrix工作流是基于[Agno Workflow(v2)](https://docs.agno.com/workflows_2/overview)实现的。

在chatter闲聊智能体之前，我们添加一个50%概率返回固定内容的前置工作流步骤（Step），并将chatter闲聊智能体做为工作流的第二个步骤。

首先，新增导入：

```python
from mindmatrix import (
    MindMatrix, 
    BaseAgent,
    BaseWorkflow,
    Step, 
    StepInput,
    StepOutput,
    OpenAILike,
)
```

然后，定义前置步骤函数（步骤可以是Agent，也可以是函数）:

```python
def random_step(step_input: StepInput) -> AsyncIterator[StepOutput | RunResponseContentEvent]:
    if random.random() < 0.5:
        # 50%概率返回固定内容
        yield RunResponseContentEvent(content="我正在思考，请稍等...")
        # 停止执行
        yield StepOutput(content="这是一个固定内容", stop=True, success=True)
    else:
        # 50%概率返回上一级内容，继续执行
        yield StepOutput(content=step_input.previous_step_content, stop=False)
```

然后，定义工作流：

```python
def create_workflow(mm: MindMatrix) -> BaseWorkflow:
    return BaseWorkflow(
        name="简单工作流",
        steps=[
            Step(name="random_step", description="随机步骤", executor=random_step),    # 前置步骤
            Step(name="chatter", description="闲聊", agent=mm.get_agent("chatter")),  # 闲聊智能体
        ],
    )
```

最后，修改最终执行代码：

```python
if __name__ == "__main__":
    ...

    # 新增注册workflow
    mm.register_workflow_factory(
            workflow_name="simple_workflow",
            workflow_factory=create_workflow,
        )

    ...

    mm.start_web_server(host="127.0.0.1", port=9527)
```

完整代码可访问：[simple_workflow.py](examples/simple_workflow.py)

代码修改完成后，安装依赖并运行工作流：

```bash
uv venv --python 3.12
source .venv/bin/activate

uv pip install mindmatrix

export OPENAI_API_MODEL="gpt-4o-mini"
export OPENAI_API_KEY="sk-xxxx"

python simple_workflow.py
```

## 示例 - 实现一个简单插件

请参照项目：[MindMatrix Sample Plugin](https://github.com/xuwenbao/mindmatrix-sample-plugin)

## 项目结构

```
mindmatrix/
├── src/mindmatrix/
│   ├── _mindmatrix.py          # MindMatrix 核心类
│   ├── agent_base/             # 智能体基类
│   ├── knowledge_base/         # 知识库组件
│   ├── memory_base/            # 记忆管理
│   ├── web/                    # Web API 实现
│   └── utils/                  # 工具类
├── docs/                       # 文档
└── pyproject.toml             # 项目配置
```


## 文档、社区和更多示例

TODO
