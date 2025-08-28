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

让我们构建一个智能体来感受 MindMatrix 的功能。

将此代码保存到文件：`agent_factory.py`

```python:examples/agent_factory.py
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

python agent_factory.py
```

## 示例 - 实现一个简单工作流

TODO

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
