import os
import random
from typing import List, Callable, AsyncIterator

from agno.run.response import RunResponseContentEvent
from mindmatrix import (
    MindMatrix, 
    BaseAgent,
    BaseWorkflow,
    Step, 
    StepInput,
    StepOutput,
    OpenAILike,
)


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


def random_step(step_input: StepInput) -> AsyncIterator[StepOutput | RunResponseContentEvent]:
    if random.random() < 0.5:
        # 50%概率返回固定内容
        yield RunResponseContentEvent(content="我正在思考，请稍等...")
        # 停止执行
        yield StepOutput(content="这是一个固定内容", stop=True, success=True)
    else:
        # 50%概率返回上一级内容，继续执行
        yield StepOutput(content=step_input.previous_step_content, stop=False)


def create_workflow(mm: MindMatrix) -> BaseWorkflow:
    return BaseWorkflow(
        name="简单工作流",
        steps=[
            Step(name="random_step", description="随机步骤", executor=random_step),
            Step(name="chatter", description="闲聊", agent=mm.get_agent("chatter")),
        ],
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
    mm.register_workflow_factory(
        workflow_name="simple_workflow",
        workflow_factory=create_workflow,
    )

    mm.start_web_server(host="127.0.0.1", port=9527)