from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from textwrap import dedent
from agno.agent import Agent
from agno.models.message import Message
from agno.memory.v2.db import MemoryDb
from agno.memory.v2.manager import MemoryManager


@dataclass
class MindmatrixMemoryManager(MemoryManager):

    def get_system_message(
        self,
        existing_memories: Optional[List[Dict[str, Any]]] = None,
        enable_delete_memory: bool = True,
        enable_clear_memory: bool = True,
    ) -> Message:
        if self.system_message is not None:
            return Message(role="system", content=self.system_message)

        # memory_capture_instructions = self.memory_capture_instructions or dedent("""\
        #     记忆应该包含能够个性化用户持续交互的详细信息，例如：
        #       - 个人信息：姓名、年龄、职业、位置、兴趣、偏好等
        #       - 用户分享的重要生活事件或经历
        #       - 关于用户当前情况、挑战或目标的重要背景信息
        #       - 用户喜欢或不喜欢的事物，他们的观点、信念、价值观等
        #       - 任何其他能够提供用户个性、观点或需求有价值洞察的详细信息\
        # """)

        memory_capture_instructions = self.memory_capture_instructions or dedent("""\
            记忆应该包含能够个性化用户持续交互的详细信息，例如：
              - 个人信息：姓名、年龄、职业、位置、兴趣、偏好等
              - 用户分享的重要生活事件或经历
              - 用户喜欢或不喜欢的事物，他们的观点、信念、价值观等
              - 任何其他能够提供用户个性、观点或需求有价值洞察的详细信息\
        """)

        # -*- 返回记忆管理器的系统消息
        system_prompt_lines = [
            "你是一个记忆管理器，负责管理关于用户的关键信息。 "
            "你将在<memories_to_capture>部分获得记忆捕获的标准，在<existing_memories>部分获得现有记忆列表。",
            "",
            "## 何时添加或更新记忆",
            "- 你的第一个任务是决定是否需要根据用户的消息添加、更新或删除记忆，或者不需要任何更改。",
            "- 如果用户的消息符合<memories_to_capture>部分的标准，并且该信息尚未在<existing_memories>部分中捕获，你应该将其捕获为记忆。",
            "- 如果用户的消息不符合<memories_to_capture>部分的标准，则不需要记忆更新。",
            "- 如果<existing_memories>部分中的现有记忆捕获了所有相关信息，则不需要记忆更新。",
            "",
            "## 如何添加或更新记忆",
            "- 如果你决定添加新记忆，创建能够捕获关键信息的记忆，就像你正在为将来参考而存储它一样。",
            "- 记忆应该是简短的第三人称陈述，封装用户输入的最重要方面，不添加任何无关信息。",
            "  - 示例：如果用户的消息是'我要去健身房'，记忆可以是`用户定期去健身房`。",
            "  - 示例：如果用户的消息是'我的名字是张三'，记忆可以是`用户的名字是张三`。",
            "- 不要让单个记忆太长或太复杂，如果需要捕获所有信息，请创建多个记忆。",
            "- 不要在多个记忆中重复相同的信息。如果需要，请更新现有记忆。",
            "- 如果用户要求更新或忘记记忆，请删除所有应该被遗忘的信息的引用。不要说'用户过去喜欢...`",
            "- 更新记忆时，将新信息附加到现有记忆中，而不是完全覆盖它。",
            "- 当用户的偏好发生变化时，更新相关记忆以反映新的偏好，但也要捕获用户过去的偏好以及发生了什么变化。",
            "",
            "## 创建记忆的标准",
            "使用以下标准来确定是否应该将用户的消息捕获为记忆。",
            "",
            "<memories_to_capture>",
            memory_capture_instructions,
            "</memories_to_capture>",
            "",
            "## 更新记忆",
            "你还将在<existing_memories>部分获得现有记忆列表。你可以：",
            "  1. 决定不进行任何更改。",
            "  2. 决定添加新记忆，使用`add_memory`工具。",
            "  3. 决定更新现有记忆，使用`update_memory`工具。",
        ]
        if enable_delete_memory:
            system_prompt_lines.append("  4. 决定删除现有记忆，使用`delete_memory`工具。")
        if enable_clear_memory:
            system_prompt_lines.append("  5. 决定清除所有记忆，使用`clear_memory`工具。")
        system_prompt_lines += [
            "如果需要，你可以在单个响应中调用多个工具。 ",
            "只有在需要捕获用户提供的关键信息时才添加或更新记忆。",
        ]

        if existing_memories and len(existing_memories) > 0:
            system_prompt_lines.append("\n<existing_memories>")
            for existing_memory in existing_memories:
                system_prompt_lines.append(f"ID: {existing_memory['memory_id']}")
                system_prompt_lines.append(f"Memory: {existing_memory['memory']}")
                system_prompt_lines.append("")
            system_prompt_lines.append("</existing_memories>")

        if self.additional_instructions:
            system_prompt_lines.append(self.additional_instructions)

        return Message(role="system", content="\n".join(system_prompt_lines))
    
    def build_agent(
        self,
        messages: Optional[List[Message]] = None,
        user_id: str = None,
        db: MemoryDb = None,
        existing_memories: Optional[List[Dict[str, Any]]] = None,
        delete_memories: bool = True,
        clear_memories: bool = True,
    ) -> Agent:
        # if messages is None:
        #     input_string = ""
        # elif len(messages) == 1:
        #     input_string = messages[0].get_content_string()
        # else:
        #     input_string = f"{', '.join([m.get_content_string() for m in messages if m.role == 'user' and m.content])}"
        
        return Agent(
            model=self.model,
            system_message=self.get_system_message(existing_memories, delete_memories, clear_memories),
            tools=self._get_fake_db_tools(),
            tool_call_limit=1,
            debug_mode=True,
        )
    
    def _get_fake_db_tools(
        self,
        enable_add_memory: bool = True,
        enable_update_memory: bool = True,
        enable_delete_memory: bool = True,
        enable_clear_memory: bool = True,
    ) -> List[Callable]:

        def add_memory(memory: str, topics: Optional[List[str]] = None) -> str:
            """Use this function to add a memory to the database.
            Args:
                memory (str): The memory to be added.
                topics (Optional[List[str]]): The topics of the memory (e.g. ["name", "hobbies", "location"]).
            Returns:
                str: A message indicating if the memory was added successfully or not.
            """
            return "Memory added successfully"

        def update_memory(memory_id: str, memory: str, topics: Optional[List[str]] = None) -> str:
            """Use this function to update an existing memory in the database.
            Args:
                memory_id (str): The id of the memory to be updated.
                memory (str): The updated memory.
                topics (Optional[List[str]]): The topics of the memory (e.g. ["name", "hobbies", "location"]).
            Returns:
                str: A message indicating if the memory was updated successfully or not.
            """
            return "Memory updated successfully"

        def delete_memory(memory_id: str) -> str:
            """Use this function to delete a single memory from the database.
            Args:
                memory_id (str): The id of the memory to be deleted.
            Returns:
                str: A message indicating if the memory was deleted successfully or not.
            """
            return "Memory deleted successfully"

        def clear_memory() -> str:
            """Use this function to remove all (or clear all) memories from the database.

            Returns:
                str: A message indicating if the memory was cleared successfully or not.
            """
            return "Memory cleared successfully"

        functions: List[Callable] = []
        if enable_add_memory:
            functions.append(add_memory)
        if enable_update_memory:
            functions.append(update_memory)
        if enable_delete_memory:
            functions.append(delete_memory)
        if enable_clear_memory:
            functions.append(clear_memory)
        return functions