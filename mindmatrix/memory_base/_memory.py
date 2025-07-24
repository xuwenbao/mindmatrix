from typing import Optional, List

from agno.memory.v2.memory import UserMemory
from agno.memory.v2 import Memory as Memory_


class Memory(Memory_):

    def __init__(
        self, 
        *args, 
        exclude_topics: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.exclude_topics = exclude_topics
    
    def get_user_memories(
        self, 
        user_id: Optional[str] = None, 
        refresh_from_db: bool = True, 
    ) -> List[UserMemory]:
        """Get the user memories for a given user id"""
        if user_id is None:
            user_id = "default"
        # Refresh from the DB
        if refresh_from_db:
            self.refresh_from_db(user_id=user_id)

        if self.memories is None:
            return []
        memories = list(self.memories.get(user_id, {}).values())
        if self.exclude_topics:
            memories = [memory for memory in memories if not any(topic in self.exclude_topics for topic in memory.topics)]
        return memories