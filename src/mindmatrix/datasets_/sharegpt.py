from typing import Dict


def extract_first_human_message(example, key: str = "first_human_message") -> Dict[str, str]:
    """提取对话中第一条human消息的value"""
    conversations = example["conversations"]
    
    # 找到第一条human消息
    for msg in conversations:
        if msg["from"] == "human":
            return {key: msg["value"]}
    
    # 如果没有找到human消息，返回空字符串
    return {key: ""}