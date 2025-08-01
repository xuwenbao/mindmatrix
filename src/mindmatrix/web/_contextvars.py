import contextvars

from loguru import logger

# 创建上下文变量
session_id_var = contextvars.ContextVar("session_id", default=None)
jwt_token_var = contextvars.ContextVar("jwt_token", default=None)
workflow_var = contextvars.ContextVar("workflow", default=None)


def get_current_session_id():
    return session_id_var.get()


def set_current_session_id(session_id: str):
    session_id_var.set(session_id)


def get_current_jwt_token():
    """获取当前上下文的 JWT token"""
    return jwt_token_var.get()


def set_current_jwt_token(token: str):
    """设置当前上下文的 JWT token
    
    Args:
        token: JWT token 字符串
    """
    logger.debug(f"set current jwt token: {token}")
    jwt_token_var.set(token)


def get_current_workflow():
    return workflow_var.get()


def set_current_workflow(workflow):
    workflow_var.set(workflow)

