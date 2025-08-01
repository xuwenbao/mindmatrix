from loguru import logger
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader, HTTPBearer

bearer_scheme = HTTPBearer(auto_error=False)


async def get_api_key(api_key: str = Security(APIKeyHeader(name='api-key', auto_error=False))):
    logger.info(f"get api_key: {api_key}")
    if api_key in settings.server.api_keys:
        return api_key
    else:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")


async def get_bearer_token(token: str = Security(bearer_scheme)):
    """获取并验证 Bearer token
    
    Args:
        token: Bearer token 字符串
        
    Returns:
        str: token 字符串
        
    Raises:
        HTTPException: 当 token 无效或缺失时抛出
    """
    if not token:
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    logger.info(f"get bearer token: {token.credentials}")
    return token.credentials
