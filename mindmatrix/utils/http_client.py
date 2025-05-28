import httpx
from loguru import logger
from urllib.parse import urljoin


class AsyncHttpClient:
    def __init__(self, base_url=None, headers=None, timeout=10):
        self.base_url = base_url
        self.headers = headers or {}
        self.timeout = timeout
        self.client = httpx.AsyncClient(base_url=base_url, headers=self.headers, timeout=self.timeout)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self._close()

    def _build_url(self, path):
        if self.base_url:
            return urljoin(self.base_url.rstrip('/') + '/', path.lstrip('/'))
        return path

    async def _request(self, method, url, **kwargs):
        try:
            response = await self.client.request(method, url, **kwargs)
            response.raise_for_status()
            try:
                return {"status": response.status_code, "data": response.json()}
            except Exception:
                return {"status": response.status_code, "data": response.text}
        except httpx.HTTPStatusError as exc:
            return {"status": exc.response.status_code, "error": str(exc)}
        except httpx.RequestError as exc:
            return {"status": None, "error": str(exc)}

    async def _get(self, path, params=None, **kwargs):
        url = self._build_url(path)
        logger.info(f"GET {url} with params: {params}")
        return await self._request("GET", url, params=params, **kwargs)

    async def _post(self, path, data=None, json=None, **kwargs):
        url = self._build_url(path)
        return await self._request("POST", url, data=data, json=json, **kwargs)

    async def _put(self, path, data=None, **kwargs):
        url = self._build_url(path)
        return await self._request("PUT", url, data=data, **kwargs)

    async def _delete(self, path, **kwargs):
        url = self._build_url(path)
        return await self._request("DELETE", url, **kwargs)

    async def _close(self):
        await self.client.aclose()


class SyncHttpClient:
    def __init__(self, base_url=None, headers=None, timeout=10):
        self.base_url = base_url
        self.headers = headers or {}
        self.timeout = timeout
        self.client = httpx.Client(base_url=base_url, headers=self.headers, timeout=self.timeout)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self._close()

    def _build_url(self, path):
        if self.base_url:
            return urljoin(self.base_url.rstrip('/') + '/', path.lstrip('/'))
        return path

    def _request(self, method, url, **kwargs):
        try:
            response = self.client.request(method, url, **kwargs)
            response.raise_for_status()
            try:
                return {"status": response.status_code, "data": response.json()}
            except Exception:
                return {"status": response.status_code, "data": response.text}
        except httpx.HTTPStatusError as exc:
            return {"status": exc.response.status_code, "error": str(exc)}
        except httpx.RequestError as exc:
            return {"status": None, "error": str(exc)}

    def _get(self, path, params=None, **kwargs):
        url = self._build_url(path)
        logger.info(f"GET {url} with params: {params}")
        return self._request("GET", url, params=params, **kwargs)

    def _post(self, path, data=None, json=None, **kwargs):
        url = self._build_url(path)
        return self._request("POST", url, data=data, json=json, **kwargs)

    def _put(self, path, data=None, **kwargs):
        url = self._build_url(path)
        return self._request("PUT", url, data=data, **kwargs)

    def _delete(self, path, **kwargs):
        url = self._build_url(path)
        return self._request("DELETE", url, **kwargs)

    def _close(self):
        self.client.close()

# 使用示例
if __name__ == "__main__":
    import asyncio
    async def main():
        async with AsyncHttpClient(base_url="https://httpbin.org") as client:
            result_get = await client._get("/get", params={"foo": "bar"})
            print("GET:", result_get)
            result_post = await client._post("/post", json={"hello": "world"})
            print("POST:", result_post)
            result_put = await client._put("/put", data={"key": "value"})
            print("PUT:", result_put)
            result_delete = await client._delete("/delete")
            print("DELETE:", result_delete)
    asyncio.run(main())

    # 同步用法示例
    with SyncHttpClient(base_url="https://httpbin.org") as client:
        result_get = client._get("/get", params={"foo": "bar"})
        print("[Sync] GET:", result_get)
        result_post = client._post("/post", json={"hello": "world"})
        print("[Sync] POST:", result_post)
        result_put = client._put("/put", data={"key": "value"})
        print("[Sync] PUT:", result_put)
        result_delete = client._delete("/delete")
        print("[Sync] DELETE:", result_delete)
