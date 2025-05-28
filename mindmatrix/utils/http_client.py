import httpx
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
            return urljoin(self.base_url, path)
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
