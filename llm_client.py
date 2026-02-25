"""
Async LLMClient for OpenAI-compatible llama-server (GPU offloading).
Handles HTTP requests to local LLM server (http://localhost:8080/v1).
"""
import asyncio
from typing import Any, Dict, Optional
import httpx

class LLMClient:
    def __init__(self, base_url: str = "http://localhost:8080/v1", api_key: str = "not-needed", timeout: float = 60.0, max_retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    async def _request(self, method: str, endpoint: str, json: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.base_url}{endpoint}"
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.request(method, url, headers=self.headers, json=json)
                    resp.raise_for_status()
                    return resp.json()
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        payload = {"model": "llama", "prompt": prompt}
        payload.update(kwargs)
        return await self._request("POST", "/chat/completions", json=payload)

    async def embeddings(self, input_text: str, **kwargs) -> Dict[str, Any]:
        payload = {"model": "llama", "input": input_text}
        payload.update(kwargs)
        return await self._request("POST", "/embeddings", json=payload)
