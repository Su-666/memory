"""
智谱 API 统一客户端 - 提取自 intent/answer/llm/vision 中重复的调用模式。
统一 headers 构造、错误处理、重试逻辑。
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Generator
from urllib import request as http_request
from urllib.error import HTTPError

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"


def _get_base_url() -> str:
    return os.getenv("LLM_BASE_URL", "").strip() or _DEFAULT_BASE_URL


def _get_api_key() -> str:
    return os.getenv("LLM_API_KEY", "").strip() or os.getenv("ZHIPU_API_KEY", "").strip()


def _get_model(env_key: str = "LOCAL_AGENT_MODEL", default: str = "glm-4-flash-250414") -> str:
    return os.getenv(env_key, default)


def _is_retryable(exc: Exception) -> bool:
    """判断异常是否值得重试（仅限瞬时错误）"""
    if isinstance(exc, HTTPError):
        return exc.code in (429, 500, 502, 503)
    if isinstance(exc, (TimeoutError, OSError, ConnectionError)):
        return True
    return False


def call_chat(
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 4096,
    tools: list[dict] | None = None,
    timeout: int = 90,
    retries: int = 1,
) -> dict[str, Any]:
    """调用智谱 Chat Completions API（带重试）。

    Returns:
        API 原始返回 dict（包含 choices 等字段）

    Raises:
        RuntimeError: API key 缺失
        Exception: 网络/解析错误（重试耗尽后）
    """
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("未配置 API Key（请在管理后台配置模型厂商）")

    model = model or _get_model()
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if tools:
        payload["tools"] = tools

    url = f"{_get_base_url()}/chat/completions"
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Connection": "keep-alive",
    }

    last_exc: Exception | None = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            req = http_request.Request(url, data=body, headers=headers, method="POST")
            with http_request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if "choices" not in data or not data["choices"]:
                logger.warning("智谱 API 返回无 choices: %s", data.get("error", data))
                raise RuntimeError(f"API 返回无 choices: {data.get('error', '')}")
            return data
        except Exception as exc:
            last_exc = exc
            if attempt < retries and _is_retryable(exc):
                time.sleep(2 ** (attempt - 1))
                continue
            raise

    raise last_exc  # type: ignore[misc]


def call_chat_stream(
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 4096,
    tools: list[dict] | None = None,
    timeout: int = 90,
) -> Generator[str, None, None]:
    """流式调用智谱 Chat Completions API，逐 token yield。

    Yields:
        每个 token 的文本片段

    Raises:
        RuntimeError: API key 缺失
        Exception: 网络/解析错误
    """
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("未配置 API Key（请在管理后台配置模型厂商）")

    model = model or _get_model()
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": True,
    }
    if tools:
        payload["tools"] = tools

    url = f"{_get_base_url()}/chat/completions"
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Connection": "keep-alive",
        "Accept": "text/event-stream",
    }

    req = http_request.Request(url, data=body, headers=headers, method="POST")
    with http_request.urlopen(req, timeout=timeout) as resp:
        buffer = ""
        while True:
            chunk = resp.read(1024)
            if not chunk:
                break
            buffer += chunk.decode("utf-8")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        return
                    try:
                        data = json.loads(data_str)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except (json.JSONDecodeError, IndexError, KeyError):
                        continue
