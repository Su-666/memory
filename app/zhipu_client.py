"""
智谱 API 统一客户端 - 提取自 intent/answer/llm/vision 中重复的调用模式。
统一 headers 构造、错误处理、重试逻辑。
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any
from urllib import request as http_request

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"


def _get_base_url() -> str:
    return os.getenv("LLM_BASE_URL", "").strip() or _DEFAULT_BASE_URL


def _get_api_key() -> str:
    return os.getenv("LLM_API_KEY", "").strip() or os.getenv("ZHIPU_API_KEY", "").strip()


def _get_model(env_key: str = "LOCAL_AGENT_MODEL", default: str = "glm-4-flash-250414") -> str:
    return os.getenv(env_key, default)


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
            if attempt < retries:
                time.sleep(2 ** (attempt - 1))
                continue
            raise

    raise last_exc  # type: ignore[misc]
