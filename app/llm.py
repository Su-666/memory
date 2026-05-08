"""
统一 LLM 聊天调用。
桌面端和 Web 端共用此模块，确保参数、提示词、温度一致。
"""
from __future__ import annotations

import json
import logging
import os
from urllib import request as http_request

logger = logging.getLogger(__name__)


# 默认系统提示词
DEFAULT_SYSTEM_PROMPT = (
    "你是暖暖，一位贴心、稳重、值得信赖的私人助手，也是用户的老朋友。"
    "你面向的是成年人和长辈用户，说话要自然、亲切、有分寸，不卖萌、不撒娇、不装幼稚。"
    "像生活中一位知性温和的朋友那样交流：简洁明了、不急不躁，用平实温暖的口语，不用网络梗和过度活泼的语气。"
    "不要客套，不用您好/请问/抱歉/谢谢理解这种客服腔。"
    "也不要提自己是人工智能、大模型或者AI。"
    "回复长短随场景而定：日常问候简短自然；用户让你讲故事、写作文、详细讲解、写代码等，就充分展开，把内容说完整。"
)


def call_llm_chat(
    user_query: str,
    history: list[dict],
    *,
    system_prompt: str | None = None,
    enable_web_search: bool = True,
) -> str | None:
    """调用智谱大模型对话。

    Args:
        user_query: 用户输入
        history: 对话历史 [{"role": "user"/"assistant", "content": "..."}]
        system_prompt: 系统提示词，默认使用暖暖人设
        enable_web_search: 是否启用联网搜索

    Returns:
        模型回复文本，失败返回 None
    """
    api_key = os.getenv("ZHIPU_API_KEY", "").strip()
    if not api_key:
        return None

    base_url = "https://open.bigmodel.cn/api/paas/v4"
    model = os.getenv("LOCAL_AGENT_MODEL", "GLM-4-Flash-250414")

    sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    messages = [{"role": "system", "content": sys_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_query})

    payload: dict = {
        "model": model,
        "messages": messages,
        "temperature": 0.85,
        "top_p": 0.92,
        "max_tokens": 4096,
    }

    if enable_web_search:
        payload["tools"] = [{"type": "web_search", "web_search": {"enable": True}}]

    try:
        req = http_request.Request(
            f"{base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with http_request.urlopen(req, timeout=90) as response:
            data = json.loads(response.read().decode("utf-8"))

        if "choices" not in data or not data["choices"]:
            logger.warning("LLM 返回无 choices: %s", data.get("error", data))
            return None

        message = data["choices"][0]["message"]

        # 处理工具调用（联网搜索）
        if "tool_calls" in message and enable_web_search:
            messages.append(message)
            for tool_call in message["tool_calls"]:
                func = tool_call.get("function", {})
                if func.get("name") == "web_search":
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": "联网搜索已完成",
                    })

            req2 = http_request.Request(
                f"{base_url.rstrip('/')}/chat/completions",
                data=json.dumps({
                    "model": model,
                    "messages": messages,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 4096,
                }, ensure_ascii=False).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with http_request.urlopen(req2, timeout=90) as response:
                data2 = json.loads(response.read().decode("utf-8"))
            if "choices" not in data2 or not data2["choices"]:
                logger.warning("LLM 第二次调用返回无 choices")
                return None
            final_message = data2["choices"][0]["message"]
            content = final_message.get("content", "")
            return str(content).strip() if content else None

        content = message.get("content", "")
        return str(content).strip() if content else None

    except Exception as e:
        logger.warning("智谱大模型对话失败: %s", e)
        return None
