from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .utils import extract_text, load_env_file, parse_json_block
from .zhipu_client import call_chat

logger = logging.getLogger(__name__)


@dataclass
class AssistantPlan:
    action: str  # "search" | "save"
    reason: str
    note_title: str = ""
    note_content: str = ""


def _fallback_plan(user_query: str) -> AssistantPlan:
    t = user_query.strip()
    if any(w in t for w in ("记一下", "记住", "记录", "保存", "帮我记", "帮我存", "存一下")):
        return AssistantPlan(action="save", reason="根据保存类关键词判断为记录任务。", note_title=t[:18], note_content=t)
    if t.endswith(("？", "?")) or any(w in t for w in ("找", "查", "哪里", "在哪", "搜索")):
        return AssistantPlan(action="search", reason="根据疑问/检索关键词判断为查找任务。")
    return AssistantPlan(action="save", reason="默认按记录处理。", note_title=t[:18], note_content=t)


def call_planning_model(user_query: str) -> AssistantPlan:
    """用智谱API做轻量意图判断。"""
    load_env_file()

    prompt = (
        "你是本地智能语音记忆助手。请根据用户输入判断是查找还是记录，输出严格 JSON，不要输出其他内容。\n"
        "格式如下：\n"
        "{\n"
        '  \"action\": \"search|save\",\n'
        '  \"reason\": \"一句中文说明\",\n'
        '  \"note_title\": \"当 action=save 时生成一个简短标题\",\n'
        '  \"note_content\": \"当 action=save 时提炼需要记住的内容\"\n'
        "}\n"
        f"用户输入：{user_query}"
    )

    messages = [
        {"role": "system", "content": "你是一个严格输出 JSON 的中文助手。"},
        {"role": "user", "content": prompt},
    ]

    try:
        data = call_chat(messages, temperature=0.1, max_tokens=300, timeout=30)
        content = extract_text(data["choices"][0]["message"]["content"])
        parsed = parse_json_block(content)
        action = str(parsed.get("action", "")).strip()
        if action not in {"search", "save"}:
            return _fallback_plan(user_query)
        return AssistantPlan(
            action=action,
            reason=str(parsed.get("reason", "已生成计划。")).strip(),
            note_title=str(parsed.get("note_title", "")).strip(),
            note_content=str(parsed.get("note_content", "")).strip(),
        )
    except Exception as e:
        logger.warning("意图模型调用失败，使用回退: %s", e)
        return _fallback_plan(user_query)
