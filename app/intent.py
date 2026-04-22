from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import request


# 智谱API配置
DEFAULT_TEXT_MODEL = "glm-4-flash-250414"
DEFAULT_ENV_FILE = ".env"


def load_env_file(env_path: str = DEFAULT_ENV_FILE) -> None:
    path = Path(env_path)
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and "text" in item:
                    texts.append(str(item["text"]))
                elif "content" in item:
                    texts.append(str(item["content"]))
        return "\n".join(part for part in texts if part)
    return str(content)


@dataclass
class AssistantPlan:
    action: str  # "search" | "save"
    reason: str
    note_title: str = ""
    note_content: str = ""


def _parse_json_block(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?", "", content).strip()
        content = re.sub(r"```$", "", content).strip()
    return json.loads(content)


def _fallback_plan(user_query: str) -> AssistantPlan:
    t = user_query.strip()
    if any(w in t for w in ("记一下", "记住", "记录", "保存", "帮我记", "帮我存", "存一下")):
        return AssistantPlan(action="save", reason="根据保存类关键词判断为记录任务。", note_title=t[:18], note_content=t)
    if t.endswith(("？", "?")) or any(w in t for w in ("找", "查", "哪里", "在哪", "搜索")):
        return AssistantPlan(action="search", reason="根据疑问/检索关键词判断为查找任务。")
    # 默认更像"记忆助手"：不明确就先记
    return AssistantPlan(action="save", reason="默认按记录处理。", note_title=t[:18], note_content=t)


def call_planning_model(user_query: str) -> AssistantPlan:
    """
    用智谱API做轻量"意图判断"。
    """
    load_env_file()
    api_key = os.getenv("ZHIPU_API_KEY", "").strip()
    if not api_key:
        return _fallback_plan(user_query)

    base_url = "https://open.bigmodel.cn/api/paas/v4"
    model = os.getenv("LOCAL_AGENT_MODEL", "glm-4-flash-250414")
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

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是一个严格输出 JSON 的中文助手。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 300,
    }

    req = request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
        content = extract_text(data["choices"][0]["message"]["content"])
        parsed = _parse_json_block(content)
        action = str(parsed.get("action", "search")).strip()
        if action not in {"search", "save"}:
            action = "search"
        return AssistantPlan(
            action=action,
            reason=str(parsed.get("reason", "已生成计划。")).strip(),
            note_title=str(parsed.get("note_title", "")).strip(),
            note_content=str(parsed.get("note_content", "")).strip(),
        )
    except Exception:
        return _fallback_plan(user_query)

