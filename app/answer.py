from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib import request

from .intent import extract_text, load_env_file

DEFAULT_TEXT_MODEL = "glm-4-flash-250414"


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    confidence: float = 0.0


_DATE_RE = re.compile(r"(\d{4})\s*[年\-/.]\s*(\d{1,2})\s*[月\-/.]\s*(\d{1,2})\s*[日号]?")
_PHONE_RE = re.compile(r"(1[3-9]\d{9}|\d{3,4}-\d{7,8})")


def _local_answer(query: str, contexts: list[str]) -> AnswerResult:
    q = query.strip()
    text = "\n".join(contexts)

    if "生日" in q:
        m = _DATE_RE.search(text)
        if m:
            y, mo, d = m.groups()
            return AnswerResult(answer=f"你的生日是 {int(y)}年{int(mo)}月{int(d)}日。", confidence=0.75)

    if any(k in q for k in ("电话", "号码", "手机号", "联系方式")):
        m = _PHONE_RE.search(text)
        if m:
            return AnswerResult(answer=f"我记到的号码是 {m.group(1)}。", confidence=0.7)

    # 找不到明确答案
    return AnswerResult(answer="", confidence=0.0)


def _call_answer_model(query: str, memories: list[dict[str, Any]]) -> AnswerResult | None:
    load_env_file()
    api_key = os.getenv("ZHIPU_API_KEY", "").strip()
    if not api_key:
        return None

    base_url = "https://open.bigmodel.cn/api/paas/v4"
    model = os.getenv("LOCAL_AGENT_MODEL", "glm-4-flash-250414")

    # 给模型最少但足够的信息：标题+摘要+正文片段
    packed: list[dict[str, str]] = []
    for m in memories[:6]:
        packed.append(
            {
                "title": str(m.get("title", ""))[:80],
                "summary": str(m.get("summary", ""))[:200],
                "body": str(m.get("body_snippet", ""))[:600],
            }
        )

    prompt = (
        "你是本地智能语音记忆助手。用户会问一个问题，你需要基于提供的记忆片段，直接给出简短答案。\n"
        "要求：\n"
        "1) 只输出严格 JSON，不要输出其他内容。\n"
        "2) 如果能回答，answer 用一句中文直接回答用户问题。\n"
        "3) 如果无法从记忆片段确定答案，answer 输出空字符串。\n"
        "输出格式：\n"
        "{\n"
        '  \"answer\": \"...\",\n'
        '  \"confidence\": 0.0\n'
        "}\n"
        f"用户问题：{query}\n"
        f"记忆片段：{json.dumps(packed, ensure_ascii=False)}"
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
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?", "", content).strip()
            content = re.sub(r"```$", "", content).strip()
        parsed = json.loads(content)
        answer = str(parsed.get("answer", "")).strip()
        conf = float(parsed.get("confidence", 0.0) or 0.0)
        return AnswerResult(answer=answer, confidence=max(0.0, min(1.0, conf)))
    except Exception:
        return None


def answer(query: str, memories: list[dict[str, Any]]) -> AnswerResult:
    # 先走模型（如果有），再回退本地规则
    model_res = _call_answer_model(query, memories)
    if model_res and model_res.answer:
        return model_res

    contexts: list[str] = []
    for m in memories[:6]:
        contexts.append(str(m.get("title", "")))
        contexts.append(str(m.get("summary", "")))
        contexts.append(str(m.get("body_snippet", "")))
    local = _local_answer(query, contexts)
    return local

