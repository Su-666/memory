from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from .utils import extract_text, load_env_file, parse_json_block
from .zhipu_client import call_chat

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    confidence: float = 0.0


_DATE_RE = re.compile(r"(\d{4})\s*[年\-/.]\s*(\d{1,2})\s*[月\-/.]\s*(\d{1,2})\s*[日号]?")
_PHONE_RE = re.compile(r"(1[3-9]\d{9}|\d{3,4}-\d{7,8})")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_ID_CARD_RE = re.compile(r"(\d{17}[\dXx])")
_BANK_CARD_RE = re.compile(r"(\d{16,19})")

# 通用 key-value 模式：XX是/为/：/: YY
_KV_RE = re.compile(r"(密码|地址|邮箱|手机号|电话|生日|身份证|银行卡|卡号|账号|车牌|QQ|微信|WiFi)[^\n]{0,5}?(?:是|为|：|:)\s*[^\n]+")


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

    if any(k in q for k in ("邮箱", "邮件", "email")):
        m = _EMAIL_RE.search(text)
        if m:
            return AnswerResult(answer=f"你的邮箱是 {m.group()}。", confidence=0.7)

    if any(k in q for k in ("身份证", "证件号")):
        m = _ID_CARD_RE.search(text)
        if m:
            return AnswerResult(answer=f"你的身份证号是 {m.group(1)}。", confidence=0.7)

    if any(k in q for k in ("银行卡", "卡号")):
        m = _BANK_CARD_RE.search(text)
        if m:
            return AnswerResult(answer=f"你的银行卡号是 {m.group(1)}。", confidence=0.65)

    # 通用 key-value 匹配（密码、地址、账号等）
    if any(k in q for k in ("密码", "地址", "账号", "车牌", "QQ", "微信", "WiFi", "wifi")):
        m = _KV_RE.search(text)
        if m:
            # 提取"是"后面的内容
            line = m.group().strip()
            for sep in ("是", "为", "：", ":"):
                if sep in line:
                    value = line.split(sep, 1)[1].strip()
                    if value:
                        return AnswerResult(answer=f"记到的是：{value}", confidence=0.6)
                    break

    return AnswerResult(answer="", confidence=0.0)


def _call_answer_model(query: str, memories: list[dict[str, Any]]) -> AnswerResult | None:
    load_env_file()

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

    messages = [
        {"role": "system", "content": "你是一个严格输出 JSON 的中文助手。"},
        {"role": "user", "content": prompt},
    ]

    try:
        data = call_chat(messages, temperature=0.1, max_tokens=200, timeout=10, retries=1)
        content = extract_text(data["choices"][0]["message"]["content"])
        parsed = parse_json_block(content)
        answer = str(parsed.get("answer", "")).strip()
        conf = float(parsed.get("confidence") or 0.0)
        return AnswerResult(answer=answer, confidence=max(0.0, min(1.0, conf)))
    except Exception as e:
        logger.warning("回答模型调用失败: %s", e)
        return None


def answer(query: str, memories: list[dict[str, Any]]) -> AnswerResult:
    # 优先尝试本地规则（零延迟）
    contexts: list[str] = []
    for m in memories[:6]:
        contexts.append(str(m.get("title", "")))
        contexts.append(str(m.get("summary", "")))
        contexts.append(str(m.get("body_snippet", "")))
    local_res = _local_answer(query, contexts)
    if local_res.answer:
        return local_res

    # 本地无法回答时才调用 LLM
    model_res = _call_answer_model(query, memories)
    if model_res and model_res.answer:
        return model_res

    return AnswerResult(answer="", confidence=0.0)
