from __future__ import annotations

import re
from dataclasses import dataclass


PHONE_RE = re.compile(r"(1[3-9]\d{9}|\d{3,4}-\d{7,8}|\d{7,12})")


@dataclass(frozen=True)
class ExtractedPerson:
    name: str
    phones: list[str]
    notes: str = ""


def extract_person(text: str) -> ExtractedPerson | None:
    cleaned = text.strip()
    if not cleaned:
        return None

    phones = PHONE_RE.findall(cleaned)
    phones = [p.strip() for p in phones if p.strip()]
    phones = list(dict.fromkeys(phones))

    # 简单姓名抽取：去掉指令与电话号码后的剩余词
    name_candidate = cleaned
    for prefix in ("/联系人", "联系人", "电话", "号码", "保存", "记录", "帮我", "请"):
        name_candidate = name_candidate.replace(prefix, " ")
    for p in phones:
        name_candidate = name_candidate.replace(p, " ")
    name_candidate = re.sub(r"\s+", " ", name_candidate).strip(" ：:，,。;；")

    # 取第一个中文短词作为姓名（可在结构化页再确认）
    name_match = re.search(r"[\u4e00-\u9fff]{2,6}", name_candidate)
    name = name_match.group(0) if name_match else (name_candidate.split(" ")[0] if name_candidate else "")

    if not name and not phones:
        return None
    return ExtractedPerson(name=name or "未命名联系人", phones=phones)

