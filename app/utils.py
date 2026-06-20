"""
公共工具函数 - 被多个模块共享的辅助功能。
从 intent.py 和 vision.py 中提取，消除重复代码。
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

DEFAULT_ENV_FILE = ".env"
_env_loaded = False


def load_env_file(env_path: str = DEFAULT_ENV_FILE) -> None:
    """加载 .env 文件到环境变量（幂等，只加载一次）"""
    global _env_loaded
    if _env_loaded:
        return
    path = Path(env_path)
    if not path.exists():
        _env_loaded = True
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
    _env_loaded = True


def extract_text(content: Any) -> str:
    """从 LLM 返回的 content 中提取纯文本（兼容 list/dict 格式）"""
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


def parse_json_block(content: str) -> dict[str, Any]:
    """解析 LLM 返回的 JSON 块（自动去除 markdown 代码块标记）"""
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```\s*$", "", content).strip()
    return json.loads(content)
