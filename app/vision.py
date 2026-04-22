from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import request

from .intent import extract_text, load_env_file


DEFAULT_VISION_MODEL = "glm-4v-flash"


@dataclass(frozen=True)
class ImageUnderstanding:
    caption: str
    tags: list[str]
    text_in_image: str


def _image_to_data_url(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(image_path.name)
    mime_type = mime_type or "application/octet-stream"
    data = image_path.read_bytes()
    encoded = base64.b64encode(data).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _parse_json_block(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?", "", content).strip()
        content = re.sub(r"```$", "", content).strip()
    return json.loads(content)


def understand_image(image_path: str) -> ImageUnderstanding:
    """
    使用智谱API的视觉模型理解图片。
    输出：caption/tags/text_in_image，用于本地检索索引。
    """
    load_env_file()
    api_key = os.getenv("ZHIPU_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("未配置 ZHIPU_API_KEY，无法进行图片理解。")

    base_url = "https://open.bigmodel.cn/api/paas/v4"
    model = os.getenv("LOCAL_AGENT_VISION_MODEL", "glm-4v-flash")
    retries = int(os.getenv("LOCAL_AGENT_VISION_RETRIES", "3") or "3")

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"图片不存在：{path}")

    data_url = _image_to_data_url(path)
    prompt = (
        "你是本地记忆助手的图片理解模块。请根据图片内容输出严格 JSON，不要输出其他内容。\n"
        "格式：\n"
        "{\n"
        '  \"caption\": \"一句中文描述，尽量具体（人物/物体/场景/动作/关键信息）\",\n'
        '  \"tags\": [\"2-8个中文短标签\"],\n'
        '  \"text_in_image\": \"如果图片中有可读文字，尽量完整提取；没有则为空字符串\"\n'
        "}\n"
    )

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "temperature": 0.1,
        "max_tokens": 600,
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

    last_exc: Exception | None = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            with request.urlopen(req, timeout=90) as response:
                data = json.loads(response.read().decode("utf-8"))
            break
        except Exception as exc:
            last_exc = exc
            # 退避重试（1.0s, 2.0s, 4.0s）
            if attempt < retries:
                time.sleep(2 ** (attempt - 1))
                continue
            raise RuntimeError(
                "图片解析失败（网络/SSL 连接被中断）。\n"
                "建议排查：\n"
                "1) 网络是否可访问智谱API（必要时使用代理/更换网络）\n"
                "2) 检查 API Key 是否正确\n"
                "3) 重试几次（已自动重试）\n"
                f"原始错误：{exc}"
            ) from exc

    content = extract_text(data["choices"][0]["message"]["content"])
    parsed = _parse_json_block(content)

    caption = str(parsed.get("caption", "")).strip()
    tags_raw = parsed.get("tags", [])
    tags: list[str] = []
    if isinstance(tags_raw, list):
        tags = [str(t).strip() for t in tags_raw if str(t).strip()]
    text_in_image = str(parsed.get("text_in_image", "")).strip()

    return ImageUnderstanding(
        caption=caption,
        tags=tags[:12],
        text_in_image=text_in_image,
    )





