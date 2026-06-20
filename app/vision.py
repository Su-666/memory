from __future__ import annotations

import base64
import logging
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path

from .utils import extract_text, load_env_file, parse_json_block
from .zhipu_client import call_chat

logger = logging.getLogger(__name__)

DEFAULT_VISION_MODEL = "glm-4v-flash"

# 图片大小限制（10MB），防止 OOM
_MAX_IMAGE_BYTES = 10 * 1024 * 1024


@dataclass(frozen=True)
class ImageUnderstanding:
    caption: str
    tags: list[str]
    text_in_image: str


def _image_to_data_url(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(image_path.name)
    mime_type = mime_type or "application/octet-stream"
    data = image_path.read_bytes()
    if len(data) > _MAX_IMAGE_BYTES:
        raise ValueError(f"图片文件过大（{len(data) // 1024 // 1024}MB），限制 {_MAX_IMAGE_BYTES // 1024 // 1024}MB")
    encoded = base64.b64encode(data).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def understand_image(image_path: str) -> ImageUnderstanding:
    """使用智谱API的视觉模型理解图片。"""
    load_env_file()

    model = os.getenv("LOCAL_AGENT_VISION_MODEL", DEFAULT_VISION_MODEL)
    try:
        retries = int(os.getenv("LOCAL_AGENT_VISION_RETRIES", "3") or "3")
    except ValueError:
        retries = 3

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

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]

    try:
        data = call_chat(messages, model=model, temperature=0.1, max_tokens=600, timeout=90, retries=retries)
    except Exception as exc:
        raise RuntimeError(
            "图片解析失败（网络/SSL 连接被中断）。\n"
            "建议排查：\n"
            "1) 网络是否可访问智谱API（必要时使用代理/更换网络）\n"
            "2) 检查 API Key 是否正确\n"
            "3) 重试几次（已自动重试）"
        ) from exc

    content = extract_text(data["choices"][0]["message"]["content"])
    parsed = parse_json_block(content)

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
