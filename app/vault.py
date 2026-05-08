from __future__ import annotations

import hashlib
import shutil
from datetime import datetime
from pathlib import Path


def ensure_vault_root(vault_root: Path) -> None:
    vault_root.mkdir(parents=True, exist_ok=True)
    (vault_root / "attachments").mkdir(parents=True, exist_ok=True)
    (vault_root / "texts").mkdir(parents=True, exist_ok=True)


def _now_folder() -> str:
    return datetime.now().strftime("%Y/%m")


def _hash_path(path: Path) -> str:
    try:
        stat = path.stat()
        payload = f"{path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}".encode("utf-8", errors="ignore")
    except Exception:
        payload = str(path).encode("utf-8", errors="ignore")
    return hashlib.sha1(payload).hexdigest()[:16]


def import_attachment(src: Path, vault_root: Path) -> Path:
    """
    把外部文件复制进记忆库，返回库内路径。
    - 不修改源文件
    - 避免重名：文件名后追加短 hash
    """
    ensure_vault_root(vault_root)
    try:
        src_resolved = src.resolve()
    except OSError:
        src_resolved = src.absolute()
    stamp_folder = _now_folder()
    dest_dir = vault_root / "attachments" / stamp_folder
    dest_dir.mkdir(parents=True, exist_ok=True)

    suffix = src_resolved.suffix
    stem = src_resolved.stem
    short = _hash_path(src_resolved)
    dest = dest_dir / f"{stem}_{short}{suffix}"

    if dest.exists():
        return dest
    try:
        shutil.copy2(str(src_resolved), str(dest))
    except Exception as e:
        raise OSError(f"复制文件失败: {src_resolved} -> {dest}: {e}") from e
    return dest


# 文件名中不允许的字符（OS 保留）
_INVALID_NAME_CHARS = set('/\\:*?"<>|')


def _sanitize_title(title: str) -> str:
    """清理标题中的非法文件名字符，保留中文标点"""
    cleaned = "".join(ch for ch in title if ch not in _INVALID_NAME_CHARS)
    return cleaned.strip()[:30] or "记忆"


def write_text_memory(text: str, vault_root: Path, *, title: str = "记忆") -> Path:
    """
    把文本写成 Markdown 文件保存到记忆库，返回文件路径。
    """
    ensure_vault_root(vault_root)
    stamp_folder = _now_folder()
    dest_dir = vault_root / "texts" / stamp_folder
    dest_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_title = _sanitize_title(title)
    dest = dest_dir / f"{ts}_{safe_title}.md"

    body = f"# {title}\n\n{text.strip()}\n"
    try:
        dest.write_text(body, encoding="utf-8")
    except Exception as e:
        raise OSError(f"写入文本记忆失败: {dest}: {e}") from e
    return dest

