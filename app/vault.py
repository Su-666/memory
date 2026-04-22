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
    # 用路径 + mtime + size 生成短 hash，避免重复文件名冲突
    try:
        stat = path.stat()
        payload = f"{path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}".encode("utf-8", errors="ignore")
    except Exception:
        payload = str(path).encode("utf-8", errors="ignore")
    return hashlib.sha1(payload).hexdigest()[:10]


def import_attachment(src: Path, vault_root: Path) -> Path:
    """
    把外部文件复制进记忆库，返回库内路径。
    - 不修改源文件
    - 避免重名：文件名后追加短 hash
    """
    ensure_vault_root(vault_root)
    src = src.resolve()
    stamp_folder = _now_folder()
    dest_dir = vault_root / "attachments" / stamp_folder
    dest_dir.mkdir(parents=True, exist_ok=True)

    suffix = src.suffix
    stem = src.stem
    short = _hash_path(src)
    dest = dest_dir / f"{stem}_{short}{suffix}"

    if dest.exists():
        return dest
    shutil.copy2(str(src), str(dest))
    return dest


def write_text_memory(text: str, vault_root: Path, *, title: str = "记忆") -> Path:
    """
    把文本写成 Markdown 文件保存到记忆库，返回文件路径。
    """
    ensure_vault_root(vault_root)
    stamp_folder = _now_folder()
    dest_dir = vault_root / "texts" / stamp_folder
    dest_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_title = "".join(ch for ch in title if ch.isalnum() or ch in ("_", "-", " ", "·", "（", "）")).strip() or "记忆"
    safe_title = safe_title[:30].strip()
    dest = dest_dir / f"{ts}_{safe_title}.md"

    body = f"# {title}\n\n{text.strip()}\n"
    dest.write_text(body, encoding="utf-8")
    return dest

