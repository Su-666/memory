from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from .vault import ensure_vault_root, import_attachment, write_text_memory


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def insert_memory(
    conn: sqlite3.Connection,
    *,
    title: str,
    summary: str = "",
    body: str = "",
    file_path: str = "",
    extra: dict[str, Any] | None = None,
    created_at: str | None = None,
    updated_at: str | None = None,
    commit: bool = True,
) -> int:
    now = _now()
    created_at = created_at or now
    updated_at = updated_at or now
    cur = conn.execute(
        "INSERT INTO memories(created_at, updated_at, title, summary, body, file_path, extra_json) VALUES(?,?,?,?,?,?,?)",
        (
            created_at,
            updated_at,
            title.strip() or "记忆",
            summary.strip(),
            body.strip(),
            file_path,
            _json_dumps(extra or {}),
        ),
    )
    if commit:
        conn.commit()
    return int(cur.lastrowid)


def get_memory(conn: sqlite3.Connection, memory_id: int) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT id, created_at, updated_at, title, summary, body, file_path, extra_json FROM memories WHERE id = ?",
        (int(memory_id),),
    ).fetchone()
    if not row:
        return None
    return {
        "id": int(row["id"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "title": row["title"],
        "summary": row["summary"],
        "body": row["body"],
        "file_path": row["file_path"],
        "extra": json.loads(row["extra_json"] or "{}"),
    }


def delete_memory(conn: sqlite3.Connection, memory_id: int, *, commit: bool = True) -> None:
    conn.execute("DELETE FROM memories WHERE id = ?", (int(memory_id),))
    if commit:
        conn.commit()


def update_memory(
    conn: sqlite3.Connection,
    *,
    memory_id: int,
    title: str | None = None,
    summary: str | None = None,
    body: str | None = None,
    extra_patch: dict[str, Any] | None = None,
    commit: bool = True,
) -> None:
    row = conn.execute(
        "SELECT title, summary, body, extra_json FROM memories WHERE id = ?",
        (int(memory_id),),
    ).fetchone()
    if not row:
        return
    new_title = title if title is not None else row["title"]
    new_summary = summary if summary is not None else row["summary"]
    new_body = body if body is not None else row["body"]
    old_extra = json.loads(row["extra_json"] or "{}")
    if extra_patch:
        old_extra.update(extra_patch)
    conn.execute(
        "UPDATE memories SET title = ?, summary = ?, body = ?, extra_json = ?, updated_at = ? WHERE id = ?",
        (new_title, new_summary, new_body, _json_dumps(old_extra), _now(), int(memory_id)),
    )
    if commit:
        conn.commit()


def remember_text_smart(conn: sqlite3.Connection, *, text: str, vault_root: Path, title: str, summary: str) -> int:
    ensure_vault_root(vault_root)
    file_path = str(write_text_memory(text, vault_root, title=title))
    return insert_memory(conn, title=title, summary=summary, body=text, file_path=file_path, extra={"source": "text"})


def remember_attachments(conn: sqlite3.Connection, *, paths: list[str], caption: str, vault_root: Path) -> list[int]:
    ensure_vault_root(vault_root)
    memory_ids: list[int] = []
    dirty = False
    normalized_caption = caption.strip()

    for raw_path in paths:
        src = Path(raw_path)
        if not src.exists():
            continue
        dest = import_attachment(src, vault_root)
        suffix = src.suffix.lower()
        media_type = "file"
        if suffix in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff"}:
            media_type = "image"
        elif suffix in {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".m4v", ".flv", ".webm", ".mpeg", ".mpg"}:
            media_type = "video"
        elif suffix in {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}:
            media_type = "audio"

        memory_ids.append(
            insert_memory(
                conn,
                title=src.name,
                summary=normalized_caption or f"已保存附件：{src.name}",
                body=normalized_caption,
                file_path=str(dest),
                extra={"source": "drop", "original_path": str(src), "media_type": media_type},
                commit=False,
            )
        )
        dirty = True

    if dirty:
        conn.commit()
    return memory_ids


def _load_existing_file_paths(conn: sqlite3.Connection) -> set[str]:
    return {
        str(row["file_path"] or "").strip()
        for row in conn.execute("SELECT file_path FROM memories WHERE file_path <> ''").fetchall()
        if str(row["file_path"] or "").strip()
    }


def bootstrap_import_vault(conn: sqlite3.Connection, vault_root: Path) -> int:
    ensure_vault_root(vault_root)
    imported = 0
    existing_paths = _load_existing_file_paths(conn)

    texts_root = vault_root / "texts"
    for path in texts_root.rglob("*.md"):
        fp = str(path.resolve())
        if fp in existing_paths:
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        lines = [line.strip() for line in content.splitlines()]
        title = path.stem
        for line in lines[:5]:
            if line.startswith("#"):
                title = line.lstrip("#").strip() or title
                break

        summary = ""
        for line in lines:
            if line and not line.startswith("#"):
                summary = line[:120]
                break

        timestamp: str | None = None
        try:
            timestamp = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass

        try:
            insert_memory(
                conn,
                title=title,
                summary=summary,
                body=content.strip()[:5000],
                file_path=fp,
                extra={"source": "vault_texts"},
                created_at=timestamp,
                updated_at=timestamp,
                commit=False,
            )
            existing_paths.add(fp)
            imported += 1
        except Exception:
            continue

    attach_root = vault_root / "attachments"
    for path in attach_root.rglob("*"):
        if not path.is_file():
            continue
        fp = str(path.resolve())
        if fp in existing_paths:
            continue

        timestamp: str | None = None
        try:
            timestamp = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass

        try:
            insert_memory(
                conn,
                title=path.name,
                summary=f"附件：{path.name}",
                body="",
                file_path=fp,
                extra={"source": "vault_attachments"},
                created_at=timestamp,
                updated_at=timestamp,
                commit=False,
            )
            existing_paths.add(fp)
            imported += 1
        except Exception:
            continue

    if imported:
        conn.commit()
    return imported


def prune_missing_file_memories(conn: sqlite3.Connection) -> int:
    rows = conn.execute("SELECT id, file_path FROM memories WHERE file_path <> ''").fetchall()
    ids_to_remove: list[tuple[int]] = []

    for row in rows:
        file_path = str(row["file_path"] or "").strip()
        if not file_path:
            continue
        try:
            if not Path(file_path).exists():
                ids_to_remove.append((int(row["id"]),))
        except Exception:
            continue

    if ids_to_remove:
        conn.executemany("DELETE FROM memories WHERE id = ?", ids_to_remove)
        conn.commit()
    return len(ids_to_remove)


def bump_open_count(conn: sqlite3.Connection, *, entity_type: str, entity_id: int) -> None:
    now = _now()
    conn.execute(
        """
        INSERT INTO usage_stats(entity_type, entity_id, open_count, last_opened_at)
        VALUES(?, ?, 1, ?)
        ON CONFLICT(entity_type, entity_id) DO UPDATE SET
          open_count = usage_stats.open_count + 1,
          last_opened_at = excluded.last_opened_at
        """,
        (entity_type, entity_id, now),
    )
    conn.commit()


def list_recent(conn: sqlite3.Connection, *, limit: int = 30) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for row in conn.execute(
        """
        SELECT id, title, summary, body, file_path, updated_at
        FROM memories
        ORDER BY updated_at DESC, id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall():
        file_path = str(row["file_path"] or "").strip()
        tag = "文本"
        if file_path:
            tag = "附件" if Path(file_path).suffix else "文件"
        items.append(
            {
                "entity_type": "memory",
                "entity_id": int(row["id"]),
                "title": row["title"],
                "time": row["updated_at"],
                "tag": tag,
                "summary": ((row["summary"] or row["body"] or "")[:60]).strip(),
                "file_path": file_path,
            }
        )
    return items
