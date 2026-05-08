from __future__ import annotations

import sqlite3
from pathlib import Path


def default_db_path(project_root: Path) -> Path:
    return project_root / "data" / "agent.db"


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA busy_timeout = 3000;")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS meta (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        );
        """
    )
    # 统一记忆表：不对外暴露“想法/联系人”等内部分类
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memories (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          title TEXT NOT NULL,
          summary TEXT NOT NULL DEFAULT '',
          body TEXT NOT NULL DEFAULT '',
          file_path TEXT NOT NULL DEFAULT '',
          extra_json TEXT NOT NULL DEFAULT '{}'
        );
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_file_path
        ON memories(file_path)
        WHERE file_path <> '';
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_memories_updated_at
        ON memories(updated_at DESC, id DESC);
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS usage_stats (
          entity_type TEXT NOT NULL,
          entity_id INTEGER NOT NULL,
          open_count INTEGER NOT NULL DEFAULT 0,
          last_opened_at TEXT NOT NULL DEFAULT '',
          PRIMARY KEY (entity_type, entity_id)
        );
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_usage_stats_open_count
        ON usage_stats(open_count DESC, last_opened_at DESC);
        """
    )
    # FTS：用于更“智能”的本地语义检索（关键词匹配 + 排序更稳）
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
        USING fts5(title, summary, body, content='memories', content_rowid='id');
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
          INSERT INTO memories_fts(rowid, title, summary, body) VALUES (new.id, new.title, new.summary, new.body);
        END;
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
          INSERT INTO memories_fts(memories_fts, rowid, title, summary, body) VALUES('delete', old.id, old.title, old.summary, old.body);
        END;
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
          INSERT INTO memories_fts(memories_fts, rowid, title, summary, body) VALUES('delete', old.id, old.title, old.summary, old.body);
          INSERT INTO memories_fts(rowid, title, summary, body) VALUES (new.id, new.title, new.summary, new.body);
        END;
        """
    )
    conn.commit()

