from __future__ import annotations

import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

SortMode = Literal["recent", "relevant", "frequent"]

_STOP_WORDS = {
    "我", "我的", "你", "你的", "他", "她", "它", "我们", "你们",
    "是", "叫", "为", "有", "在", "吗", "呢", "呀",
    "什么", "怎么", "多少", "几", "哪里", "哪儿", "什么时候",
    "时候", "今天", "昨天", "明天", "一下", "帮我", "请", "麻烦",
}

_STOP_WORDS_SORTED = sorted(_STOP_WORDS, key=len, reverse=True)

_TRIGGER_KEYWORDS = frozenset(("生日", "电话", "号码", "地址", "微信", "账号", "密码", "身份证", "银行卡"))
_KEEP_SINGLE_CHARS = frozenset(("日", "月", "年", "号"))


def _tokenize(query: str) -> list[str]:
    q = (query or "").strip()
    if not q:
        return []
    tokens = re.findall(r"[A-Za-z0-9_+\-]+", q)

    for kw in _TRIGGER_KEYWORDS:
        if kw in q:
            tokens.append(kw)

    for seg in re.findall(r"[一-鿿]{2,30}", q):
        seg = seg.strip()
        if not seg:
            continue
        for sw in _STOP_WORDS_SORTED:
            if sw and sw in seg:
                seg = seg.replace(sw, "")
        seg = seg.strip()
        if len(seg) < 2:
            continue
        seen_ng: set[str] = set()
        # 短句段优先 2-gram，长句段才拆 3-gram
        ngrams = (2, 3) if len(seg) > 6 else (2,)
        for n in ngrams:
            if len(seg) < n:
                continue
            for i in range(0, len(seg) - n + 1):
                t = seg[i : i + n]
                if t not in _STOP_WORDS and t not in seen_ng:
                    tokens.append(t)
                    seen_ng.add(t)
                if len(seen_ng) >= 8:
                    break
            if len(seen_ng) >= 8:
                break
    cleaned: list[str] = []
    seen: set[str] = set()
    for t in tokens:
        t = t.strip()
        if not t or t in _STOP_WORDS:
            continue
        if len(t) == 1 and t not in _KEEP_SINGLE_CHARS:
            continue
        if t not in seen:
            cleaned.append(t)
            seen.add(t)
    return cleaned[:12]


def _fts_query(tokens: list[str]) -> str:
    if not tokens:
        return ""
    parts: list[str] = []
    for t in tokens:
        t = t.replace('"', "")
        if len(t) >= 2:
            parts.append(f'{t}*')
        else:
            parts.append(t)
    return " OR ".join(parts)


def _score(text: str, query: str) -> int:
    if not query:
        return 0
    lowered = text.lower()
    q = query.lower().strip()
    score = 0
    if q in lowered:
        score += 12
    for token in re.findall(r"[A-Za-z0-9_+\-]+|[一-鿿]{1,8}", q)[:6]:
        t = token.lower()
        if t and t in lowered:
            score += 3 + len(t) + lowered.count(t)
    return score


# Module-level file existence cache with TTL
_fexists_cache: dict[str, tuple[bool, float]] = {}
_FEXISTS_TTL = 120  # 2 minutes


def _path_exists(file_path: str) -> bool:
    """Check if a file path exists, with a module-level TTL cache."""
    if not file_path:
        return True  # empty path = no file to check, treat as valid
    now = time.time()
    cached = _fexists_cache.get(file_path)
    if cached is not None:
        exists, ts = cached
        if now - ts < _FEXISTS_TTL:
            return exists
    try:
        exists = Path(file_path).exists()
    except Exception:
        exists = False
    _fexists_cache[file_path] = (exists, now)
    # Prune stale entries periodically
    if len(_fexists_cache) > 500:
        stale = [k for k, (_, ts) in _fexists_cache.items() if now - ts > _FEXISTS_TTL * 2]
        for k in stale:
            del _fexists_cache[k]
    return exists


# 搜索结果缓存（短 TTL，避免重复查询）
_search_cache: dict[str, tuple[list, float]] = {}
_SEARCH_CACHE_TTL = 60  # 60秒


def search(conn: sqlite3.Connection, *, query: str, sort_mode: SortMode = "relevant", limit: int = 50) -> list[dict[str, Any]]:
    q = query.strip()
    if not q:
        return []

    # 检查缓存
    cache_key = f"{q}:{sort_mode}:{limit}"
    now = time.time()
    cached = _search_cache.get(cache_key)
    if cached is not None:
        results, ts = cached
        if now - ts < _SEARCH_CACHE_TTL:
            return results

    results: list[dict[str, Any]] = []
    tokens = _tokenize(q)
    fts_q = _fts_query(tokens) or q

    try:
        fts_rows = conn.execute(
            """
            SELECT m.id, m.title, m.summary, m.body, m.file_path, m.updated_at,
                   bm25(memories_fts) AS rank
            FROM memories_fts
            JOIN memories m ON m.id = memories_fts.rowid
            WHERE memories_fts MATCH ?
            ORDER BY rank
            LIMIT 200
            """,
            (fts_q,),
        ).fetchall()
        for row in fts_rows:
            fp = str(row["file_path"] or "").strip()
            if fp and not _path_exists(fp):
                continue
            results.append(
                {
                    "entity_type": "memory",
                    "entity_id": int(row["id"]),
                    "title": row["title"],
                    "time": row["updated_at"],
                    "summary": (row["summary"] or row["body"] or "")[:90],
                    "body_snippet": (row["body"] or "")[:800],
                    "file_path": fp,
                    "score": float(row["rank"]),
                }
            )
    except Exception as e:
        logger.debug("FTS 搜索失败，回退到 LIKE: %s", e)

    if not results:
        like_terms = tokens or [q]
        wheres: list[str] = []
        params: list[str] = []
        for t in like_terms[:3]:
            t = t.replace(chr(92), chr(92)*2).replace(chr(37), chr(92)+chr(37)).replace(chr(95), chr(92)+chr(95))
            wheres.append("(title LIKE ? OR summary LIKE ? OR body LIKE ?)")
            params.extend([f"%{t}%", f"%{t}%", f"%{t}%"])
        where_sql = " OR ".join(wheres) if wheres else "(title LIKE ? OR summary LIKE ? OR body LIKE ?)"
        if not params:
            params = [f"%{q}%", f"%{q}%", f"%{q}%"]

        for row in conn.execute(
            f"SELECT id, title, summary, body, file_path, updated_at FROM memories WHERE {where_sql} LIMIT 200",
            tuple(params),
        ).fetchall():
            fp = str(row["file_path"] or "").strip()
            if fp and not _path_exists(fp):
                continue
            body_preview = (row['body'] or '')[:500]
            text = f"{row['title']} {row['summary']} {body_preview}"
            results.append(
                {
                    "entity_type": "memory",
                    "entity_id": int(row["id"]),
                    "title": row["title"],
                    "time": row["updated_at"],
                    "summary": (row["summary"] or row["body"] or "")[:90],
                    "body_snippet": (row["body"] or "")[:800],
                    "file_path": fp,
                    "score": _score(text, q),
                }
            )

    if sort_mode == "recent":
        results.sort(key=lambda x: (x.get("time", ""), x.get("entity_id", 0)), reverse=True)
    elif sort_mode == "frequent":
        freq_map: dict[tuple[str, int], int] = {}
        for row in conn.execute("SELECT entity_type, entity_id, open_count FROM usage_stats").fetchall():
            freq_map[(row["entity_type"], int(row["entity_id"]))] = int(row["open_count"])
        results.sort(
            key=lambda x: (freq_map.get((x["entity_type"], x["entity_id"]), 0), x.get("time", "")),
            reverse=True,
        )
        for item in results:
            item["open_count"] = freq_map.get((item["entity_type"], item["entity_id"]), 0)
    else:
        for item in results:
            score = item.get("score", 0)
            if isinstance(score, float) and score < 0:
                item["_nscore"] = -score
            else:
                item["_nscore"] = float(score)
        results.sort(key=lambda x: (x.get("_nscore", 0), x.get("time", "")), reverse=True)

    final = results[:limit]
    # 存入缓存
    _search_cache[cache_key] = (final, time.time())
    # 定期清理过期缓存
    if len(_search_cache) > 200:
        stale_keys = [k for k, (_, ts) in _search_cache.items() if time.time() - ts > _SEARCH_CACHE_TTL * 2]
        for k in stale_keys:
            del _search_cache[k]
    return final
