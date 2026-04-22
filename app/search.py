from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any, Literal


SortMode = Literal["recent", "relevant", "frequent"]

_STOP_WORDS = {
    "我", "我的", "你", "你的", "他", "她", "它", "我们", "你们",
    "是", "叫", "为", "有", "在", "吗", "呢", "呀",
    "什么", "怎么", "多少", "几", "哪里", "哪儿", "什么时候",
    "时候", "今天", "昨天", "明天", "一下", "帮我", "请", "麻烦",
}


def _tokenize(query: str) -> list[str]:
    q = (query or "").strip()
    if not q:
        return []
    # 先取英文/数字 token；中文用更细粒度拆分，否则容易整句变成一个 token 导致命中率极差
    tokens = re.findall(r"[A-Za-z0-9_+\-]+", q)

    # 关键触发词：直接加入（提升“生日/电话/地址”等常见记忆可检索性）
    for kw in ("生日", "电话", "号码", "地址", "微信", "账号", "密码", "身份证", "银行卡"):
        if kw in q:
            tokens.append(kw)

    # 中文片段：做 2-4 字 ngram（受控数量），用于“问句→关键词”场景
    for seg in re.findall(r"[\u4e00-\u9fff]{2,30}", q):
        seg = seg.strip()
        if not seg:
            continue
        # 先去掉高频停用词片段
        for sw in sorted(_STOP_WORDS, key=len, reverse=True):
            if sw and sw in seg:
                seg = seg.replace(sw, "")
        seg = seg.strip()
        if len(seg) < 2:
            continue
        ngrams: list[str] = []
        for n in (2, 3, 4):
            if len(seg) < n:
                continue
            for i in range(0, len(seg) - n + 1):
                ngrams.append(seg[i : i + n])
        # 去重并限制数量，避免爆炸
        seen_ng: set[str] = set()
        for t in ngrams:
            if t in _STOP_WORDS:
                continue
            if t not in seen_ng:
                tokens.append(t)
                seen_ng.add(t)
            if len(seen_ng) >= 20:
                break
    cleaned: list[str] = []
    seen: set[str] = set()
    for t in tokens:
        t = t.strip()
        if not t or t in _STOP_WORDS:
            continue
        # 中文单字通常信息量低，丢掉；保留“日/月/年”等在生日场景有用的单字
        if len(t) == 1 and t not in {"日", "月", "年", "号"}:
            continue
        if t not in seen:
            cleaned.append(t)
            seen.add(t)
    return cleaned[:12]


def _fts_query(tokens: list[str]) -> str:
    # 用 OR 提升命中率；加 * 做前缀匹配（更像“智能搜索”）
    if not tokens:
        return ""
    parts: list[str] = []
    for t in tokens:
        t = t.replace('"', "")
        # 过短词不加前缀星号，避免过度扩展
        if len(t) >= 2:
            parts.append(f'{t}*')
        else:
            parts.append(t)
    return " OR ".join(parts)


def _score(text: str, query: str) -> int:
    if not query:
        return 0
    lowered = (text or "").lower()
    q = query.lower().strip()
    score = 0
    if q and q in lowered:
        score += 12
    # 简单分词：英文/数字/中文短词
    tokens = re.findall(r"[A-Za-z0-9_+\-]+|[\u4e00-\u9fff]{1,8}", q)
    for token in tokens[:6]:
        t = token.lower()
        if t and t in lowered:
            score += 3 + len(t) + lowered.count(t)
    return score


def _path_exists(file_path: str, cache: dict[str, bool]) -> bool:
    cached = cache.get(file_path)
    if cached is not None:
        return cached
    try:
        exists = Path(file_path).exists()
    except Exception:
        exists = True
    cache[file_path] = exists
    return exists


def search(conn: sqlite3.Connection, *, query: str, sort_mode: SortMode = "relevant", limit: int = 50) -> list[dict[str, Any]]:
    q = query.strip()
    if not q:
        return []

    results: list[dict[str, Any]] = []
    file_exists_cache: dict[str, bool] = {}
    tokens = _tokenize(q)
    fts_q = _fts_query(tokens) or q

    # 优先使用 FTS5（更接近“智能检索”的感觉），找不到再回退 LIKE
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
            if fp and not _path_exists(fp, file_exists_cache):
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
    except Exception:
        pass

    if not results:
        # LIKE 回退：对每个 token 做 OR，提高“问句”场景命中率
        like_terms = tokens or [q]
        wheres: list[str] = []
        params: list[str] = []
        for t in like_terms[:6]:
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
            if fp and not _path_exists(fp, file_exists_cache):
                continue
            text = f"{row['title']} {row['summary']} {row['body']}"
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
        # join usage_stats
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
        # FTS 的 bm25 越小越相关，所以这里做一个统一：如果是 float rank 用负号反转
        def key(item: dict[str, Any]) -> tuple[float, str]:
            score = item.get("score", 0)
            if isinstance(score, float):
                return (-score, item.get("time", ""))
            return (float(score), item.get("time", ""))

        results.sort(key=key, reverse=True)

    return results[:limit]

