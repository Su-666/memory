"""
统一意图判断、保存确认、搜索查询提取、记忆元数据构建。
桌面端和 Web 端共用此模块，避免逻辑重复和不一致。
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from . import repo as app_repo
from . import search as app_search
from .answer import answer as app_answer
from .intent import call_planning_model


# ============ 意图判断 ============

def determine_intent(text: str) -> str:
    """判断用户意图：save / search / chat
    检测顺序：明确命令 -> 问候/问句 -> 关键词匹配 -> 默认 chat
    """
    stripped = text.strip()
    if not stripped:
        return "chat"

    # 明确命令
    if stripped.startswith(("/聊", "/chat")):
        return "chat"
    if stripped.startswith(("/搜", "/search")):
        return "search"

    # ===== 保存意图（优先于问候/问句检测） =====
    save_keywords = (
        "帮我记", "记一下", "记住", "记录一下", "保存一下",
        "帮我存", "存一下", "提醒我", "别忘了", "备忘",
        "把这个记下来", "收藏", "记下来", "要记住", "写下来",
    )
    if any(k in stripped for k in save_keywords):
        return "save"

    # ===== 搜索意图 =====
    search_keywords = (
        "帮我找", "帮我查", "查一下", "搜一下", "找一下",
        "查找", "找到", "搜索", "记得", "我记过",
        "之前记", "以前记", "我存过", "我的记忆", "记忆库",
        "保存过", "记录过", "查询", "检索",
    )
    if any(k in stripped for k in search_keywords):
        return "search"

    memory_question_keywords = (
        "我记了什么", "记得什么", "存了什么", "保存了什么",
        "记忆里有", "记忆库有", "密码", "号码", "电话", "生日",
        "之前记的", "以前记的", "我记录过",
    )
    if any(k in stripped for k in memory_question_keywords):
        return "search"

    # 问候/闲聊（短文本 + 问候词，且不含保存/搜索关键词）
    greeting_keywords = (
        "你好", "嗨", "hi", "hello", "在吗", "在不在", "早上好",
        "晚上好", "中午好", "晚安", "再见", "拜拜", "感谢",
        "谢谢", "辛苦了", "加油", "嘿", "哈喽", "干嘛", "干啥",
    )
    if len(stripped) <= 10 and any(k in stripped.lower() for k in greeting_keywords):
        return "chat"

    # 问句 -> chat（让大模型联网搜索）
    question_marks = ("？", "?", "嘛", "吗", "呀", "啊")
    if any(stripped.endswith(m) for m in question_marks):
        return "chat"

    return "chat"


# ============ 保存确认 ============

# 通用的确认/取消词集合
CONFIRM_WORDS = frozenset({
    "记住", "保存", "要", "好", "好的", "是", "嗯", "ok", "对", "没错", "行", "可以",
    "好呀", "好嘞", "好吧", "行吧", "嗯嗯", "好哒", "确定", "当然", "要的",
})
CANCEL_WORDS = frozenset({
    "不用", "不", "不要", "取消", "算了", "算了算了", "别", "不用了", "不记了",
    "不必", "不需要", "不保存", "不想要",
})


def check_save_pending(text: str) -> tuple[bool, str | None]:
    """检查用户是否只表达了保存意图但没有具体内容。
    返回 (need_wait, pending_text)
    """
    text_stripped = text.strip()

    # 有"是"且后面有实质内容 -> 不需要等待
    if "是" in text_stripped:
        parts = text_stripped.split("是", 1)
        after_is = parts[1].strip() if len(parts) == 2 else ""
        if after_is and (len(after_is) > 2 or any(c.isdigit() for c in after_is)):
            return (False, None)

    # 有数字 -> 有具体内容
    if any(c.isdigit() for c in text_stripped):
        return (False, None)

    # 纯意图短语（后面没有内容或只有占位词）
    save_verbs = [
        "帮我记住", "帮我记", "帮我存", "帮我保存",
        "记一下", "存一下", "记录一下", "保存一下",
        "记下来", "存下来", "写下来", "写一下",
        "备忘", "别忘了", "提醒我",
    ]
    for verb in save_verbs:
        if text_stripped.startswith(verb):
            after = text_stripped[len(verb):].strip()
            if not after:
                return (True, text_stripped)
            placeholder_keywords = [
                "我的手机号", "我的电话", "我的地址", "我的生日",
                "我的邮箱", "我的卡号", "我的账号", "我的密码",
                "手机号", "电话", "地址", "生日", "邮箱", "卡号", "账号", "密码",
            ]
            after_clean = after.replace("我的", "").strip()
            if any(after_clean == kw or after == kw for kw in placeholder_keywords):
                return (True, text_stripped)
            if len(after) <= 3:
                return (True, text_stripped)
            return (False, None)

    # 短文本纯保存词
    if len(text_stripped) <= 6:
        short_intent = ["记住", "记下", "保存", "存下", "备忘"]
        if any(w == text_stripped for w in short_intent):
            return (True, text_stripped)

    return (False, None)


# ============ 搜索查询提取 ============

def extract_search_query(text: str) -> str:
    """从用户输入中提取搜索关键词"""
    query = text.strip()

    if query.startswith("/搜"):
        query = query[2:].strip(" ：,，,。")
    elif query.startswith("/search"):
        query = query[len("/search"):].strip()

    remove_prefixes = [
        "帮我找一下", "帮我查一下", "帮我找", "帮我查",
        "查一下", "搜一下", "找一下", "查找", "搜索",
        "我记过的", "我保存的", "我的",
    ]
    for prefix in remove_prefixes:
        if query.startswith(prefix):
            query = query[len(prefix):].strip(" ：,，,。")

    remove_suffixes = ["是什么", "是多少", "在哪里", "在哪", "吗", "呢", "吧"]
    for suffix in remove_suffixes:
        if query.endswith(suffix):
            query = query[:-len(suffix)].strip(" ：,，,。")

    return query or text.strip()


# ============ 记忆元数据构建 ============

def build_memory_metadata(text: str) -> tuple[str, str]:
    """从用户输入中提取标题和摘要（会调用 LLM）"""
    title = ""
    summary = ""
    try:
        plan = call_planning_model(text)
        title = (getattr(plan, "note_title", "") or "").strip()
        summary = (getattr(plan, "note_content", "") or "").strip()
    except Exception:
        pass

    if not title:
        if "我的" in text and ("是" in text or ":" in text or "：" in text):
            parts = text.replace("：", ":").split(":")
            if len(parts) >= 2:
                title = parts[0].strip()[:20]
            else:
                parts = text.split("是")
                if len(parts) >= 2:
                    title = ("我的" + parts[1].strip())[:20] if "我的" in parts[0] else parts[0].strip()[:20]

        if not title:
            keywords = ["密码", "手机号", "电话", "地址", "生日", "邮箱", "卡号", "账号", "身份证", "车牌"]
            for kw in keywords:
                if kw in text:
                    title = f"我的{kw}"
                    break

    fallback_title = title or ((text[:18] + "…") if len(text) > 18 else (text or "记忆"))
    fallback_summary = summary or ((text[:80] + "…") if len(text) > 80 else (text or "记忆"))
    return (fallback_title, fallback_summary)


# ============ 搜索记忆 ============

def search_memory(conn, user_text: str, limit: int = 5) -> list:
    """统一搜索记忆，智能扩展搜索词"""
    query = extract_search_query(user_text)

    search_queries = [query]
    important_keywords = ["手机号", "电话", "密码", "地址", "生日", "邮箱", "卡号", "账号"]
    for kw in important_keywords:
        if kw in query and kw not in search_queries:
            search_queries.append(kw)

    results: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for sq in search_queries:
        try:
            items = app_search.search(conn, query=sq, sort_mode="relevant", limit=20)
            for item in items:
                item_id = item.get("id") or item.get("entity_id")
                if item_id and item_id not in seen_ids:
                    seen_ids.add(item_id)
                    results.append(item)
        except Exception:
            continue

    return results[:limit]


# ============ 构建记忆上下文 ============

def build_memory_context(results: list, max_items: int = 3) -> str:
    """把搜索结果构建成 LLM 可读的记忆提示"""
    if not results:
        return ""

    lines = []
    for r in results[:max_items]:
        title = r.get("title", "")
        summary = r.get("summary", "")
        body = r.get("body", "")
        time_str = r.get("time", "")

        content = summary or body or ""
        if content and len(content) > 120:
            content = content[:120] + "..."

        parts = []
        if title:
            parts.append(title)
        if content and content != title:
            parts.append(content)

        if parts:
            line = "• " + "：".join(parts[:2])
            if time_str:
                line += f" （{time_str}）"
            lines.append(line)

    return "\n".join(lines) if lines else ""


# ============ 处理意图（Web/桌面共用核心逻辑） ============

# 通用的随机回复
_SAVE_REPLIES = [
    "记好啦~ 以后忘了随时问我呀",
    "收到~ 帮你记下了",
    "好嘞~ 记住了",
    "记好啦，放心~",
    "搞定~ 记好了",
]
_SAVE_PENDING_REPLIES = [
    "好呀，具体是什么内容呢？说给我听听～",
    "行，说具体内容吧~",
    "好嘞，告诉我具体内容~",
    "嗯嗯，说具体内容给我吧~",
]
_SEARCH_NOT_FOUND_REPLIES = [
    "没找到呢～换个说法试试？",
    "好像没记过这个，要不先记一下？",
    "翻了一圈没找到~",
]
_CHAT_FALLBACK_REPLIES = [
    "嗯～我在呢，想聊啥或者想记啥都说哦~",
    "在呢在呢，说呗~",
    "听着呢，你说~",
]


def handle_intent(conn, vault_root: Path, user_text: str, call_llm_chat_fn=None, history: list | None = None) -> dict:
    """处理用户意图，返回结果 dict。
    返回格式: {'text': str, 'results': list, 'saved': bool, 'pending_save': str|None}

    Args:
        conn: 数据库连接
        vault_root: 记忆库路径
        user_text: 用户输入
        call_llm_chat_fn: LLM 聊天函数，签名 (user_query, history) -> str|None
        history: 对话历史 [{"role": "user"/"assistant", "content": "..."}]
    """
    intent = determine_intent(user_text)

    # ===== 保存意图 =====
    if intent == "save":
        need_wait, pending_text = check_save_pending(user_text)
        if need_wait:
            return {
                "text": random.choice(_SAVE_PENDING_REPLIES),
                "pending_save": pending_text,
                "results": [],
                "saved": False,
            }
        try:
            title, summary = build_memory_metadata(user_text)
            app_repo.remember_text_smart(conn, text=user_text, vault_root=vault_root, title=title, summary=summary)
            return {
                "text": random.choice(_SAVE_REPLIES),
                "saved": True,
                "results": [],
                "pending_save": None,
            }
        except Exception as e:
            return {"text": f"保存失败: {e}", "error": True, "results": [], "saved": False, "pending_save": None}

    # ===== 搜索意图 =====
    if intent == "search":
        results = search_memory(conn, user_text, limit=8)
        if results:
            ans = app_answer(user_text, results[:8])
            if ans and ans.answer:
                top_time = str(results[0].get("time", "") or "").strip()
                response_text = ans.answer
                if top_time:
                    response_text += f"\n（{top_time}）"
            else:
                if len(results) == 1:
                    r = results[0]
                    response_text = f"找到啦~ {r.get('title', '这条记忆')}"
                    if r.get("summary"):
                        response_text += f"：{r['summary']}"
                else:
                    response_text = f"找到 {len(results)} 条相关记忆~"
            return {"text": response_text, "results": results[:8], "saved": False, "pending_save": None}
        else:
            return {
                "text": random.choice(_SEARCH_NOT_FOUND_REPLIES),
                "results": [],
                "saved": False,
                "pending_save": None,
            }

    # ===== 聊天意图 =====
    if call_llm_chat_fn:
        response = call_llm_chat_fn(user_text, history or [])
    else:
        response = None

    if not response:
        response = random.choice(_CHAT_FALLBACK_REPLIES)
    return {"text": response, "results": [], "saved": False, "pending_save": None}


def confirm_save_action(text: str, pending_text: str) -> tuple[str, str | None]:
    """处理确认保存的逻辑。
    返回 (reply_text, final_text_to_save)。
    如果 final_text_to_save 为 None 表示取消或不明确。
    """
    decision = text.strip().lower()

    if decision in CONFIRM_WORDS:
        return (random.choice(["记好啦~", "收到~", "好嘞~", "记下来了~", "搞定~"]), pending_text)

    if decision in CANCEL_WORDS:
        return (random.choice(["好哒，那就不记啦~", "好嘞，不记了~", "行，不保存了~"]), None)

    # 用户说了具体内容，合并保存
    final_text = pending_text
    if text and text not in pending_text:
        final_text = pending_text + " " + text

    return (random.choice(["嗯嗯，帮你记好啦~", "收到~ 合并记下来了", "好嘞，都记下了~"]), final_text)
