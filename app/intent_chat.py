"""
统一意图判断、保存确认、搜索查询提取、记忆元数据构建。
桌面端和 Web 端共用此模块，避免逻辑重复和不一致。
"""
from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path
from typing import Any

from . import repo as app_repo
from . import search as app_search
from .answer import _local_answer
from .zhipu_client import call_chat

logger = logging.getLogger(__name__)


# ============ 意图判断 ============

def determine_intent(text: str) -> str:
    """判断用户意图：save / search / chat
    检测顺序：明确命令 -> 保存关键词 -> 搜索关键词 -> 搜索问题模式 -> 问候 -> 默认 chat
    """
    stripped = text.strip()
    if not stripped:
        return "chat"

    # 明确命令
    if stripped.startswith(("/聊", "/chat")):
        return "chat"
    if stripped.startswith(("/搜", "/search")):
        return "search"

    # ===== 保存意图（优先级最高） =====
    save_keywords = (
        "帮我记", "帮我存", "帮我保存", "帮我备注", "帮我写上",
        "记一下", "存一下", "记录一下", "保存一下", "备注一下",
        "标记一下", "标注一下",
        "记住", "记下来", "记下来吧", "存下来", "写下来",
        "收藏", "备忘", "留个底", "存着", "mark一下",
        "记着", "记上", "存档", "存个档",
        "提醒我", "别忘了", "要记住",
        "把这个记下来", "添加到记忆", "加入记忆",
    )
    if any(k in stripped for k in save_keywords):
        return "save"

    # ===== 搜索意图 =====
    search_keywords = (
        "帮我找", "帮我查", "帮我搜",
        "查一下", "搜一下", "找一下", "查查", "找找", "搜搜",
        "查找", "找到", "搜索", "检索", "查询",
        "找找看", "搜搜看", "找出来", "调出来", "翻一下",
        "记得", "我记过", "有没有记过",
        "之前记", "以前记", "我之前记", "上次记",
        "之前存", "以前存", "我之前存",
        "我存过", "保存过", "记录过",
        "我的记忆", "记忆库", "记忆里",
        "告诉我", "我想知道", "看看我记了什么",
    )
    if any(k in stripped for k in search_keywords):
        return "search"

    # 搜索问题模式（"我的X是什么/多少/在哪"）
    _search_data_kw = (
        "密码", "手机号", "电话", "电话号码", "地址", "生日",
        "邮箱", "卡号", "账号", "身份证", "车牌", "银行卡",
    )
    _question_patterns = ("是什么", "是多少", "在哪", "在哪里")
    if "我的" in stripped and any(kw in stripped for kw in _search_data_kw):
        if any(p in stripped for p in _question_patterns):
            return "search"

    memory_question_keywords = (
        "我记了什么", "记得什么", "存了什么", "保存了什么",
        "记忆里有", "记忆库有",
        "密码是什么", "密码是多少", "密码忘了", "密码多少",
        "手机号是什么", "手机号是多少", "手机号多少",
        "电话号码", "电话是多少", "号码是多少",
        "地址是什么", "地址是多少", "地址在哪",
        "生日是什么", "生日是哪天", "生日是多少",
        "身份证号", "身份证号码",
        "我的密码", "我的手机号", "我的电话", "我的地址",
        "我的生日", "我的邮箱", "我的卡号", "我的账号",
        "我的身份证", "我的车牌", "我的银行卡",
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


def _has_actual_value(text: str) -> bool:
    """检测文本是否包含具体值（数字、邮箱、URL、日期等）"""
    if any(c.isdigit() for c in text):
        return True
    if "@" in text and "." in text:
        return True
    if len(text) > 10:
        return True
    if re.search(r'https?://', text):
        return True
    if re.search(r'\d{1,2}[月:\-/]\d{1,2}', text):
        return True
    return False


def check_save_pending(text: str) -> tuple[bool, str | None]:
    """检查用户是否只表达了保存意图但没有具体内容。
    返回 (need_wait, pending_text)
    """
    text_stripped = text.strip()

    save_verbs = [
        "帮我记住", "帮我记", "帮我存", "帮我保存", "帮我备注", "帮我写上",
        "记一下", "存一下", "记录一下", "保存一下", "备注一下",
        "标记一下", "标注一下",
        "记下来", "存下来", "写下来", "记下来吧",
        "备忘", "别忘了", "提醒我", "留个底",
    ]

    for verb in save_verbs:
        if text_stripped.startswith(verb):
            after = text_stripped[len(verb):].strip()
            # verb 后没有内容
            if not after:
                return (True, text_stripped)

            # 数据关键词（涵盖日常生活中需要记忆的各类信息）
            data_keywords = [
                "密码", "手机号", "电话号码", "手机号码", "电话", "地址",
                "生日", "邮箱", "卡号", "账号", "身份证", "车牌", "银行卡",
                "社保", "护照", "驾照", "工号", "学号",
                "QQ号", "微信号", "游戏账号", "IP地址", "WiFi密码", "wifi密码",
                "网站", "网址", "服务器", "银行卡号", "信用卡", "会员卡",
                "门牌号", "楼层", "会议室", "收货地址", "公司地址", "家庭地址",
                "邮编", "紧急联系人", "过敏信息", "血型", "病历号", "医保号",
                "签证", "股票", "基金", "行李箱密码", "保险箱密码", "密码锁",
            ]

            has_data_keyword = any(kw in after for kw in data_keywords)

            # 有数据关键词但没有实际值 → 追问
            if has_data_keyword and not _has_actual_value(after):
                return (True, text_stripped)

            # 纯数据类型模式检测（去掉所有格和修饰词后只剩数据类型词）
            text_no_verb = after
            for p in ("我的", "你的", "他的", "她的", "它的", "咱们的", "我家的"):
                text_no_verb = text_no_verb.replace(p, "")
            for m in ("手机", "电脑", "银行", "邮箱", "网站", "游戏",
                       "qq", "QQ", "微信", "微博", "支付宝", "淘宝",
                       "家里", "公司", "学校", "医院"):
                text_no_verb = text_no_verb.replace(m, "")
            text_no_verb = text_no_verb.strip()
            pure_type_words = [
                "密码", "手机号", "电话", "电话号码", "地址", "生日",
                "邮箱", "卡号", "账号", "身份证", "车牌", "银行卡",
                "QQ号", "微信号", "WiFi密码",
            ]
            if text_no_verb in pure_type_words:
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

def build_memory_metadata_fast(text: str) -> tuple[str, str]:
    """从用户输入中提取标题和摘要（仅本地逻辑，不调用 LLM）"""
    title = ""
    if "我的" in text and ("是" in text or ":" in text or "：" in text):
        parts = text.replace("：", ":").split(":")
        if len(parts) >= 2:
            title = parts[0].strip()[:20]
        else:
            parts = text.split("是")
            if len(parts) >= 2:
                title = ("我的" + parts[1].strip())[:20] if "我的" in parts[0] else parts[0].strip()[:20]

    if not title:
        keywords = [
            "密码", "手机号", "电话号码", "手机号码", "电话", "地址", "生日", "邮箱",
            "卡号", "账号", "身份证", "车牌", "银行卡", "银行卡号",
            "信用卡", "会员卡", "QQ号", "微信号", "游戏账号", "WiFi密码",
            "收货地址", "公司地址", "紧急联系人", "保险箱密码", "密码锁",
        ]
        for kw in keywords:
            if kw in text:
                title = f"我的{kw}"
                break

    fallback_title = title or ((text[:18] + "…") if len(text) > 18 else (text or "记忆"))
    fallback_summary = (text[:80] + "…") if len(text) > 80 else (text or "记忆")
    return (fallback_title, fallback_summary)


def build_memory_metadata_llm(text: str) -> tuple[str, str]:
    """用 LLM 从用户原文中提取标题和摘要，失败时回退到本地规则。"""
    try:
        messages = [
            {"role": "system", "content": "你是一个严格输出 JSON 的中文助手。"},
            {"role": "user", "content": (
                "请从以下用户输入中提取一个简短标题和一句话摘要。\n"
                "要求：\n"
                "1) title：10字以内，概括核心内容\n"
                "2) summary：30字以内，保留关键信息（数字、日期、名称等不要遗漏）\n"
                "3) 只输出严格 JSON，不要输出其他内容\n"
                "输出格式：\n"
                '{"title": "...", "summary": "..."}\n'
                f"用户输入：{text}"
            )},
        ]
        data = call_chat(messages, temperature=0.1, max_tokens=150, timeout=10, retries=1)
        content = str(data["choices"][0]["message"]["content"]).strip()
        # 解析 JSON
        if "{" in content:
            json_str = content[content.index("{"):content.rindex("}") + 1]
            parsed = json.loads(json_str)
            title = str(parsed.get("title", "")).strip()
            summary = str(parsed.get("summary", "")).strip()
            if title and summary:
                return (title[:20], summary[:80])
    except Exception as e:
        logger.warning("LLM 元数据提取失败，回退到本地规则: %s", e)
    return build_memory_metadata_fast(text)


# ============ 搜索记忆 ============

def search_memory(conn, user_text: str, limit: int = 5) -> list:
    """统一搜索记忆，智能扩展搜索词"""
    query = extract_search_query(user_text)

    # 先用完整查询搜索，如果已有足够结果就不再扩展
    results: list[dict[str, Any]] = []
    seen_ids: set[int] = set()

    try:
        items = app_search.search(conn, query=query, sort_mode="relevant", limit=20)
        for item in items:
            item_id = item.get("id") or item.get("entity_id")
            if item_id and item_id not in seen_ids:
                seen_ids.add(item_id)
                results.append(item)
    except Exception:
        pass

    # 结果不足时才用关键词扩展搜索
    if len(results) < limit:
        important_keywords = [
            "手机号", "电话号码", "手机号码", "电话", "密码", "地址", "生日", "邮箱",
            "卡号", "账号", "身份证", "银行卡", "QQ号", "微信号", "WiFi密码",
        ]
        for kw in important_keywords:
            if kw in query and kw not in seen_ids:
                try:
                    items = app_search.search(conn, query=kw, sort_mode="relevant", limit=10)
                    for item in items:
                        item_id = item.get("id") or item.get("entity_id")
                        if item_id and item_id not in seen_ids:
                            seen_ids.add(item_id)
                            results.append(item)
                except Exception:
                    continue
                if len(results) >= limit:
                    break

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
            title, summary = build_memory_metadata_llm(user_text)
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
            # 仅用本地规则提取精确答案（零延迟，不调用 LLM）
            contexts = []
            for m in results[:6]:
                contexts.append(str(m.get("title", "")))
                contexts.append(str(m.get("summary", "")))
                contexts.append(str(m.get("body_snippet", "")))
            local_ans = _local_answer(user_text, contexts)
            if local_ans.answer:
                top_time = str(results[0].get("time", "") or "").strip()
                response_text = local_ans.answer
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
