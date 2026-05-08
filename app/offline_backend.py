"""
离线后端 - 直接调用 app/ 模块，无需 Flask 服务器
当服务端不可用时，桌面端通过此模块继续工作
"""
import sqlite3
from pathlib import Path

from . import db as app_db
from . import repo as app_repo
from . import search as app_search
from .answer import answer as app_answer
from .intent_chat import handle_intent, confirm_save_action, build_memory_metadata
from .llm import call_llm_chat
from .vault import ensure_vault_root
from .vision import understand_image


class OfflineBackend:
    """离线后端 - 封装本地 app/ 模块，提供与 MemoryApiClient 相同的接口"""

    def __init__(self, conn: sqlite3.Connection, vault_root: Path):
        self._conn = conn
        self._vault_root = vault_root
        self._chat_history: list[dict] = []
        self._max_history = 50

    def _get_history(self) -> list[dict]:
        return list(self._chat_history[-10:])

    def _append_history(self, role: str, content: str):
        self._chat_history.append({"role": role, "content": content})
        if len(self._chat_history) > self._max_history:
            self._chat_history = self._chat_history[-self._max_history:]

    # ---- 核心聊天 ----

    def chat(self, text: str) -> dict:
        """主聊天接口 - 复用 intent_chat.handle_intent"""
        def llm_chat_fn(query):
            history = self._get_history()
            self._append_history("user", query)
            response = call_llm_chat(query, history)
            if response:
                self._append_history("assistant", response)
            return response

        result = handle_intent(self._conn, self._vault_root, text, call_llm_chat_fn=llm_chat_fn)
        resp = {"type": "assistant", "text": result["text"]}
        if result.get("results"):
            resp["results"] = result["results"]
        if result.get("pending_save"):
            resp["pending_save"] = result["pending_save"]
        if result.get("saved"):
            resp["saved"] = True
        return resp

    def confirm_save(self, text: str, pending_text: str) -> dict:
        """确认保存"""
        reply, final_text = confirm_save_action(text, pending_text)
        if final_text:
            title, summary = build_memory_metadata(final_text)
            app_repo.remember_text_smart(
                self._conn, text=final_text,
                vault_root=self._vault_root, title=title, summary=summary,
            )
        return {"type": "assistant", "text": reply}

    # ---- 搜索 ----

    def search(self, query: str) -> dict:
        results = app_search.search(self._conn, query=query, sort_mode="relevant", limit=20)
        return {"results": results}

    def search_advanced(self, query: str = "", tags=None, date_from=None, date_to=None, limit=20) -> dict:
        conditions = {
            "query": query, "tags": tags or [],
            "date_from": date_from, "date_to": date_to, "limit": limit,
        }
        results = app_repo.search_advanced(self._conn, conditions)
        return {"results": results, "total": len(results)}

    # ---- 保存 ----

    def save(self, text: str) -> dict:
        title, summary = build_memory_metadata(text)
        app_repo.remember_text_smart(
            self._conn, text=text,
            vault_root=self._vault_root, title=title, summary=summary,
        )
        return {"success": True, "message": "已保存"}

    def upload(self, file_paths: list, caption: str = "") -> dict:
        """上传文件到本地 vault"""
        from pathlib import Path
        valid_paths = []
        for p in file_paths:
            fp = Path(p)
            if fp.exists():
                valid_paths.append(str(fp))
        if not valid_paths:
            return {"error": "没有有效文件"}
        ids = app_repo.remember_attachments(
            self._conn, paths=valid_paths, caption=caption, vault_root=self._vault_root,
        )
        return {"success": True, "message": f"已保存 {len(ids)} 个文件"}

    def understand_image_local(self, memory_id: int, file_path: str) -> dict:
        """本地图片理解（离线模式）"""
        u = understand_image(file_path)
        body_parts = []
        if u.tags:
            body_parts.append("标签：" + "、".join(str(t) for t in u.tags if str(t).strip()))
        if u.text_in_image:
            body_parts.append("图片文字：" + u.text_in_image)
        new_body = "\n".join(body_parts).strip()
        app_repo.update_memory(
            self._conn, memory_id=memory_id, summary=u.caption or None,
            body=new_body or None,
            extra_patch={"vision": {"caption": u.caption, "tags": u.tags, "text_in_image": u.text_in_image}},
            commit=True,
        )
        return {"success": True}

    # ---- 记忆管理 ----

    def list_memories(self, limit: int = 30) -> dict:
        items = app_repo.list_recent(self._conn, limit=limit)
        return {"memories": items}

    def get_memory(self, memory_id: int) -> dict:
        m = app_repo.get_memory(self._conn, memory_id)
        if not m:
            return {"error": "记忆不存在"}
        return {"memory": m}

    def update_memory(self, memory_id: int, content: str) -> dict:
        app_repo.update_memory(self._conn, memory_id=memory_id, body=content)
        return {"success": True, "message": "记忆更新成功"}

    def delete_memory(self, memory_id: int) -> dict:
        app_repo.delete_memory(self._conn, memory_id)
        return {"success": True, "message": "记忆删除成功"}

    # ---- 标签和统计 ----

    def get_tags(self) -> dict:
        try:
            tags = app_repo.get_all_tags(self._conn)
        except AttributeError:
            recent = app_repo.list_recent(self._conn, limit=100)
            tag_set = set()
            for item in recent:
                if "tags" in item and item["tags"]:
                    tag_set.update(item["tags"])
            tags = list(tag_set)
        return {"tags": tags}

    def get_stats(self) -> dict:
        total = app_repo.get_total_memories(self._conn)
        recent = app_repo.list_recent(self._conn, limit=10)
        tags = app_repo.get_all_tags(self._conn)
        return {
            "total_memories": total,
            "recent_count": len(recent),
            "tags": tags[:20],
            "last_updated": recent[0].get("time") if recent else None,
        }

    # ---- 聊天历史 ----

    def clear_history(self) -> dict:
        self._chat_history.clear()
        return {"success": True}

    # ---- 健康检查（离线模式永远返回 ok） ----

    def health(self) -> dict:
        return {"status": "offline", "service": "nuan-nuan-memory", "version": "5.1"}
