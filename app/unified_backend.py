"""
统一后端门面 - 自动切换在线/离线模式
桌面端主程序只与此模块交互
"""
import threading
import time
from typing import Callable

from .api_client import MemoryApiClient, ApiError
from .offline_backend import OfflineBackend


class UnifiedBackend:
    """统一后端 - 优先使用远程 API，失败时自动降级到离线模式"""

    def __init__(self, api_client: MemoryApiClient, offline: OfflineBackend):
        self._api = api_client
        self._offline = offline
        self._online = False
        self._lock = threading.Lock()
        self._status_callbacks: list[Callable[[bool], None]] = []
        self._health_timer: threading.Timer | None = None

    @property
    def is_online(self) -> bool:
        return self._online

    def on_status_change(self, callback: Callable[[bool], None]):
        """注册连接状态变化回调"""
        self._status_callbacks.append(callback)

    def _set_online(self, online: bool):
        with self._lock:
            changed = self._online != online
            self._online = online
        if changed:
            for cb in self._status_callbacks:
                try:
                    cb(online)
                except Exception:
                    pass

    def check_server(self) -> bool:
        """检查服务端是否可用"""
        try:
            result = self._api.health()
            ok = result.get("status") == "ok"
            self._set_online(ok)
            return ok
        except Exception:
            self._set_online(False)
            return False

    def start_health_check(self, interval: int = 30):
        """启动定期健康检查"""
        def _tick():
            self.check_server()
            self._health_timer = threading.Timer(interval, _tick)
            self._health_timer.daemon = True
            self._health_timer.start()
        self._health_timer = threading.Timer(interval, _tick)
        self._health_timer.daemon = True
        self._health_timer.start()

    def stop_health_check(self):
        if self._health_timer:
            self._health_timer.cancel()
            self._health_timer = None

    def _call(self, method: str, *args, **kwargs):
        """统一调用入口 - 优先 API，失败降级"""
        if self._online:
            try:
                fn = getattr(self._api, method)
                return fn(*args, **kwargs)
            except (ApiError, OSError, ConnectionError, TimeoutError) as e:
                # 网络错误降级到离线
                self._set_online(False)
        # 离线模式
        fn = getattr(self._offline, method)
        return fn(*args, **kwargs)

    # ---- 代理所有方法 ----

    def chat(self, text: str) -> dict:
        return self._call("chat", text)

    def confirm_save(self, text: str, pending_text: str) -> dict:
        return self._call("confirm_save", text, pending_text)

    def search(self, query: str) -> dict:
        return self._call("search", query)

    def search_advanced(self, query: str = "", tags=None, date_from=None, date_to=None, limit=20) -> dict:
        return self._call("search_advanced", query, tags, date_from, date_to, limit)

    def save(self, text: str) -> dict:
        return self._call("save", text)

    def upload(self, file_paths: list, caption: str = "") -> dict:
        return self._call("upload", file_paths, caption)

    def list_memories(self, limit: int = 30) -> dict:
        return self._call("list_memories", limit)

    def get_memory(self, memory_id: int) -> dict:
        return self._call("get_memory", memory_id)

    def update_memory(self, memory_id: int, content: str) -> dict:
        return self._call("update_memory", memory_id, content)

    def delete_memory(self, memory_id: int) -> dict:
        return self._call("delete_memory", memory_id)

    def get_tags(self) -> dict:
        return self._call("get_tags")

    def get_stats(self) -> dict:
        return self._call("get_stats")

    def clear_history(self) -> dict:
        return self._call("clear_history")

    def health(self) -> dict:
        return self._call("health")

    # ---- 离线专属方法 ----

    def understand_image_local(self, memory_id: int, file_path: str) -> dict:
        """本地图片理解（仅离线模式使用）"""
        return self._offline.understand_image_local(memory_id, file_path)
