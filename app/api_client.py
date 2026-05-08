"""
HTTP API 客户端 - 桌面端通过此模块与 Flask 后端通信
所有方法为同步调用，应在线程中执行以避免阻塞 UI
"""
import json
import mimetypes
import uuid
import urllib.request
import urllib.error
from pathlib import Path


class ApiError(Exception):
    """API 调用异常"""
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


class MemoryApiClient:
    """记忆助手 HTTP API 客户端"""

    def __init__(self, base_url: str = "https://memory-n.ccwu.cc", client_id: str = "", timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id
        self.timeout = timeout

    def _make_request(self, method: str, path: str, data=None, files=None) -> dict:
        """发送 HTTP 请求"""
        url = f"{self.base_url}{path}"
        headers = {"X-Client-Id": self.client_id} if self.client_id else {}

        if files:
            boundary = uuid.uuid4().hex
            body = self._build_multipart(data or {}, files, boundary)
            headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
            req = urllib.request.Request(url, data=body, headers=headers, method=method)
        elif data is not None:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
            req = urllib.request.Request(url, data=body, headers=headers, method=method)
        else:
            req = urllib.request.Request(url, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
                if not raw:
                    return {}
                return json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                err_body = e.read()
                err_data = json.loads(err_body.decode("utf-8"))
                msg = err_data.get("error", str(e))
            except Exception:
                msg = err_body.decode("utf-8", errors="replace")[:200] if err_body else str(e)
            raise ApiError(msg, status_code=e.code)
        except urllib.error.URLError as e:
            raise ApiError(f"连接失败: {e.reason}")
        except Exception as e:
            raise ApiError(str(e))

    def _build_multipart(self, fields: dict, files: list, boundary: str) -> bytes:
        """构造 multipart/form-data 请求体"""
        body = bytearray()
        for key, value in fields.items():
            body += f'--{boundary}\r\n'.encode()
            body += f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode()
            body += f'{value}\r\n'.encode()
        for file_path in files:
            p = Path(file_path)
            filename = p.name.replace('"', '_')
            mime = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
            body += f'--{boundary}\r\n'.encode()
            body += f'Content-Disposition: form-data; name="files"; filename="{filename}"\r\n'.encode()
            body += f'Content-Type: {mime}\r\n\r\n'.encode()
            try:
                body += p.read_bytes()
            except Exception as e:
                raise ApiError(f"读取文件失败: {file_path}: {e}")
            body += b'\r\n'
        body += f'--{boundary}--\r\n'.encode()
        return bytes(body)

    # ---- 核心聊天 ----

    def chat(self, text: str) -> dict:
        """主聊天接口 - 自动判断意图（保存/搜索/聊天）"""
        return self._make_request("POST", "/api/chat", data={"text": text})

    def confirm_save(self, text: str, pending_text: str) -> dict:
        """确认保存待保存内容"""
        return self._make_request("POST", "/api/chat/confirm_save",
                                  data={"text": text, "pending_text": pending_text})

    # ---- 搜索 ----

    def search(self, query: str) -> dict:
        """简单搜索"""
        return self._make_request("POST", "/api/search", data={"query": query})

    def search_advanced(self, query: str = "", tags=None, date_from=None, date_to=None, limit=20) -> dict:
        """高级搜索"""
        return self._make_request("POST", "/api/memories/search", data={
            "query": query, "tags": tags or [],
            "date_from": date_from, "date_to": date_to, "limit": limit,
        })

    # ---- 保存 ----

    def save(self, text: str) -> dict:
        """直接保存文本"""
        return self._make_request("POST", "/api/save", data={"text": text})

    def upload(self, file_paths: list, caption: str = "") -> dict:
        """上传文件（multipart）"""
        return self._make_request("POST", "/api/upload",
                                  data={"caption": caption}, files=file_paths)

    # ---- 记忆管理 ----

    def list_memories(self, limit: int = 30) -> dict:
        """列出最近记忆"""
        return self._make_request("GET", f"/api/memories?limit={limit}")

    def get_memory(self, memory_id: int) -> dict:
        """获取单条记忆"""
        return self._make_request("GET", f"/api/memory/{memory_id}")

    def update_memory(self, memory_id: int, content: str) -> dict:
        """更新记忆内容"""
        return self._make_request("PUT", f"/api/memory/{memory_id}",
                                  data={"content": content})

    def delete_memory(self, memory_id: int) -> dict:
        """删除记忆"""
        return self._make_request("DELETE", f"/api/memory/{memory_id}")

    # ---- 标签和统计 ----

    def get_tags(self) -> dict:
        """获取所有标签"""
        return self._make_request("GET", "/api/tags")

    def get_stats(self) -> dict:
        """获取统计信息"""
        return self._make_request("GET", "/api/stats")

    # ---- 聊天历史 ----

    def clear_history(self) -> dict:
        """清除聊天历史"""
        return self._make_request("POST", "/api/clear")

    # ---- 语音（桌面端通常本地处理，但保留远程能力） ----

    def speech_recognize(self, audio_bytes: bytes) -> dict:
        """语音识别（通过服务端 Baidu ASR）"""
        boundary = uuid.uuid4().hex
        body = bytearray()
        body += f'--{boundary}\r\n'.encode()
        body += f'Content-Disposition: form-data; name="audio"; filename="audio.wav"\r\n'.encode()
        body += f'Content-Type: audio/wav\r\n\r\n'.encode()
        body += audio_bytes
        body += b'\r\n'
        body += f'--{boundary}--\r\n'.encode()

        url = f"{self.base_url}/api/speech_recognize"
        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
        if self.client_id:
            headers["X-Client-Id"] = self.client_id
        req = urllib.request.Request(url, data=bytes(body), headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
                return json.loads(raw.decode("utf-8")) if raw else {}
        except urllib.error.HTTPError as e:
            try:
                err_body = e.read()
                err_data = json.loads(err_body.decode("utf-8"))
                msg = err_data.get("error", str(e))
            except Exception:
                msg = err_body.decode("utf-8", errors="replace")[:200] if err_body else str(e)
            raise ApiError(msg, status_code=e.code)
        except urllib.error.URLError as e:
            raise ApiError(f"连接失败: {e.reason}")
        except Exception as e:
            raise ApiError(str(e))

    def speech_synthesize(self, text: str) -> dict:
        """语音合成（通过服务端 Baidu TTS）"""
        return self._make_request("POST", "/api/speech_synthesize", data={"text": text})

    # ---- 健康检查 ----

    def health(self) -> dict:
        """健康检查"""
        return self._make_request("GET", "/api/health")

    def get_version(self) -> dict:
        """获取服务端版本信息"""
        return self._make_request("GET", "/api/version")
