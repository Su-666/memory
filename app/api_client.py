"""
HTTP API 客户端 - 桌面端通过此模块与 Flask 后端通信
所有方法为同步调用，应在线程中执行以避免阻塞 UI

v2: 增加重试、退避、SSL 修复、超时分离、错误分类
"""
import json
import mimetypes
import os
import socket
import ssl
import time
import uuid
import urllib.request
import urllib.error
from pathlib import Path

# SSL 证书修复（打包后常见问题）
try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()


class ApiError(Exception):
    """API 调用异常"""
    def __init__(self, message, status_code=None, retryable=False):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable  # 是否值得重试


class MemoryApiClient:
    """记忆助手 HTTP API 客户端（带重试和退避）"""

    # 重试配置
    MAX_RETRIES = 3
    RETRY_DELAYS = [1.0, 2.5, 5.0]  # 指数退避秒数

    def __init__(self, base_url: str = "https://memory-n.ccwu.cc", client_id: str = "", timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id
        self.timeout = timeout
        self._connected = False  # 连接状态缓存

    @property
    def connected(self) -> bool:
        return self._connected

    def _make_request(self, method: str, path: str, data=None, files=None, _retry=True) -> dict:
        """发送 HTTP 请求（带自动重试）"""
        last_error = None
        attempts = self.MAX_RETRIES if _retry else 1

        for attempt in range(attempts):
            try:
                result = self._do_request(method, path, data, files)
                self._connected = True
                return result
            except ApiError as e:
                last_error = e
                # 4xx 客户端错误不重试（除了 408/429）
                if e.status_code and 400 <= e.status_code < 500:
                    if e.status_code not in (408, 429):
                        raise
                # 可重试的错误：网络/超时/5xx
                if attempt < attempts - 1:
                    delay = self.RETRY_DELAYS[min(attempt, len(self.RETRY_DELAYS) - 1)]
                    time.sleep(delay)
            except Exception as e:
                last_error = ApiError(str(e), retryable=True)
                if attempt < attempts - 1:
                    delay = self.RETRY_DELAYS[min(attempt, len(self.RETRY_DELAYS) - 1)]
                    time.sleep(delay)

        # 所有重试都失败
        self._connected = False
        raise last_error

    def _do_request(self, method: str, path: str, data=None, files=None) -> dict:
        """执行单次 HTTP 请求"""
        url = f"{self.base_url}{path}"
        headers = {
            "X-Client-Id": self.client_id,
            "Accept": "application/json",
        } if self.client_id else {"Accept": "application/json"}

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
            with urllib.request.urlopen(req, timeout=self.timeout, context=_SSL_CTX) as resp:
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
            raise ApiError(msg, status_code=e.code, retryable=e.code >= 500)
        except socket.timeout:
            raise ApiError("服务器响应超时，请稍后重试", retryable=True)
        except urllib.error.URLError as e:
            reason = str(e.reason) if hasattr(e, 'reason') else str(e)
            if "timed out" in reason.lower():
                raise ApiError("连接超时，请检查网络", retryable=True)
            elif "connection refused" in reason.lower():
                raise ApiError("服务器暂时不可用", retryable=True)
            elif "name resolution" in reason.lower() or "getaddrinfo" in reason.lower():
                raise ApiError("无法解析服务器地址，请检查网络", retryable=True)
            elif "ssl" in reason.lower() or "certificate" in reason.lower():
                raise ApiError("SSL 证书验证失败", retryable=False)
            raise ApiError(f"网络连接失败: {reason}", retryable=True)
        except ssl.SSLError as e:
            raise ApiError(f"安全连接失败: {e}", retryable=False)
        except json.JSONDecodeError:
            raise ApiError("服务器返回了无效的数据", retryable=True)

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
        """语音识别（通过服务端 Baidu ASR）- 带重试"""
        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                boundary = uuid.uuid4().hex
                body = bytearray()
                body += f'--{boundary}\r\n'.encode()
                body += f'Content-Disposition: form-data; name="audio"; filename="audio.wav"\r\n'.encode()
                body += f'Content-Type: audio/wav\r\n\r\n'.encode()
                body += audio_bytes
                body += b'\r\n'
                body += f'--{boundary}--\r\n'.encode()

                url = f"{self.base_url}/api/speech_recognize"
                headers = {
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                    "Accept": "application/json",
                }
                if self.client_id:
                    headers["X-Client-Id"] = self.client_id
                req = urllib.request.Request(url, data=bytes(body), headers=headers, method="POST")

                with urllib.request.urlopen(req, timeout=self.timeout, context=_SSL_CTX) as resp:
                    raw = resp.read()
                    self._connected = True
                    return json.loads(raw.decode("utf-8")) if raw else {}
            except ApiError:
                raise
            except socket.timeout:
                last_error = ApiError("语音识别超时", retryable=True)
            except urllib.error.HTTPError as e:
                try:
                    err_body = e.read()
                    err_data = json.loads(err_body.decode("utf-8"))
                    msg = err_data.get("error", str(e))
                except Exception:
                    msg = str(e)
                last_error = ApiError(msg, status_code=e.code, retryable=e.code >= 500)
                if e.code < 500 and e.code not in (408, 429):
                    raise last_error
            except urllib.error.URLError as e:
                last_error = ApiError(f"语音识别网络失败: {e.reason}", retryable=True)
            except Exception as e:
                last_error = ApiError(f"语音识别失败: {e}", retryable=True)

            if attempt < self.MAX_RETRIES - 1:
                time.sleep(self.RETRY_DELAYS[min(attempt, len(self.RETRY_DELAYS) - 1)])

        self._connected = False
        raise last_error

    def speech_synthesize(self, text: str) -> dict:
        """语音合成（通过服务端 Baidu TTS）"""
        return self._make_request("POST", "/api/speech_synthesize", data={"text": text})

    # ---- 健康检查 ----

    def health(self) -> dict:
        """健康检查"""
        return self._make_request("GET", "/api/health")

    def health_fast(self) -> bool:
        """快速健康检查（短超时，不重试）"""
        try:
            url = f"{self.base_url}/api/health"
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=8, context=_SSL_CTX) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                ok = data.get("status") == "ok"
                self._connected = ok
                return ok
        except Exception:
            self._connected = False
            return False

    def get_version(self) -> dict:
        """获取服务端版本信息"""
        return self._make_request("GET", "/api/version")
