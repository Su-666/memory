"""
暖暖记忆助手 - Web 版
独立部署入口，不依赖桌面端代码
"""
import os
import sys
import base64
import json
import threading
import re
import time
import uuid
from collections import OrderedDict
from functools import wraps
from pathlib import Path

# 优先设置导入路径
from import_setup import setup_paths
setup_paths()

from flask import Flask, jsonify, request, send_from_directory, Response
from flask_cors import CORS

# 版本信息
APP_VERSION = "6.0"
CHANGELOG = "全新界面设计，在线模式，暗色主题支持"
DOWNLOAD_URL = ""

# Railway 使用环境变量 PORT
PORT = int(os.environ.get("PORT", 5000))

# 项目根目录（web/ 的父目录）
project_root = Path(__file__).resolve().parent.parent

# 确保项目根目录在 Python 路径中
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 数据目录 - 默认使用根目录下的 data/
DATA_DIR = Path(os.environ.get("DATA_DIR", str(project_root / "data")))
VAULT_DIR = Path(os.environ.get("VAULT_DIR", str(project_root / "data" / "memory_vault")))
DATA_DIR.mkdir(parents=True, exist_ok=True)
VAULT_DIR.mkdir(parents=True, exist_ok=True)

# 导入应用模块
try:
    from app import db as app_db
    from app import repo as app_repo
    from app import search as app_search
    from app.vault import ensure_vault_root
    from app.vision import understand_image
    from app.intent_chat import (
        build_memory_metadata_fast,
        handle_intent,
        confirm_save_action,
    )
    from app.llm import call_llm_chat
except ModuleNotFoundError as e:
    print(f"[ERROR] Failed to import app module: {e}", file=sys.stderr, flush=True)
    raise

# Flask 应用
app = Flask(__name__, static_folder='static', static_url_path='/static')
ALLOWED_ORIGINS = os.environ.get("CORS_ORIGINS", "*")
CORS(app, resources={r"/api/*": {"origins": ALLOWED_ORIGINS}})

# 加载环境变量（复用 app.utils 的统一实现）
from app.utils import load_env_file
load_env_file(str(project_root / ".env"))

# ============ 优化：数据库单例连接 ============
_db_conn = None
_db_conn_lock = threading.Lock()
_vault_initialized = False

def get_db_conn():
    """获取数据库连接（单例，线程安全）"""
    global _db_conn
    if _db_conn is not None:
        return _db_conn
    with _db_conn_lock:
        if _db_conn is not None:
            return _db_conn
        db_path = DATA_DIR / "agent.db"
        conn = app_db.connect(db_path)
        app_db.init_db(conn)
        _db_conn = conn
        return conn

def get_vault_root():
    ensure_vault_root(VAULT_DIR)
    return VAULT_DIR

def init_vault_once():
    """仅在启动时调用一次，扫描文件系统导入记忆"""
    global _vault_initialized
    if _vault_initialized:
        return
    _vault_initialized = True
    vault_root = get_vault_root()
    ensure_vault_root(vault_root)
    conn = get_db_conn()
    try:
        imported = app_repo.bootstrap_import_vault(conn, vault_root)
        if imported:
            print(f"[STARTUP] 导入 {imported} 条新记忆", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[STARTUP] vault bootstrap import failed: {e}", file=sys.stderr, flush=True)
    # 仅在本地环境清理失效文件记忆（Railway 上数据目录是持久化的，不需要清理）
    if not os.environ.get("RAILWAY_STATIC_URL"):
        try:
            removed = app_repo.prune_missing_file_memories(conn)
            if removed:
                print(f"[STARTUP] 清理 {removed} 条失效记忆", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[STARTUP] prune failed: {e}", file=sys.stderr, flush=True)

# ============ 优化：per-client 聊天历史 ============
_client_chat_histories: dict[str, list[dict]] = {}
_chat_history_lock = threading.Lock()
_MAX_HISTORY_PER_CLIENT = 50

def _get_client_id() -> str:
    """获取客户端标识 - 优先使用 X-Client-Id 请求头，回退到 IP"""
    client_id = request.headers.get("X-Client-Id", "").strip()
    if client_id:
        client_id = re.sub(r'[^a-zA-Z0-9_\-]', '', client_id)[:64]
        if client_id:
            return client_id
    return request.remote_addr or "unknown"

def _get_chat_history(client_id: str) -> list[dict]:
    with _chat_history_lock:
        return list(_client_chat_histories.get(client_id, [])[-10:])

def _append_chat_history(client_id: str, role: str, content: str):
    with _chat_history_lock:
        if client_id not in _client_chat_histories:
            _client_chat_histories[client_id] = []
        _client_chat_histories[client_id].append({"role": role, "content": content})
        # 限制历史长度，防止内存泄漏
        if len(_client_chat_histories[client_id]) > _MAX_HISTORY_PER_CLIENT:
            _client_chat_histories[client_id] = _client_chat_histories[client_id][-_MAX_HISTORY_PER_CLIENT:]

def _clear_chat_history(client_id: str):
    with _chat_history_lock:
        _client_chat_histories.pop(client_id, None)

# ============ 优化功能 ============

class SimpleCache:
    def __init__(self, max_size=100, ttl=300):
        self.cache: OrderedDict = OrderedDict()  # key -> (value, timestamp)
        self.max_size = max_size
        self.ttl = ttl
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key in self.cache:
                data, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl:
                    self.cache.move_to_end(key)
                    return data
                else:
                    del self.cache[key]
            return None

    def set(self, key, value):
        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            elif len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
            self.cache[key] = (value, time.time())

    def invalidate(self, prefix):
        with self._lock:
            keys_to_remove = [k for k in self.cache if k.startswith(prefix)]
            for k in keys_to_remove:
                del self.cache[k]

response_cache = SimpleCache()


def _invalidate_caches():
    """数据变更后统一清除相关缓存"""
    response_cache.invalidate("recent_memories")
    response_cache.invalidate("stats")
    response_cache.invalidate("tags")


_storage_size_cache = {"size": 0, "time": 0}
_STORAGE_CACHE_TTL = 120  # 2 分钟缓存存储大小

class RateLimiter:
    def __init__(self, max_requests=20, window_seconds=60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}
        self._lock = threading.Lock()
        self._last_prune = time.time()
        self._prune_interval = 300

    def is_allowed(self, client_id):
        now = time.time()
        with self._lock:
            if client_id not in self.requests:
                self.requests[client_id] = []
            self.requests[client_id] = [
                t for t in self.requests[client_id]
                if now - t < self.window_seconds
            ]
            if not self.requests[client_id]:
                del self.requests[client_id]
            if now - self._last_prune > self._prune_interval:
                self._last_prune = now
                stale = [cid for cid, times in self.requests.items()
                         if not times or now - times[-1] > self.window_seconds * 2]
                for cid in stale:
                    del self.requests[cid]
            if client_id not in self.requests:
                self.requests[client_id] = []
            if len(self.requests[client_id]) < self.max_requests:
                self.requests[client_id].append(now)
                return True
            return False

rate_limiter = RateLimiter(max_requests=20, window_seconds=60)

_ALLOWED_UPLOAD_EXTENSIONS = frozenset({
    '.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg',
    '.pdf', '.doc', '.docx', '.txt', '.md', '.csv', '.json',
    '.mp3', '.wav', '.m4a', '.ogg',
})

def _safe_get_json():
    data = request.get_json(silent=True)
    return data if data is not None else {}

def rate_limit(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        client_id = _get_client_id()
        if not rate_limiter.is_allowed(client_id):
            return jsonify({'error': '请求过于频繁，请稍后再试'}), 429
        return f(*args, **kwargs)
    return decorated_function

def cached_response(cache_key_func=None, ttl=300):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                cache_key = f"{request.path}:{hash(str(dict(request.args)))}"
            cached = response_cache.get(cache_key)
            if cached:
                return cached
            response = f(*args, **kwargs)
            if response and hasattr(response, 'get_json'):
                try:
                    response_data = response.get_json()
                    response_cache.set(cache_key, response_data)
                except Exception:
                    pass
            return response
        return decorated_function
    return decorator

def performance_monitor(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        try:
            result = f(*args, **kwargs)
            duration = time.time() - start_time
            print(f"[PERF] {f.__name__}: {duration:.3f}s", file=sys.stderr, flush=True)
            return result
        except Exception as e:
            duration = time.time() - start_time
            print(f"[ERROR] {f.__name__}: {duration:.3f}s - {e}", file=sys.stderr, flush=True)
            raise
    return decorated_function


# HTTP Cache-Control headers by endpoint path prefix
_CACHE_MAX_AGES = {
    '/api/tags': 300,
    '/api/stats': 60,
    '/api/memories/recent': 30,
    '/api/version': 3600,
}


@app.after_request
def add_cache_headers(response):
    """Add Cache-Control headers based on endpoint."""
    path = request.path
    for prefix, max_age in _CACHE_MAX_AGES.items():
        if path.startswith(prefix):
            response.headers['Cache-Control'] = f'public, max-age={max_age}'
            return response
    return response

# ============ 路由 ============

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/health')
def health():
    return jsonify({"status": "ok", "service": "nuan-nuan-memory", "version": APP_VERSION})


@app.route('/api/diag')
def diag():
    """诊断端点：检查 LLM 配置和连通性"""
    api_key = os.getenv("ZHIPU_API_KEY", "").strip()
    model = os.getenv("LOCAL_AGENT_MODEL", "").strip()
    result = {
        "zhipu_api_key_set": bool(api_key),
        "zhipu_api_key_prefix": api_key[:8] + "..." if api_key else "",
        "model": model or "glm-4-flash-250414 (default)",
    }
    # 尝试一次轻量 LLM 调用
    try:
        from app.zhipu_client import call_chat
        data = call_chat(
            [{"role": "user", "content": "hi"}],
            max_tokens=10, timeout=10, retries=0,
        )
        result["llm_test"] = "ok"
        result["llm_response"] = str(data.get("choices", [{}])[0].get("message", {}).get("content", ""))[:50]
    except Exception as e:
        result["llm_test"] = "failed"
        result["llm_error"] = str(e)[:200]
    return jsonify(result)


@app.route('/api/version')
def get_version():
    return jsonify({
        "version": APP_VERSION,
        "download_url": DOWNLOAD_URL,
        "changelog": CHANGELOG,
    })

@app.route('/api/vault/path', methods=['GET'])
def get_vault_path():
    return jsonify({"path": str(get_vault_root())})

@app.route('/api/memories', methods=['GET'])
def list_memories():
    conn = get_db_conn()
    try:
        items = app_repo.list_recent(conn, limit=30)
        return jsonify({'memories': items})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============ 共享工具函数 ============

def _make_llm_chat_fn(client_id: str):
    """创建带聊天历史的 LLM 聊天闭包（供 /api/chat 和 /api/voice_dialogue 共用）"""
    def llm_chat_fn(query, hist=None):
        history = _get_chat_history(client_id)
        _append_chat_history(client_id, "user", query)
        try:
            response = call_llm_chat(query, history)
        except Exception as e:
            print(f"[LLM_ERROR] call_llm_chat 异常: {e}", file=sys.stderr, flush=True)
            response = None
        if response:
            _append_chat_history(client_id, "assistant", response)
        else:
            print(f"[LLM_WARN] call_llm_chat 返回 None, query={query[:50]}", file=sys.stderr, flush=True)
        return response
    return llm_chat_fn


# ============ 核心路由（使用共享模块） ============

@app.route('/api/chat', methods=['POST'])
@rate_limit
@performance_monitor
def chat():
    data = _safe_get_json()
    user_text = data.get('text', '').strip()
    if not user_text:
        return jsonify({'error': '请输入内容'}), 400
    if len(user_text) > 4000:
        return jsonify({'error': '输入内容过长'}), 400

    conn = get_db_conn()
    vault_root = get_vault_root()
    client_id = _get_client_id()

    llm_chat_fn = _make_llm_chat_fn(client_id)
    result = handle_intent(conn, vault_root, user_text, call_llm_chat_fn=llm_chat_fn)

    if result.get('error'):
        return jsonify({'error': result['text']}), 500

    resp = {'type': 'assistant', 'text': result['text']}
    if result.get('results'):
        resp['results'] = result['results']
    if result.get('pending_save'):
        resp['pending_save'] = result['pending_save']
    if result.get('saved'):
        resp['saved'] = True
    return jsonify(resp)

@app.route('/api/chat/confirm_save', methods=['POST'])
@rate_limit
def confirm_save():
    data = _safe_get_json()
    text = data.get('text', '').strip()
    to_save = data.get('pending_text', '').strip()
    if not to_save:
        return jsonify({'error': '没有待保存的内容'}), 400
    if len(text) > 4000 or len(to_save) > 4000:
        return jsonify({'error': '输入内容过长'}), 400

    conn = get_db_conn()
    vault_root = get_vault_root()

    reply, final_text = confirm_save_action(text, to_save)
    if final_text:
        title, summary = build_memory_metadata_fast(final_text)
        mid = app_repo.remember_text_smart(conn, text=final_text, vault_root=vault_root, title=title, summary=summary)
        if mid:
            _invalidate_caches()
        else:
            return jsonify({'type': 'assistant', 'text': reply, 'warning': '保存可能未成功'})
    return jsonify({'type': 'assistant', 'text': reply})

@app.route('/api/search', methods=['POST'])
@rate_limit
@performance_monitor
def search_memories():
    data = _safe_get_json()
    query = data.get('query', '').strip()
    if not query:
        return jsonify({'error': '请输入搜索内容'}), 400
    conn = get_db_conn()
    try:
        results = app_search.search(conn, query=query, sort_mode="relevant", limit=20)
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/save', methods=['POST'])
@rate_limit
def save_memory():
    data = _safe_get_json()
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': '请输入要保存的内容'}), 400
    conn = get_db_conn()
    vault_root = get_vault_root()
    try:
        title, summary = build_memory_metadata_fast(text)
        app_repo.remember_text_smart(conn, text=text, vault_root=vault_root, title=title, summary=summary)
        _invalidate_caches()
        return jsonify({'success': True, 'message': '已保存'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
@rate_limit
def upload_file():
    files = request.files.getlist('files')
    caption = request.form.get('caption', '').strip()
    if not files:
        return jsonify({'error': '请上传文件'}), 400
    if len(files) > 10:
        return jsonify({'error': '单次最多上传10个文件'}), 400
    total_size = sum(f.content_length or 0 for f in files)
    if total_size > 50 * 1024 * 1024:
        return jsonify({'error': '总大小不能超过50MB'}), 400

    conn = get_db_conn()
    vault_root = get_vault_root()
    paths = []
    for f in files:
        safe_name = f.filename.replace('\\', '/').replace('..', '').strip()
        safe_name = re.sub(r'[^\w\.\-一-鿿㐀-䶿]', '_', safe_name)
        if not safe_name:
            safe_name = 'unnamed_file'
        ext = Path(safe_name).suffix.lower()
        if ext not in _ALLOWED_UPLOAD_EXTENSIONS:
            continue
        temp_path = vault_root / "temp" / safe_name
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(f, 'content_length') and f.content_length > 10 * 1024 * 1024:
            continue
        temp_path.write_bytes(f.read())
        paths.append(str(temp_path))

    try:
        ids = app_repo.remember_attachments(conn, paths=paths, caption=caption, vault_root=vault_root)
        if ids:
            def background_understand():
                conn2 = None
                try:
                    conn2 = app_db.connect(DATA_DIR / "agent.db")
                    success_count = 0
                    for mid in ids:
                        m = app_repo.get_memory(conn2, int(mid))
                        if not m:
                            continue
                        fp = str(m.get("file_path", "") or "")
                        extra = m.get("extra") or {}
                        if extra.get("media_type") == "image" and fp:
                            try:
                                u = understand_image(fp)
                                body_parts = []
                                if u.tags:
                                    body_parts.append("标签：" + "、".join(str(t) for t in u.tags if str(t).strip()))
                                if u.text_in_image:
                                    body_parts.append("图片文字：" + u.text_in_image)
                                new_body = "\n".join(body_parts).strip()
                                app_repo.update_memory(
                                    conn2, memory_id=mid, summary=u.caption or None,
                                    body=new_body or None,
                                    extra_patch={"vision": {"caption": u.caption, "tags": u.tags, "text_in_image": u.text_in_image}},
                                    commit=False,
                                )
                                success_count += 1
                            except Exception as e:
                                print(f"[图片理解失败] ID={mid}, error={e}", file=sys.stderr, flush=True)
                    conn2.commit()
                    print(f"[图片理解完成] 成功={success_count}", file=sys.stderr, flush=True)
                except Exception as e:
                    print(f"[图片理解线程异常] {e}", file=sys.stderr, flush=True)
                finally:
                    if conn2:
                        try:
                            conn2.close()
                        except Exception:
                            pass
            threading.Thread(target=background_understand, daemon=True).start()
            _invalidate_caches()
            return jsonify({'success': True, 'message': f'已保存 {len(ids)} 个文件，图片正在后台理解中。'})
        else:
            return jsonify({'error': '无法保存文件'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============ 语音 ============

_baidu_client = None
_baidu_client_lock = threading.Lock()

def _get_baidu_client():
    global _baidu_client
    if _baidu_client is not None:
        return _baidu_client
    with _baidu_client_lock:
        if _baidu_client is not None:
            return _baidu_client
        from aip import AipSpeech
        app_id = os.getenv("BAIDU_APP_ID", "").strip()
        api_key = os.getenv("BAIDU_API_KEY", "").strip()
        secret_key = os.getenv("BAIDU_SECRET_KEY", "").strip()
        if not all([app_id, api_key, secret_key]):
            missing = [k for k, v in {"BAIDU_APP_ID": app_id, "BAIDU_API_KEY": api_key, "BAIDU_SECRET_KEY": secret_key}.items() if not v]
            raise RuntimeError(f"百度语音API配置不完整，缺失: {', '.join(missing)}")
        _baidu_client = AipSpeech(app_id, api_key, secret_key)
        return _baidu_client

def synthesize_with_qwen(text: str) -> bytes:
    client = _get_baidu_client()
    text = text.strip()[:300]
    result = client.synthesis(text, 'zh', 1, {'per': 5, 'spd': 5, 'pit': 5, 'vol': 7, 'aue': 6})
    if isinstance(result, dict):
        raise RuntimeError(f"百度语音合成失败: {result.get('err_msg', '未知错误')}")
    if not result or len(result) < 100:
        raise RuntimeError("百度语音合成返回空音频")
    return result

def recognize_with_baidu(wav_bytes: bytes) -> str:
    client = _get_baidu_client()
    result = client.asr(wav_bytes, 'wav', 16000, {'dev_pid': 1537})
    if 'result' in result and result['result']:
        return result['result'][0].strip()
    return ""

@app.route('/api/speech_recognize', methods=['POST'])
@rate_limit
def speech_recognize():
    if 'audio' not in request.files:
        return jsonify({'error': '请上传音频文件'}), 400
    audio_file = request.files['audio']
    try:
        audio_data = audio_file.read()
        if len(audio_data) < 1000:
            return jsonify({'error': '音频数据太短'}), 400
        text = recognize_with_baidu(audio_data)
        if text:
            return jsonify({'text': text, 'success': True})
        else:
            return jsonify({'error': '未能识别到文字，请重试'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/speech_synthesize', methods=['POST'])
@rate_limit
def speech_synthesize():
    data = _safe_get_json()
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': '请输入要合成的内容'}), 400
    try:
        wav_bytes = synthesize_with_qwen(text[:300])
        audio_b64 = base64.b64encode(wav_bytes).decode('utf-8')
        return jsonify({'audio': audio_b64})
    except Exception as e:
        return jsonify({'error': f'语音合成失败: {e}'}), 500

@app.route('/api/voice_dialogue', methods=['POST'])
@rate_limit
def voice_dialogue():
    """语音对话API。传入 skip_tts=true 可跳过语音合成，由客户端单独请求TTS。"""
    if 'audio' not in request.files:
        return jsonify({'error': '请上传音频文件'}), 400

    skip_tts = request.form.get('skip_tts', '').lower() in ('true', '1', 'yes')

    audio_file = request.files['audio']
    try:
        audio_data = audio_file.read()
        if len(audio_data) < 1000:
            return jsonify({'error': '音频数据太短'}), 400

        user_text = recognize_with_baidu(audio_data)
        if not user_text:
            return jsonify({
                'user_text': '',
                'text': '没听清呢，再说一次呗~',
                'audio': None,
                'retry': True,
            })

        conn = get_db_conn()
        vault_root = get_vault_root()
        client_id = _get_client_id()

        # 前端有待保存内容
        pending_save_text = request.form.get('pending_save_text', '').strip()
        if pending_save_text:
            reply, final_text = confirm_save_action(user_text, pending_save_text)
            if final_text:
                title, summary = build_memory_metadata_fast(final_text)
                app_repo.remember_text_smart(conn, text=final_text, vault_root=vault_root, title=title, summary=summary)
            resp = {'user_text': user_text, 'type': 'assistant', 'text': reply, 'saved': True}
            if not skip_tts:
                try:
                    audio_bytes = synthesize_with_qwen(reply[:150])
                    resp['audio'] = base64.b64encode(audio_bytes).decode('utf-8')
                except Exception:
                    resp['audio'] = None
            return jsonify(resp)

        llm_chat_fn = _make_llm_chat_fn(client_id)
        result = handle_intent(conn, vault_root, user_text, call_llm_chat_fn=llm_chat_fn)

        if result.get('error'):
            return jsonify({'error': result['text']}), 500

        resp = {
            'user_text': user_text,
            'type': 'assistant',
            'text': result['text'],
        }
        if result.get('results'):
            resp['results'] = result['results']
        if result.get('pending_save'):
            resp['pending_save'] = result['pending_save']
        if result.get('saved'):
            resp['saved'] = True

        if not skip_tts:
            try:
                tts_text = result['text'][:300]
                if result.get('pending_save'):
                    tts_text = result['text'][:150]
                audio_bytes = synthesize_with_qwen(tts_text)
                resp['audio'] = base64.b64encode(audio_bytes).decode('utf-8')
            except Exception as e:
                print(f"[voice_dialogue] TTS 失败: {e}", file=sys.stderr, flush=True)
                resp['audio'] = None

        return jsonify(resp)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear', methods=['POST'])
def clear_history():
    client_id = _get_client_id()
    _clear_chat_history(client_id)
    return jsonify({'success': True})

# ============ 图片 ============

IMAGE_MIME_TYPES = {
    '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
    '.gif': 'image/gif', '.webp': 'image/webp', '.bmp': 'image/bmp', '.svg': 'image/svg+xml'
}

def is_path_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent)
        return True
    except ValueError:
        return False

def _is_safe_path(file_path):
    try:
        p = Path(file_path).resolve()
        ext = p.suffix.lower()
        if ext not in IMAGE_MIME_TYPES:
            return False, f"不允许的文件类型: {ext}"
        allowed_roots = [VAULT_DIR.resolve(), DATA_DIR.resolve()]
        if not any(is_path_under(p, root) for root in allowed_roots):
            return False, "路径不在允许的目录内"
        if p.exists() and p.stat().st_size > 50 * 1024 * 1024:
            return False, "文件过大"
        return True, None
    except Exception as e:
        return False, str(e)

@app.route('/api/file/image')
def get_image():
    file_path = request.args.get('path', '').strip()
    if not file_path or file_path == 'undefined':
        return Response('', status=404)
    is_safe, error_msg = _is_safe_path(file_path)
    if not is_safe:
        return Response(f'访问被拒绝: {error_msg}', status=403)
    try:
        p = Path(file_path)
        if not p.exists():
            return Response('文件不存在', status=404)
        ext = p.suffix.lower()
        mime_type = IMAGE_MIME_TYPES.get(ext, 'image/jpeg')
        response = send_from_directory(str(p.parent), p.name, mimetype=mime_type)
        response.headers['Cache-Control'] = 'private, max-age=3600'
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
    except Exception:
        return Response('加载失败', status=500)

# ============ 数据管理路由 ============

@app.route('/api/memories/recent', methods=['GET'])
@rate_limit
@performance_monitor
@cached_response(lambda: f"recent_memories:{request.args.get('limit', '10')}", ttl=60)
def recent_memories():
    conn = get_db_conn()
    try:
        limit = int(request.args.get('limit', 10))
        if limit > 50:
            limit = 50
        items = app_repo.list_recent(conn, limit=limit)
        return jsonify({'memories': items})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/memories/search', methods=['POST'])
@rate_limit
@performance_monitor
def search_memories_advanced():
    try:
        data = _safe_get_json()
        query = data.get('query', '').strip()
        tags = data.get('tags', [])
        date_from = data.get('date_from')
        date_to = data.get('date_to')
        try:
            limit = max(1, min(100, int(data.get('limit', 20))))
        except (ValueError, TypeError):
            limit = 20

        if not query and not tags:
            return jsonify({'error': '请提供搜索关键词或标签'}), 400

        conn = get_db_conn()
        search_conditions = {
            'query': query, 'tags': tags,
            'date_from': date_from, 'date_to': date_to, 'limit': limit,
        }
        results = app_repo.search_advanced(conn, search_conditions)
        return jsonify({'results': results, 'total': len(results)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tags', methods=['GET'])
@rate_limit
@performance_monitor
@cached_response(lambda: "all_tags", ttl=300)
def get_all_tags():
    try:
        conn = get_db_conn()
        tags = []
        try:
            tags = app_repo.get_all_tags(conn)
        except AttributeError:
            recent = app_repo.list_recent(conn, limit=100)
            tag_set = set()
            for item in recent:
                if 'tags' in item and item['tags']:
                    tag_set.update(item['tags'])
            tags = list(tag_set)
        return jsonify({'tags': tags})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
@rate_limit
@performance_monitor
@cached_response(lambda: "stats", ttl=60)
def get_stats():
    try:
        conn = get_db_conn()
        total_memories = app_repo.get_total_memories(conn)
        recent = app_repo.list_recent(conn, limit=10)
        # 统计文件类记忆数量
        row = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE file_path <> ''"
        ).fetchone()
        total_files = row[0] if row else 0

        # 标签统计（单次查询，同时获取标签列表和计数）
        tag_counts = app_repo.get_tag_counts(conn)
        all_tags = [t for t, _ in tag_counts]
        top_tags = tag_counts[:10]

        # 存储信息（带缓存，避免每次遍历目录）
        now_ts = time.time()
        if now_ts - _storage_size_cache["time"] > _STORAGE_CACHE_TTL:
            vault_root = get_vault_root()
            storage_size = 0
            if vault_root.exists():
                for dirpath, _dirnames, filenames in os.walk(str(vault_root)):
                    for fn in filenames:
                        try:
                            storage_size += os.path.getsize(os.path.join(dirpath, fn))
                        except OSError:
                            pass
            _storage_size_cache["size"] = storage_size
            _storage_size_cache["time"] = now_ts
        storage_size = _storage_size_cache["size"]

        stats = {
            'total_memories': total_memories,
            'recent_count': len(recent),
            'total_files': total_files,
            'tags': all_tags[:20],
            'top_tags': top_tags,
            'last_updated': recent[0].get('time') if recent else None,
            'storage_size': storage_size,
            'storage_formatted': f"{storage_size / 1024:.1f} KB" if storage_size < 1024 * 1024 else f"{storage_size / 1024 / 1024:.1f} MB",
        }
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/memory/<int:memory_id>', methods=['GET'])
@rate_limit
@performance_monitor
def get_memory(memory_id):
    conn = get_db_conn()
    try:
        m = app_repo.get_memory(conn, memory_id)
        if not m:
            return jsonify({'error': '记忆不存在'}), 404
        return jsonify({'memory': m})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/memory/<int:memory_id>', methods=['PUT'])
@rate_limit
@performance_monitor
def update_memory(memory_id):
    try:
        data = _safe_get_json()
        content = data.get('content', '').strip()
        if not content:
            return jsonify({'error': '内容不能为空'}), 400
        conn = get_db_conn()
        app_repo.update_memory(conn, memory_id=memory_id, body=content)
        _invalidate_caches()
        return jsonify({'success': True, 'message': '记忆更新成功'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/memory/<int:memory_id>', methods=['DELETE'])
@rate_limit
@performance_monitor
def delete_memory(memory_id):
    try:
        conn = get_db_conn()
        m = app_repo.get_memory(conn, memory_id)
        if not m:
            return jsonify({'error': '记忆不存在'}), 404
        fp = str(m.get("file_path", "") or "").strip()
        if fp:
            p = Path(fp)
            if p.exists() and p.is_file() and is_path_under(p.resolve(), VAULT_DIR.resolve()):
                try:
                    p.unlink()
                except OSError as e:
                    print(f"[DELETE] 删除文件失败 {fp}: {e}", file=sys.stderr, flush=True)
        app_repo.delete_memory(conn, memory_id)
        _invalidate_caches()
        return jsonify({'success': True, 'message': '记忆删除成功'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============ 管理员页面 & 接口 ============

def _check_admin_key():
    """校验管理员密钥（通过 ADMIN_KEY 环境变量设置）"""
    admin_key = os.environ.get("ADMIN_KEY", "").strip()
    if not admin_key:
        return None
    provided = request.headers.get("X-Admin-Key", "") or request.args.get("key", "")
    return provided == admin_key


@app.route('/admin')
def admin_page():
    """管理员页面（SPA，登录逻辑由前端处理）"""
    return send_from_directory('.', 'admin.html')


@app.route('/api/admin/verify')
def admin_verify():
    """验证管理员密码"""
    admin_key = os.environ.get("ADMIN_KEY", "").strip()
    if not admin_key:
        return jsonify({"ok": False, "error": "管理员密码未配置"}), 500
    provided = request.args.get("key", "")
    if provided == admin_key:
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "密码错误"}), 401


@app.route('/api/admin/stats')
def admin_stats():
    """返回统计信息（不含记忆列表），用于仪表盘快速加载"""
    if not _check_admin_key():
        return jsonify({'error': '未授权'}), 403
    conn = get_db_conn()
    try:
        total = app_repo.get_total_memories(conn)
        tag_counts = app_repo.get_tag_counts(conn)
        total_tags = len(tag_counts)
        top_tags = tag_counts[:10]
        row_file = conn.execute("SELECT COUNT(*) FROM memories WHERE file_path <> ''").fetchone()
        total_files = row_file[0] if row_file else 0
        row_last = conn.execute("SELECT updated_at FROM memories ORDER BY updated_at DESC LIMIT 1").fetchone()
        last_updated = row_last[0] if row_last else ""
        # 最近 7 天每日新增数
        rows_daily = conn.execute(
            "SELECT DATE(created_at) as d, COUNT(*) as c FROM memories "
            "WHERE created_at >= DATE('now', '-7 days') GROUP BY d ORDER BY d"
        ).fetchall()
        daily = [{"date": r["d"], "count": r["c"]} for r in rows_daily]
        # 存储大小
        vault_root = get_vault_root()
        storage_size = 0
        if vault_root.exists():
            for dirpath, _dirnames, filenames in os.walk(str(vault_root)):
                for fn in filenames:
                    try:
                        storage_size += os.path.getsize(os.path.join(dirpath, fn))
                    except OSError:
                        pass
        storage_formatted = f"{storage_size / 1024:.1f} KB" if storage_size < 1024 * 1024 else f"{storage_size / 1024 / 1024:.1f} MB"
        # 今日新增
        row_today = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE DATE(created_at) = DATE('now')"
        ).fetchone()
        today_count = row_today[0] if row_today else 0
        return jsonify({
            "total": total,
            "total_files": total_files,
            "total_text": total - total_files,
            "total_tags": total_tags,
            "today_count": today_count,
            "top_tags": top_tags,
            "last_updated": last_updated,
            "daily": daily,
            "storage_size": storage_size,
            "storage_formatted": storage_formatted,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin/memories')
def admin_memories():
    """分页返回记忆列表（需管理员密码）"""
    if not _check_admin_key():
        return jsonify({'error': '未授权'}), 403
    page = max(1, int(request.args.get("page", 1)))
    page_size = min(100, max(1, int(request.args.get("page_size", 20))))
    q = request.args.get("q", "").strip()
    tag = request.args.get("tag", "").strip()
    offset = (page - 1) * page_size
    conn = get_db_conn()
    try:
        where, params = [], []
        if q:
            where.append("(title LIKE ? OR summary LIKE ? OR body LIKE ? OR CAST(id AS TEXT) LIKE ?)")
            like = f"%{q}%"
            params.extend([like, like, like, like])
        if tag:
            where.append("extra_json LIKE ?")
            params.append(f'%"{tag}"%')
        where_sql = (" WHERE " + " AND ".join(where)) if where else ""

        total_row = conn.execute(f"SELECT COUNT(*) FROM memories{where_sql}", params).fetchone()
        total = total_row[0] if total_row else 0

        rows = conn.execute(
            f"SELECT id, title, summary, body, file_path, extra_json, created_at, updated_at "
            f"FROM memories{where_sql} ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            params + [page_size, offset]
        ).fetchall()

        memories = []
        for r in rows:
            tags = []
            try:
                extra = json.loads(r["extra_json"] or "{}")
                tags = extra.get("tags", [])
            except Exception:
                pass
            memories.append({
                "id": r["id"],
                "title": r["title"] or "",
                "summary": r["summary"] or "",
                "body": (r["body"] or "")[:500],
                "tags": tags,
                "file_path": r["file_path"] or "",
                "created_at": r["created_at"] or "",
                "updated_at": r["updated_at"] or "",
            })

        return jsonify({
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size,
            "memories": memories,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin/data')
def admin_data():
    """返回所有记忆数据 + 统计信息（需管理员密码）- 兼容旧接口"""
    if not _check_admin_key():
        return jsonify({'error': '未授权'}), 403
    conn = get_db_conn()
    try:
        total = app_repo.get_total_memories(conn)
        tag_counts = app_repo.get_tag_counts(conn)
        total_tags = len(tag_counts)
        top_tags = tag_counts[:10]
        rows = conn.execute(
            "SELECT id, title, summary, body, file_path, extra_json, created_at, updated_at FROM memories ORDER BY updated_at DESC"
        ).fetchall()
        row_file = conn.execute("SELECT COUNT(*) FROM memories WHERE file_path <> ''").fetchone()
        total_files = row_file[0] if row_file else 0

        memories = []
        for r in rows:
            tags = []
            try:
                extra = json.loads(r["extra_json"] or "{}")
                tags = extra.get("tags", [])
            except Exception:
                pass
            memories.append({
                "id": r["id"],
                "title": r["title"] or "",
                "summary": r["summary"] or "",
                "body": (r["body"] or "")[:500],
                "tags": tags,
                "file_path": r["file_path"] or "",
                "created_at": r["created_at"] or "",
                "updated_at": r["updated_at"] or "",
            })

        return jsonify({
            "total": total,
            "total_files": total_files,
            "total_tags": total_tags,
            "top_tags": top_tags,
            "memories": memories,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin/memory/<int:memory_id>', methods=['DELETE'])
def admin_delete_memory(memory_id):
    """删除记忆及其关联文件（需管理员密码）"""
    if not _check_admin_key():
        return jsonify({'error': '未授权'}), 403
    conn = get_db_conn()
    try:
        m = app_repo.get_memory(conn, memory_id)
        if not m:
            return jsonify({'error': '记忆不存在'}), 404
        # 删除关联的物理文件（校验路径在允许范围内）
        fp = str(m.get("file_path", "") or "").strip()
        if fp:
            p = Path(fp).resolve()
            if p.exists() and p.is_file() and is_path_under(p, VAULT_DIR.resolve()):
                try:
                    p.unlink()
                except OSError as e:
                    print(f"[ADMIN] 删除文件失败 {fp}: {e}", file=sys.stderr, flush=True)
        app_repo.delete_memory(conn, memory_id)
        _invalidate_caches()
        return jsonify({'success': True, 'message': '记忆及文件已删除'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin/memory/<int:memory_id>', methods=['PUT'])
def admin_update_memory(memory_id):
    """编辑记忆（需管理员密码）"""
    if not _check_admin_key():
        return jsonify({'error': '未授权'}), 403
    data = _safe_get_json()
    conn = get_db_conn()
    try:
        m = app_repo.get_memory(conn, memory_id)
        if not m:
            return jsonify({'error': '记忆不存在'}), 404
        title = data.get('title')
        summary = data.get('summary')
        body = data.get('body')
        tags = data.get('tags')
        kwargs = {"memory_id": memory_id}
        if title is not None:
            kwargs["title"] = str(title).strip()
        if summary is not None:
            kwargs["summary"] = str(summary).strip()
        if body is not None:
            kwargs["body"] = str(body).strip()
        if tags is not None:
            if isinstance(tags, list):
                kwargs["extra_patch"] = {"tags": [str(t).strip() for t in tags if str(t).strip()]}
            else:
                return jsonify({'error': '标签必须是数组'}), 400
        if len(kwargs) <= 1:
            return jsonify({'error': '没有要更新的内容'}), 400
        app_repo.update_memory(conn, **kwargs)
        _invalidate_caches()
        return jsonify({'success': True, 'message': '记忆已更新'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin/backup', methods=['GET'])
def admin_backup():
    """下载 SQLite 数据库文件（需 ADMIN_KEY）"""
    if not _check_admin_key():
        return jsonify({'error': '未授权'}), 403
    db_path = DATA_DIR / "agent.db"
    if not db_path.exists():
        return jsonify({'error': '数据库文件不存在'}), 404
    return send_from_directory(
        str(DATA_DIR), "agent.db",
        mimetype="application/octet-stream",
        as_attachment=True,
        download_name=f"agent_backup_{time.strftime('%Y%m%d_%H%M%S')}.db",
    )


@app.route('/api/admin/export', methods=['GET'])
def admin_export():
    """导出所有记忆为 JSON（需 ADMIN_KEY）"""
    if not _check_admin_key():
        return jsonify({'error': '未授权'}), 403
    conn = get_db_conn()
    try:
        rows = conn.execute(
            "SELECT id, title, summary, body, file_path, extra_json, created_at, updated_at FROM memories ORDER BY id"
        ).fetchall()
        memories = []
        for r in rows:
            memories.append({
                "id": r["id"],
                "title": r["title"],
                "summary": r["summary"],
                "body": r["body"],
                "file_path": r["file_path"],
                "extra_json": r["extra_json"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
            })
        export_data = {
            "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total": len(memories),
            "memories": memories,
        }
        resp = Response(
            json.dumps(export_data, ensure_ascii=False, indent=2),
            mimetype="application/json",
        )
        resp.headers["Content-Disposition"] = f'attachment; filename=memories_export_{time.strftime("%Y%m%d_%H%M%S")}.json'
        return resp
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============ 管理员配置接口 ============

_CONFIG_FILE = DATA_DIR / "admin_config.json"

_DEFAULT_PROVIDER = {
    "id": "zhipu",
    "name": "智谱 AI",
    "base_url": "https://open.bigmodel.cn/api/paas/v4",
    "api_key": "",
    "chat_model": "glm-4-flash-250414",
    "vision_model": "glm-4v-flash",
    "is_active": True,
}


def _read_raw_config() -> dict:
    if not _CONFIG_FILE.exists():
        return {}
    try:
        with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _write_raw_config(cfg: dict):
    with open(_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def _activate_provider(provider: dict):
    """将指定 provider 的配置写入 os.environ 并同步到 .env 文件，使 LLM 调用立即生效"""
    os.environ["LLM_BASE_URL"] = provider.get("base_url", "")
    os.environ["LLM_API_KEY"] = provider.get("api_key", "")
    os.environ["LOCAL_AGENT_MODEL"] = provider.get("chat_model", "glm-4-flash-250414")
    os.environ["LOCAL_AGENT_VISION_MODEL"] = provider.get("vision_model", "glm-4v-flash")
    # 向后兼容
    os.environ["ZHIPU_API_KEY"] = provider.get("api_key", "")
    # 同步到 .env 文件（确保重启后仍然生效）
    _sync_env_file({
        "LOCAL_AGENT_MODEL": provider.get("chat_model", "glm-4-flash-250414"),
        "LOCAL_AGENT_VISION_MODEL": provider.get("vision_model", "glm-4v-flash"),
        "ZHIPU_API_KEY": provider.get("api_key", ""),
    })


def _sync_env_file(updates: dict):
    """将指定的键值对同步写入 .env 文件（保留原有内容和注释）"""
    env_path = project_root / ".env"
    if not env_path.exists():
        return
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
        updated_keys = set()
        new_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                key = stripped.split("=", 1)[0].strip()
                if key in updates:
                    new_lines.append(f"{key}={updates[key]}")
                    updated_keys.add(key)
                    continue
            new_lines.append(line)
        # 追加 .env 中不存在的键
        for key, val in updates.items():
            if key not in updated_keys:
                new_lines.append(f"{key}={val}")
        env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    except Exception:
        pass


def _get_providers(cfg: dict) -> list:
    providers = cfg.get("llm_providers", [])
    if not providers:
        # 首次使用：从旧配置迁移生成默认 provider
        default = dict(_DEFAULT_PROVIDER)
        default["api_key"] = cfg.get("ZHIPU_API_KEY", os.environ.get("ZHIPU_API_KEY", ""))
        default["chat_model"] = cfg.get("LOCAL_AGENT_MODEL", os.environ.get("LOCAL_AGENT_MODEL", "glm-4-flash-250414"))
        default["vision_model"] = cfg.get("LOCAL_AGENT_VISION_MODEL", os.environ.get("LOCAL_AGENT_VISION_MODEL", "glm-4v-flash"))
        providers = [default]
        cfg["llm_providers"] = providers
        cfg["active_provider_id"] = "zhipu"
        _write_raw_config(cfg)
    return providers


def _load_config():
    """从 admin_config.json 加载配置并激活当前 provider"""
    cfg = _read_raw_config()
    # 加载旧式 flat 配置到 env（向后兼容）
    for key, val in cfg.items():
        if isinstance(val, str) and val.strip():
            os.environ[key] = val
    # 加载 provider 并激活
    providers = _get_providers(cfg)
    active_id = cfg.get("active_provider_id", "zhipu")
    active = next((p for p in providers if p.get("id") == active_id), providers[0] if providers else None)
    if active:
        _activate_provider(active)


_load_config()


def _mask_key(key: str) -> str:
    if len(key) <= 8:
        return "*" * len(key) if key else ""
    return key[:4] + "*" * (len(key) - 8) + key[-4:]


# ============ 管理员 Provider 接口 ============

@app.route('/api/admin/config', methods=['GET'])
def admin_get_config():
    """获取当前模型配置（需管理员密码）"""
    if not _check_admin_key():
        return jsonify({'error': '未授权'}), 403
    cfg = _read_raw_config()
    providers = _get_providers(cfg)
    active_id = cfg.get("active_provider_id", "zhipu")
    active = next((p for p in providers if p.get("id") == active_id), providers[0] if providers else {})
    return jsonify({
        "chat_model": os.environ.get("LOCAL_AGENT_MODEL", "glm-4-flash-250414"),
        "vision_model": os.environ.get("LOCAL_AGENT_VISION_MODEL", "glm-4v-flash"),
        "api_key_masked": _mask_key(os.environ.get("LLM_API_KEY", "")),
        "has_api_key": bool(os.environ.get("LLM_API_KEY", "").strip()),
        "active_provider_id": active_id,
        "active_provider_name": active.get("name", ""),
    })


@app.route('/api/admin/config', methods=['POST'])
def admin_set_config():
    """更新当前活跃 provider 的配置（需管理员密码）"""
    if not _check_admin_key():
        return jsonify({'error': '未授权'}), 403
    data = request.get_json(silent=True) or {}
    cfg = _read_raw_config()
    providers = _get_providers(cfg)
    active_id = cfg.get("active_provider_id", "zhipu")
    active = next((p for p in providers if p.get("id") == active_id), None)
    if not active:
        return jsonify({"error": "没有活跃的模型厂商"}), 400
    if "chat_model" in data and str(data["chat_model"]).strip():
        active["chat_model"] = str(data["chat_model"]).strip()
    if "vision_model" in data and str(data["vision_model"]).strip():
        active["vision_model"] = str(data["vision_model"]).strip()
    if "api_key" in data and str(data["api_key"]).strip():
        active["api_key"] = str(data["api_key"]).strip()
    cfg["llm_providers"] = providers
    _write_raw_config(cfg)
    _activate_provider(active)
    return jsonify({"success": True, "message": "配置已保存，立即生效"})


@app.route('/api/admin/providers', methods=['GET'])
def admin_list_providers():
    """列出所有模型厂商"""
    if not _check_admin_key():
        return jsonify({'error': '未授权'}), 403
    cfg = _read_raw_config()
    providers = _get_providers(cfg)
    active_id = cfg.get("active_provider_id", "zhipu")
    result = []
    for p in providers:
        result.append({
            "id": p.get("id"),
            "name": p.get("name"),
            "base_url": p.get("base_url"),
            "api_key_masked": _mask_key(p.get("api_key", "")),
            "has_api_key": bool(p.get("api_key", "").strip()),
            "chat_model": p.get("chat_model"),
            "vision_model": p.get("vision_model"),
            "is_active": p.get("id") == active_id,
        })
    return jsonify({"providers": result, "active_provider_id": active_id})


@app.route('/api/admin/providers', methods=['POST'])
def admin_add_provider():
    """添加新的模型厂商"""
    if not _check_admin_key():
        return jsonify({'error': '未授权'}), 403
    data = request.get_json(silent=True) or {}
    name = str(data.get("name", "")).strip()
    base_url = str(data.get("base_url", "")).strip()
    if not name or not base_url:
        return jsonify({"error": "名称和 API 地址不能为空"}), 400
    cfg = _read_raw_config()
    providers = _get_providers(cfg)
    provider_id = str(data.get("id", "")).strip() or name.lower().replace(" ", "_") + "_" + uuid.uuid4().hex[:6]
    # 检查 id 唯一
    if any(p.get("id") == provider_id for p in providers):
        return jsonify({"error": f"厂商 ID '{provider_id}' 已存在"}), 400
    new_provider = {
        "id": provider_id,
        "name": name,
        "base_url": base_url,
        "api_key": str(data.get("api_key", "")).strip(),
        "chat_model": str(data.get("chat_model", "")).strip(),
        "vision_model": str(data.get("vision_model", "")).strip(),
        "is_active": False,
    }
    providers.append(new_provider)
    cfg["llm_providers"] = providers
    _write_raw_config(cfg)
    return jsonify({"success": True, "provider": {**new_provider, "api_key": "***"}}), 201


@app.route('/api/admin/providers/<provider_id>', methods=['PUT'])
def admin_update_provider(provider_id):
    """更新模型厂商配置"""
    if not _check_admin_key():
        return jsonify({'error': '未授权'}), 403
    data = request.get_json(silent=True) or {}
    cfg = _read_raw_config()
    providers = _get_providers(cfg)
    provider = next((p for p in providers if p.get("id") == provider_id), None)
    if not provider:
        return jsonify({"error": "厂商不存在"}), 404
    if "name" in data and str(data["name"]).strip():
        provider["name"] = str(data["name"]).strip()
    if "base_url" in data and str(data["base_url"]).strip():
        provider["base_url"] = str(data["base_url"]).strip()
    if "api_key" in data and str(data["api_key"]).strip():
        provider["api_key"] = str(data["api_key"]).strip()
    if "chat_model" in data and str(data["chat_model"]).strip():
        provider["chat_model"] = str(data["chat_model"]).strip()
    if "vision_model" in data:
        provider["vision_model"] = str(data["vision_model"]).strip()
    cfg["llm_providers"] = providers
    _write_raw_config(cfg)
    # 如果更新的是当前活跃 provider，立即生效
    if cfg.get("active_provider_id") == provider_id:
        _activate_provider(provider)
    return jsonify({"success": True})


@app.route('/api/admin/providers/<provider_id>', methods=['DELETE'])
def admin_delete_provider(provider_id):
    """删除模型厂商"""
    if not _check_admin_key():
        return jsonify({'error': '未授权'}), 403
    cfg = _read_raw_config()
    providers = _get_providers(cfg)
    if len(providers) <= 1:
        return jsonify({"error": "至少保留一个模型厂商"}), 400
    provider = next((p for p in providers if p.get("id") == provider_id), None)
    if not provider:
        return jsonify({"error": "厂商不存在"}), 404
    if cfg.get("active_provider_id") == provider_id:
        return jsonify({"error": "不能删除当前正在使用的厂商，请先切换"}), 400
    providers = [p for p in providers if p.get("id") != provider_id]
    cfg["llm_providers"] = providers
    _write_raw_config(cfg)
    return jsonify({"success": True})


@app.route('/api/admin/providers/<provider_id>/activate', methods=['POST'])
def admin_activate_provider(provider_id):
    """切换活跃的模型厂商"""
    if not _check_admin_key():
        return jsonify({'error': '未授权'}), 403
    cfg = _read_raw_config()
    providers = _get_providers(cfg)
    provider = next((p for p in providers if p.get("id") == provider_id), None)
    if not provider:
        return jsonify({"error": "厂商不存在"}), 404
    if not provider.get("api_key", "").strip():
        return jsonify({"error": f"「{provider.get('name', provider_id)}」未配置 API Key，请先编辑添加"}), 400
    if not provider.get("chat_model", "").strip():
        return jsonify({"error": f"「{provider.get('name', provider_id)}」未配置对话模型，请先编辑添加"}), 400
    cfg["active_provider_id"] = provider_id
    _write_raw_config(cfg)
    _activate_provider(provider)
    return jsonify({"success": True, "message": f"已切换到 {provider.get('name', provider_id)}"})

# ============ 管理员工具接口 ============

@app.route('/api/admin/test_llm', methods=['POST'])
def admin_test_llm():
    """LLM 连通性测试（需管理员密码）"""
    if not _check_admin_key():
        return jsonify({'error': '未授权'}), 403
    model = os.environ.get("LOCAL_AGENT_MODEL", "")
    base_url = os.environ.get("LLM_BASE_URL", "")
    api_key = os.environ.get("LLM_API_KEY", "")
    result = {
        "model": model or "glm-4-flash-250414 (default)",
        "base_url": base_url or "(默认)",
        "has_api_key": bool(api_key),
    }
    try:
        from app.zhipu_client import call_chat
        data = call_chat(
            [{"role": "user", "content": "hi"}],
            max_tokens=10, timeout=15, retries=0,
        )
        content = str(data.get("choices", [{}])[0].get("message", {}).get("content", ""))[:100]
        result["ok"] = True
        result["response"] = content
    except Exception as e:
        result["ok"] = False
        result["error"] = str(e)[:300]
    return jsonify(result)


@app.route('/api/admin/clear_cache', methods=['POST'])
def admin_clear_cache():
    """清除服务端缓存（需管理员密码）"""
    if not _check_admin_key():
        return jsonify({'error': '未授权'}), 403
    _invalidate_caches()
    _storage_size_cache["size"] = 0
    _storage_size_cache["time"] = 0
    return jsonify({"success": True, "message": "缓存已清除"})


@app.route('/api/admin/prune', methods=['POST'])
def admin_prune():
    """清理失效记忆（文件不存在的记录）（需管理员密码）"""
    if not _check_admin_key():
        return jsonify({'error': '未授权'}), 403
    conn = get_db_conn()
    try:
        removed = app_repo.prune_missing_file_memories(conn)
        if removed:
            _invalidate_caches()
        return jsonify({"success": True, "removed": removed, "message": f"已清理 {removed} 条失效记忆" if removed else "没有失效记忆"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============ 启动 ============

# 启动时初始化 vault（仅一次）
init_vault_once()

if __name__ == '__main__':
    is_debug = os.environ.get('FLASK_DEBUG', '').lower() in ('true', '1', 'yes')
    print(f"暖暖 - 记忆助手 Web 版 | 端口: {PORT} | 数据: {DATA_DIR}", file=sys.stderr, flush=True)
    print(f"管理后台: http://127.0.0.1:{PORT}/admin", file=sys.stderr, flush=True)
    app.run(host='0.0.0.0', port=PORT, debug=is_debug)
