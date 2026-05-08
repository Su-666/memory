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
    from app.answer import answer as app_answer
    from app.intent import call_planning_model
    from app.vault import ensure_vault_root
    from app.vision import understand_image
    from app.intent_chat import (
        build_memory_metadata,
        handle_intent,
        confirm_save_action,
    )
    from app.llm import call_llm_chat
except ModuleNotFoundError as e:
    print(f"[ERROR] Failed to import app module: {e}", file=sys.stderr, flush=True)
    raise

# Flask 应用
app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app, resources={r"/api/*": {"origins": "*"}})

# 加载环境变量
def load_env():
    dotenv_path = project_root / ".env"
    if dotenv_path.exists():
        for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line and "=" in line and not line.startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())

load_env()

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
    except Exception:
        pass
    # 仅在本地环境清理失效文件记忆（Railway 上数据目录是持久化的，不需要清理）
    if not os.environ.get("RAILWAY_STATIC_URL"):
        try:
            removed = app_repo.prune_missing_file_memories(conn)
            if removed:
                print(f"[STARTUP] 清理 {removed} 条失效记忆", file=sys.stderr, flush=True)
        except Exception:
            pass

# ============ 优化：per-client 聊天历史 ============
_client_chat_histories: dict[str, list[dict]] = {}
_chat_history_lock = threading.Lock()
_MAX_HISTORY_PER_CLIENT = 50

def _get_client_id() -> str:
    """获取客户端标识 - 优先使用 X-Client-Id 请求头，回退到 IP"""
    client_id = request.headers.get("X-Client-Id", "").strip()
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
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key in self.cache:
                data, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl:
                    return data
                else:
                    del self.cache[key]
            return None

    def set(self, key, value):
        with self._lock:
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
                del self.cache[oldest_key]
            self.cache[key] = (value, time.time())

    def invalidate(self, prefix):
        with self._lock:
            keys_to_remove = [k for k in self.cache if k.startswith(prefix)]
            for k in keys_to_remove:
                del self.cache[k]

response_cache = SimpleCache()

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

# ============ 路由 ============

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/health')
def health():
    return jsonify({"status": "ok", "service": "nuan-nuan-memory", "version": APP_VERSION})


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


# ============ 核心路由（使用共享模块） ============

@app.route('/api/chat', methods=['POST'])
@rate_limit
@performance_monitor
def chat():
    data = _safe_get_json()
    user_text = data.get('text', '').strip()
    if not user_text:
        return jsonify({'error': '请输入内容'}), 400

    conn = get_db_conn()
    vault_root = get_vault_root()
    client_id = _get_client_id()

    # 构建带历史的 LLM 聊天函数
    def llm_chat_fn(query, hist=None):
        history = _get_chat_history(client_id)
        _append_chat_history(client_id, "user", query)
        response = call_llm_chat(query, history)
        if response:
            _append_chat_history(client_id, "assistant", response)
        return response

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

    conn = get_db_conn()
    vault_root = get_vault_root()

    reply, final_text = confirm_save_action(text, to_save)
    if final_text:
        title, summary = build_memory_metadata(final_text)
        app_repo.remember_text_smart(conn, text=final_text, vault_root=vault_root, title=title, summary=summary)
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
        title, summary = build_memory_metadata(text)
        app_repo.remember_text_smart(conn, text=text, vault_root=vault_root, title=title, summary=summary)
        response_cache.invalidate("recent_memories")
        response_cache.invalidate("stats")
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
            threading.Thread(target=background_understand, daemon=True).start()
            response_cache.invalidate("recent_memories")
            response_cache.invalidate("stats")
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
    """语音对话API"""
    if 'audio' not in request.files:
        return jsonify({'error': '请上传音频文件'}), 400

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
                title, summary = build_memory_metadata(final_text)
                app_repo.remember_text_smart(conn, text=final_text, vault_root=vault_root, title=title, summary=summary)
            resp = {'user_text': user_text, 'type': 'assistant', 'text': reply, 'saved': True}
            try:
                audio_bytes = synthesize_with_qwen(reply[:150])
                resp['audio'] = base64.b64encode(audio_bytes).decode('utf-8')
            except Exception:
                resp['audio'] = None
            return jsonify(resp)

        # 构建带历史的 LLM 聊天函数
        def llm_chat_fn(query, hist=None):
            history = _get_chat_history(client_id)
            _append_chat_history(client_id, "user", query)
            response = call_llm_chat(query, history)
            if response:
                _append_chat_history(client_id, "assistant", response)
            return response

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
        if '..' in file_path or '~' in file_path:
            return False, "检测到路径遍历"
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
        with open(p, 'rb') as f:
            data = f.read()
        response = Response(data, mimetype=mime_type)
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
        all_tags = app_repo.get_all_tags(conn)

        # 统计文件类记忆数量
        row = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE file_path <> ''"
        ).fetchone()
        total_files = row[0] if row else 0

        # 热门标签（从 extra_json 提取带计数）
        tag_rows = conn.execute(
            "SELECT extra_json FROM memories WHERE extra_json LIKE '%tags%'"
        ).fetchall()
        tag_counter: dict[str, int] = {}
        for tr in tag_rows:
            try:
                extra = json.loads(tr[0])
                for t in extra.get("tags", []):
                    tag_counter[t] = tag_counter.get(t, 0) + 1
            except Exception:
                pass
        top_tags = sorted(tag_counter.items(), key=lambda x: -x[1])[:10]

        # 存储信息
        vault_root = get_vault_root()
        storage_size = 0
        if vault_root.exists():
            for f in vault_root.rglob("*"):
                if f.is_file():
                    storage_size += f.stat().st_size

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
        response_cache.invalidate("recent_memories")
        response_cache.invalidate("stats")
        return jsonify({'success': True, 'message': '记忆更新成功'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/memory/<int:memory_id>', methods=['DELETE'])
@rate_limit
@performance_monitor
def delete_memory(memory_id):
    try:
        conn = get_db_conn()
        app_repo.delete_memory(conn, memory_id)
        response_cache.invalidate("recent_memories")
        response_cache.invalidate("stats")
        return jsonify({'success': True, 'message': '记忆删除成功'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============ 启动 ============

# 启动时初始化 vault（仅一次）
init_vault_once()

if __name__ == '__main__':
    is_debug = os.environ.get('FLASK_DEBUG', '').lower() in ('true', '1', 'yes')
    print(f"暖暖 - 记忆助手 Web 版 | 端口: {PORT} | 数据: {DATA_DIR}", file=sys.stderr, flush=True)
    app.run(host='0.0.0.0', port=PORT, debug=is_debug)
