"""
暖暖记忆助手 - Web 版
独立部署入口，不依赖桌面端代码
"""
import os
import sys
import json
import io
import wave
import base64
import threading
import re
import time
import hashlib
import random
from functools import wraps
from pathlib import Path

# 优先设置导入路径
from import_setup import setup_paths
setup_paths()

from flask import Flask, jsonify, request, send_from_directory, Response
from flask_cors import CORS

# Railway 使用环境变量 PORT
PORT = int(os.environ.get("PORT", 5000))

# 项目根目录
project_root = Path(__file__).resolve().parent

# 确保项目根目录在 Python 路径中（解决 Railway 部署时的导入问题）
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 调试：打印路径信息
print(f"[DEBUG] project_root: {project_root}", file=sys.stderr, flush=True)
print(f"[DEBUG] sys.path[0]: {sys.path[0]}", file=sys.stderr, flush=True)
print(f"[DEBUG] app package exists: {(project_root / 'app').exists()}", file=sys.stderr, flush=True)
print(f"[DEBUG] app/__init__.py exists: {(project_root / 'app' / '__init__.py').exists()}", file=sys.stderr, flush=True)

# 数据目录（Railway Volume 挂载点或本地目录）
DATA_DIR = Path(os.environ.get("DATA_DIR", str(project_root / "data")))
VAULT_DIR = Path(os.environ.get("VAULT_DIR", str(project_root / "memory_vault")))
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
except ModuleNotFoundError as e:
    print(f"[ERROR] Failed to import app module: {e}", file=sys.stderr, flush=True)
    print(f"[ERROR] Current working directory: {os.getcwd()}", file=sys.stderr, flush=True)
    print(f"[ERROR] Python path: {sys.path}", file=sys.stderr, flush=True)
    raise

# Flask 应用
app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app)

PROJECT_ROOT = project_root

# 加载环境变量
def load_env():
    dotenv_path = PROJECT_ROOT / ".env"
    if dotenv_path.exists():
        for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line and "=" in line and not line.startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())

load_env()

# 启动时打印API配置状态（调试用）
print("[STARTUP] API Configuration Status:", file=sys.stderr, flush=True)
print(f"[STARTUP] BAIDU_APP_ID: {'✓ Set' if os.getenv('BAIDU_APP_ID', '').strip() else '✗ Not Set'}", file=sys.stderr, flush=True)
print(f"[STARTUP] BAIDU_API_KEY: {'✓ Set' if os.getenv('BAIDU_API_KEY', '').strip() else '✗ Not Set'}", file=sys.stderr, flush=True)
print(f"[STARTUP] BAIDU_SECRET_KEY: {'✓ Set' if os.getenv('BAIDU_SECRET_KEY', '').strip() else '✗ Not Set'}", file=sys.stderr, flush=True)
print(f"[STARTUP] ZHIPU_API_KEY: {'✓ Set' if os.getenv('ZHIPU_API_KEY', '').strip() else '✗ Not Set'}", file=sys.stderr, flush=True)
print(f"[STARTUP] LOCAL_AGENT_MODEL: {os.getenv('LOCAL_AGENT_MODEL', 'GLM-4-Flash-250414 (default)')}", file=sys.stderr, flush=True)
print(f"[STARTUP] LOCAL_AGENT_VISION_MODEL: {os.getenv('LOCAL_AGENT_VISION_MODEL', 'glm-4v-flash (default)')}", file=sys.stderr, flush=True)

# ============ 优化功能 ============

# 简单缓存系统
class SimpleCache:
    def __init__(self, max_size=100, ttl=300):  # 5分钟TTL
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl

    def get(self, key):
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return data
            else:
                del self.cache[key]
        return None

    def set(self, key, value):
        if len(self.cache) >= self.max_size:
            # 简单清理策略：删除最旧的条目
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        self.cache[key] = (value, time.time())

# 全局缓存实例
response_cache = SimpleCache()

# 速率限制
class RateLimiter:
    def __init__(self, max_requests=10, window_seconds=60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}

    def is_allowed(self, client_id):
        now = time.time()
        if client_id not in self.requests:
            self.requests[client_id] = []

        # 清理过期请求
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if now - req_time < self.window_seconds
        ]

        if len(self.requests[client_id]) < self.max_requests:
            self.requests[client_id].append(now)
            return True
        return False

# 全局速率限制器
rate_limiter = RateLimiter(max_requests=20, window_seconds=60)  # 每分钟最多20个请求

def get_client_id():
    """获取客户端标识（基于IP）"""
    return request.remote_addr or "unknown"

def rate_limit(f):
    """速率限制装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        client_id = get_client_id()
        if not rate_limiter.is_allowed(client_id):
            return jsonify({'error': '请求过于频繁，请稍后再试'}), 429
        return f(*args, **kwargs)
    return decorated_function

def cached_response(cache_key_func=None, ttl=300):
    """响应缓存装饰器"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                # 默认使用请求路径和参数作为缓存键
                cache_key = f"{request.path}:{hash(str(dict(request.args)))}"

            cached = response_cache.get(cache_key)
            if cached:
                return cached

            response = f(*args, **kwargs)
            if response and hasattr(response, 'get_json'):
                try:
                    response_data = response.get_json()
                    response_cache.set(cache_key, response_data)
                except:
                    pass
            return response
        return decorated_function
    return decorator

# 性能监控
def performance_monitor(f):
    """性能监控装饰器"""
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
            print(f"[ERROR] {f.__name__}: {duration:.3f}s - {str(e)}", file=sys.stderr, flush=True)
            raise
    return decorated_function

# 数据库和 Vault
def get_db_conn():
    db_path = DATA_DIR / "agent.db"
    conn = app_db.connect(db_path)
    app_db.init_db(conn)
    return conn

def get_vault_root():
    ensure_vault_root(VAULT_DIR)
    return VAULT_DIR

def init_vault():
    vault_root = get_vault_root()
    ensure_vault_root(vault_root)
    conn = get_db_conn()
    try:
        app_repo.bootstrap_import_vault(conn, vault_root)
    except Exception:
        pass
    try:
        app_repo.prune_missing_file_memories(conn)
    except Exception:
        pass
    return conn

SESSION_CHAT_HISTORY = []

# ============ 路由 ============

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/health')
def health():
    return jsonify({"status": "ok", "service": "nuan-nuan-memory", "version": "5.1"})

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

# ============ 意图判断 ============

def determine_intent(text: str) -> str:
    """判断用户意图：save / search / chat
    save: 明确要保存东西（有实质内容）
    search: 明确在找/查记忆（或问记忆相关的问题）
    chat: 其他所有情况
    """
    stripped = text.strip()
    if not stripped:
        return "chat"
    if stripped.startswith(("/聊", "/chat")):
        return "chat"
    if stripped.startswith(("/搜", "/search")):
        return "search"

    # ===== 保存意图 =====
    # 有明确保存动词 + 实质内容
    save_verbs = [
        "帮我记住", "帮我记", "帮我存", "帮我保存",
        "记一下", "存一下", "记录一下", "保存一下",
        "记下来", "存下来", "写下来", "写一下",
        "写一下", "备忘", "别忘了", "提醒我", "把这个记下来",
    ]
    for verb in save_verbs:
        if stripped.startswith(verb):
            after = stripped[len(verb):].strip()
            if not after:
                return "save"  # 纯意图，等补充
            # 有数字 → 有内容
            if any(c.isdigit() for c in after):
                return "save"
            # 有"是"+内容 → 有内容
            if "是" in after:
                parts = after.split("是", 1)
                if len(parts) == 2 and parts[1].strip():
                    return "save"
            # 只有占位词（≤6字）→ 等补充
            if len(after) <= 6:
                return "save"
            # 较长文本，检查是否包含闲聊语气词（说明不是保存内容）
            chat_words = ["呗", "呀", "呢", "吧", "哈哈", "嘿嘿", "我怕", "我想", "我觉得", "不知道"]
            if any(w in after for w in chat_words):
                return "chat"
            # 大概率有实质内容
            return "save"

    # "XX是YY" 模式（直接陈述信息）
    info_patterns = [
        ("密码是", 4), ("手机号是", 4), ("电话是", 4),
        ("卡号是", 4), ("账号是", 4), ("地址是", 4),
        ("邮箱是", 4), ("生日是", 4),
    ]
    for pattern, min_len in info_patterns:
        if pattern in stripped:
            idx = stripped.index(pattern) + len(pattern)
            after = stripped[idx:].strip()
            if after and (len(after) >= min_len or any(c.isdigit() for c in after)):
                return "save"

    # ===== 搜索意图 =====
    # 明确的搜索动词
    search_verbs = ["帮我找", "帮我查", "查一下", "搜一下", "找一下", "查询"]
    for verb in search_verbs:
        if verb in stripped:
            return "search"

    # 问记忆内容的问题
    memory_queries = [
        "我记过什么", "我存过什么", "我保存过什么", "我记录过什么",
        "我记了什么", "我存了什么", "有什么记忆", "记了哪些",
        "有哪些记忆", "记忆库里有什么",
    ]
    for phrase in memory_queries:
        if phrase in stripped:
            return "search"

    # 问具体记忆内容（疑问句 + 记忆关键词）
    memory_words = ["密码", "手机号", "电话", "地址", "生日", "邮箱", "卡号", "账号", "身份证"]
    has_question = "?" in stripped or "？" in stripped or "多少" in stripped or "什么" in stripped
    if has_question:
        for word in memory_words:
            if word in stripped:
                return "search"
        # "我记过/存过"类问题
        if any(k in stripped for k in ["记过", "存过", "保存过", "记录过"]):
            return "search"

    return "chat"


def search_memory(conn, user_text: str, limit: int = 5) -> list:
    """统一搜索记忆，智能扩展搜索词
    返回按时间倒序排列的结果（最新的优先）
    """
    query = extract_search_query(user_text)

    search_queries = [query]
    # 提取关键词扩展
    important_keywords = ["手机号", "电话", "密码", "地址", "生日", "邮箱", "卡号", "账号"]
    for kw in important_keywords:
        if kw in query and kw not in search_queries:
            search_queries.append(kw)

    results = []
    seen_ids = set()
    for sq in search_queries:
        try:
            # 搜相关结果
            items = app_search.search(conn, query=sq, sort_mode="relevant", limit=20)
            for item in items:
                item_id = item.get('id') or item.get('entity_id')
                if item_id and item_id not in seen_ids:
                    seen_ids.add(item_id)
                    results.append(item)
        except Exception:
            continue

    return results[:limit]


def build_memory_context(results: list, max_items: int = 3) -> str:
    """把搜索结果构建成 LLM 可读的记忆提示
    输出会被注入到 system prompt 的【你的记忆】部分
    """
    if not results:
        return ""

    lines = []
    for r in results[:max_items]:
        title = r.get('title', '')
        summary = r.get('summary', '')
        body = r.get('body', '')
        time_str = r.get('time', '')

        content = summary or body or ''
        if content and len(content) > 120:
            content = content[:120] + '...'

        parts = []
        if title:
            parts.append(title)
        if content and content != title:
            parts.append(content)
        if time_str:
            parts.append(f"（{time_str}）")

        if parts:
            lines.append("• " + "：".join(parts[:2]) + (f" （{time_str}）" if time_str else ""))

    if not lines:
        return ""

    return "\n".join(lines)

def check_save_pending(text: str) -> tuple:
    """检查用户是否只表达了保存意图但没有具体内容
    返回 (need_wait, pending_text)
    核心规则：只有"是"+内容 或 数字 才算有实质内容
    """
    text_stripped = text.strip()

    # 有"是"且后面有实质内容 → 不需要等待
    if "是" in text_stripped:
        parts = text_stripped.split("是", 1)
        after_is = parts[1].strip() if len(parts) == 2 else ""
        # "是"后面有内容（超过2个字或包含数字）
        if after_is and (len(after_is) > 2 or any(c.isdigit() for c in after_is)):
            return (False, None)

    # 有数字 → 有具体内容
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
            # 后面没有内容 → 等待
            if not after:
                return (True, text_stripped)
            # 后面只有占位词（如"我的手机号"、"密码"）→ 等待
            # 占位词特征：长度短（<=8）且只包含记忆关键词
            placeholder_keywords = ["我的手机号", "我的电话", "我的地址", "我的生日",
                                     "我的邮箱", "我的卡号", "我的账号", "我的密码",
                                     "手机号", "电话", "地址", "生日", "邮箱", "卡号", "账号", "密码"]
            # 去掉"我的"前缀再比较
            after_clean = after.replace("我的", "").strip()
            if any(after_clean == kw or after == kw for kw in placeholder_keywords):
                return (True, text_stripped)
            # 后面内容很短（<=3字）→ 等待
            if len(after) <= 3:
                return (True, text_stripped)
            # 其他情况认为有内容
            return (False, None)

    # 短文本纯保存词
    if len(text_stripped) <= 6:
        short_intent = ["记住", "记下", "保存", "存下", "备忘"]
        if any(w == text_stripped for w in short_intent):
            return (True, text_stripped)

    return (False, None)

def extract_search_query(text: str) -> str:
    """从用户输入中提取搜索关键词"""
    query = text.strip()

    # 去掉命令前缀
    if query.startswith("/搜"):
        query = query[2:].strip(" ：,，,。")
    elif query.startswith("/search"):
        query = query[len("/search"):].strip()

    # 去掉常见的搜索引导词
    remove_prefixes = [
        "帮我找一下", "帮我查一下", "帮我找", "帮我查",
        "查一下", "搜一下", "找一下", "查找", "搜索",
        "我记过的", "我保存的", "我的",
    ]
    for prefix in remove_prefixes:
        if query.startswith(prefix):
            query = query[len(prefix):].strip(" ：,，,。")

    # 去掉疑问词
    remove_suffixes = ["是什么", "是多少", "在哪里", "在哪", "吗", "呢", "吧"]
    for suffix in remove_suffixes:
        if query.endswith(suffix):
            query = query[:-len(suffix)].strip(" ：,，,。")

    return query or text.strip()

def build_memory_metadata(text: str) -> tuple:
    """从用户输入中提取标题和摘要"""
    title = ""
    summary = ""
    try:
        plan = call_planning_model(text)
        title = (getattr(plan, "note_title", "") or "").strip()
        summary = (getattr(plan, "note_content", "") or "").strip()
    except Exception:
        pass

    # 智能提取标题
    if not title:
        # 尝试提取 "XX是YY" 结构
        if "我的" in text and ("是" in text or ":" in text or "：" in text):
            parts = text.replace("：", ":").split(":")
            if len(parts) >= 2:
                title = parts[0].strip()[:20]
            else:
                parts = text.split("是")
                if len(parts) >= 2:
                    title = ("我的" + parts[1].strip())[:20] if "我的" in parts[0] else parts[0].strip()[:20]

        # 提取关键词作为标题
        if not title:
            keywords = ["密码", "手机号", "电话", "地址", "生日", "邮箱", "卡号", "账号", "身份证", "车牌"]
            for kw in keywords:
                if kw in text:
                    title = f"我的{kw}"
                    break

    fallback_title = title or ((text[:18] + "…") if len(text) > 18 else (text or "记忆"))
    fallback_summary = summary or ((text[:80] + "…") if len(text) > 80 else (text or "记忆"))
    return (fallback_title, fallback_summary)

# ============ API 路由 ============

@app.route('/api/chat', methods=['POST'])
def chat():
    global SESSION_CHAT_HISTORY
    data = request.get_json()
    user_text = data.get('text', '').strip()
    if not user_text:
        return jsonify({'error': '请输入内容'}), 400

    conn = init_vault()
    vault_root = get_vault_root()
    intent = determine_intent(user_text)

    # ===== 保存意图 =====
    if intent == "save":
        need_wait, pending_text = check_save_pending(user_text)
        if need_wait:
            wait_replies = [
                '好呀，具体是什么内容呢？说给我听听～',
                '行，说具体内容吧~',
                '好嘞，告诉我具体内容~',
                '嗯嗯，说具体内容给我吧~',
            ]
            return jsonify({
                'type': 'assistant',
                'text': random.choice(wait_replies),
                'pending_save': pending_text
            })
        try:
            title, summary = build_memory_metadata(user_text)
            app_repo.remember_text_smart(conn, text=user_text, vault_root=vault_root, title=title, summary=summary)
            save_replies = [
                '记好啦~ 以后忘了随时问我呀',
                '收到~ 帮你记下了',
                '好嘞~ 记住了',
                '记好啦，放心~',
                '搞定~ 记好了',
            ]
            return jsonify({'type': 'assistant', 'text': random.choice(save_replies), 'saved': True})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    # ===== 搜索意图 =====
    if intent == "search":
        results = search_memory(conn, user_text, limit=8)

        if results:
            # 有结果 → 用 answer 模块生成自然回复 + 展示结果卡片
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
                    if r.get('summary'):
                        response_text += f"：{r['summary']}"
                else:
                    response_text = f'找到 {len(results)} 条相关记忆~'
            return jsonify({
                'type': 'assistant',
                'text': response_text,
                'results': results[:8]
            })
        else:
            # 没结果 → 温柔地说没找到
            return jsonify({
                'type': 'assistant',
                'text': random.choice([
                    '没找到呢～换个说法试试？',
                    '好像没记过这个，要不先记一下？',
                    '翻了一圈没找到~',
                ]),
                'results': []
            })

    # ===== 聊天意图 =====
    # chat 不搜记忆，直接让 LLM 聊天
    SESSION_CHAT_HISTORY.append({"role": "user", "content": user_text})
    response = call_llm_chat(user_text, SESSION_CHAT_HISTORY[-10:])
    if response:
        SESSION_CHAT_HISTORY.append({"role": "assistant", "content": response})
    else:
        response = random.choice([
            "嗯～我在呢，想聊啥或者想记啥都说哦~",
            "在呢在呢，说呗~",
            "听着呢，你说~",
        ])

    return jsonify({'type': 'assistant', 'text': response})

@app.route('/api/chat/confirm_save', methods=['POST'])
def confirm_save():
    global SESSION_CHAT_HISTORY
    data = request.get_json()
    text = data.get('text', '').strip()
    to_save = data.get('pending_text', '').strip()
    if not to_save:
        return jsonify({'error': '没有待保存的内容'}), 400

    conn = init_vault()
    vault_root = get_vault_root()
    decision = text.strip().lower()

    if decision in {"记住", "保存", "要", "好", "好的", "是", "嗯", "ok", "对", "没错", "行", "可以"}:
        title, summary = build_memory_metadata(to_save)
        app_repo.remember_text_smart(conn, text=to_save, vault_root=vault_root, title=title, summary=summary)
        confirm_replies = ['记好啦~', '收到~', '好嘞~', '记下来了~', '搞定~']
        return jsonify({'type': 'assistant', 'text': random.choice(confirm_replies)})

    if decision in {"不用", "不", "不要", "取消", "算了", "算了算了", "别"}:
        cancel_replies = ['好哒，那就不记啦~', '好嘞，不记了~', '行，不保存了~']
        return jsonify({'type': 'assistant', 'text': random.choice(cancel_replies)})

    final_text = to_save
    if text and text not in to_save:
        final_text = to_save + ' ' + text

    title, summary = build_memory_metadata(final_text)
    app_repo.remember_text_smart(conn, text=final_text, vault_root=vault_root, title=title, summary=summary)
    merge_replies = ['嗯嗯，帮你记好啦~', '收到~ 合并记下来了', '好嘞，都记下了~']
    return jsonify({'type': 'assistant', 'text': random.choice(merge_replies)})

@app.route('/api/search', methods=['POST'])
def search_memories():
    data = request.get_json()
    query = data.get('query', '').strip()
    if not query:
        return jsonify({'error': '请输入搜索内容'}), 400
    conn = init_vault()
    try:
        results = app_search.search(conn, query=query, sort_mode="relevant", limit=20)
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/save', methods=['POST'])
def save_memory():
    data = request.get_json()
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': '请输入要保存的内容'}), 400
    conn = init_vault()
    vault_root = get_vault_root()
    try:
        title, summary = build_memory_metadata(text)
        app_repo.remember_text_smart(conn, text=text, vault_root=vault_root, title=title, summary=summary)
        return jsonify({'success': True, 'message': '已保存'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
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

    conn = init_vault()
    vault_root = get_vault_root()
    paths = []
    for f in files:
        safe_name = f.filename.replace('\\', '/').replace('..', '').strip()
        safe_name = re.sub(r'[^\w\.\-\u4e00-\u9fff\u3400-\u4dbf]', '_', safe_name)
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
                conn2 = get_db_conn()
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
                            print(f"[图片理解] ID={mid}, caption={u.caption}, tags={u.tags}")
                        except Exception as e:
                            print(f"[图片理解失败] ID={mid}, error={e}")
                            import traceback
                            traceback.print_exc()
                conn2.commit()
            threading.Thread(target=background_understand, daemon=True).start()
            return jsonify({'success': True, 'message': f'已保存 {len(ids)} 个文件，图片正在后台理解中。'})
        else:
            return jsonify({'error': '无法保存文件'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/file/open', methods=['POST'])
def open_file():
    return jsonify({'success': False, 'message': '此功能在云端不可用'})

@app.route('/api/file/open_folder', methods=['POST'])
def open_file_folder():
    return jsonify({'success': False, 'message': '此功能在云端不可用'})

# 语音
def synthesize_with_qwen(text: str) -> bytes:
    try:
        from aip import AipSpeech
    except ImportError as e:
        print(f"[ERROR] Failed to import AipSpeech: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise RuntimeError(f"百度AIP包导入失败: {e}")
    app_id = os.getenv("BAIDU_APP_ID", "").strip()
    api_key_baidu = os.getenv("BAIDU_API_KEY", "").strip()
    secret_key = os.getenv("BAIDU_SECRET_KEY", "").strip()
    
    print(f"[TTS DEBUG] APP_ID={bool(app_id)}, API_KEY={bool(api_key_baidu)}, SECRET={bool(secret_key)}", file=sys.stderr, flush=True)
    
    if not all([app_id, api_key_baidu, secret_key]):
        missing = []
        if not app_id:
            missing.append("BAIDU_APP_ID")
        if not api_key_baidu:
            missing.append("BAIDU_API_KEY")
        if not secret_key:
            missing.append("BAIDU_SECRET_KEY")
        raise RuntimeError(f"百度语音API配置不完整，缺失: {', '.join(missing)}")
    try:
        client = AipSpeech(app_id, api_key_baidu, secret_key)
        text = text.strip()[:300]
        # aue=6 是 MP3 格式，直接返回不需要再包 WAV
        result = client.synthesis(text, 'zh', 1, {'per': 5, 'spd': 5, 'pit': 5, 'vol': 7, 'aue': 6})
        if isinstance(result, dict):
            raise RuntimeError(f"百度语音合成失败: {result.get('err_msg', '未知错误')}")
        if not result or len(result) < 100:
            raise RuntimeError("百度语音合成返回空音频")
        # aue=6 返回的是 MP3，前端可以直接播放
        return result
    except Exception as e:
        raise RuntimeError(f"百度语音合成失败: {e}")

# 语音识别（百度ASR）
def recognize_with_baidu(wav_bytes: bytes) -> str:
    """使用百度语音识别API将音频转为文字"""
    try:
        from aip import AipSpeech
    except ImportError as e:
        print(f"[ERROR] Failed to import AipSpeech: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise RuntimeError(f"百度AIP包导入失败: {e}")

    app_id = os.getenv("BAIDU_APP_ID", "").strip()
    api_key = os.getenv("BAIDU_API_KEY", "").strip()
    secret_key = os.getenv("BAIDU_SECRET_KEY", "").strip()

    print(f"[ASR DEBUG] APP_ID={bool(app_id)}, API_KEY={bool(api_key)}, SECRET={bool(secret_key)}", file=sys.stderr, flush=True)

    if not all([app_id, api_key, secret_key]):
        missing = []
        if not app_id:
            missing.append("BAIDU_APP_ID")
        if not api_key:
            missing.append("BAIDU_API_KEY")
        if not secret_key:
            missing.append("BAIDU_SECRET_KEY")
        raise RuntimeError(f"百度语音API配置不完整，缺失: {', '.join(missing)}")

    try:
        client = AipSpeech(app_id, api_key, secret_key)

        # 保存为临时WAV文件供百度API使用
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = tmp.name

        try:
            with open(tmp_path, 'rb') as f:
                result = client.asr(f.read(), 'wav', 16000, {
                    'dev_pid': 1537,  # 识别中文
                })
        finally:
            import os as os_module
            os_module.unlink(tmp_path)

        if 'result' in result and result['result']:
            return result['result'][0].strip()
        elif 'err_msg' in result:
            raise RuntimeError(f"百度语音识别失败: {result.get('err_msg', '未知错误')}")
        else:
            return ""
    except Exception as e:
        raise RuntimeError(f"语音识别失败: {e}")

@app.route('/api/speech_recognize', methods=['POST'])
def speech_recognize():
    """处理前端上传的语音，识别为文字"""
    if 'audio' not in request.files:
        return jsonify({'error': '请上传音频文件'}), 400

    audio_file = request.files['audio']
    try:
        # 读取音频数据
        audio_data = audio_file.read()

        if len(audio_data) < 1000:
            return jsonify({'error': '音频数据太短'}), 400

        # 尝试识别
        text = recognize_with_baidu(audio_data)

        if text:
            return jsonify({'text': text, 'success': True})
        else:
            return jsonify({'error': '未能识别到文字，请重试'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/speech_synthesize', methods=['POST'])
def speech_synthesize():
    data = request.get_json()
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': '请输入要合成的内容'}), 400
    try:
        wav_bytes = synthesize_with_qwen(text[:300])
        audio_b64 = base64.b64encode(wav_bytes).decode('utf-8')
        return jsonify({'audio': audio_b64})
    except Exception as e:
        return jsonify({'error': f'语音合成失败: {str(e)}'}), 500

@app.route('/api/voice_dialogue', methods=['POST'])
def voice_dialogue():
    """语音对话API - 接收语音，处理，返回回复语音"""
    if 'audio' not in request.files:
        return jsonify({'error': '请上传音频文件'}), 400

    audio_file = request.files['audio']
    try:
        # 读取音频数据
        audio_data = audio_file.read()

        if len(audio_data) < 1000:
            return jsonify({'error': '音频数据太短'}), 400

        # 语音识别
        user_text = recognize_with_baidu(audio_data)
        if not user_text:
            return jsonify({'error': '未能识别到文字，请重试'}), 400

        # 初始化数据库和vault
        conn = init_vault()
        vault_root = get_vault_root()

        # 判断意图
        intent = determine_intent(user_text)

        response_text = ""
        saved = False

        # ===== 保存意图 =====
        if intent == "save":
            need_wait, pending_text = check_save_pending(user_text)
            if need_wait:
                response_text = random.choice(['好呀，说具体内容给我吧～', '行，说具体内容吧~', '好嘞，告诉我具体内容~'])
            else:
                try:
                    title, summary = build_memory_metadata(user_text)
                    app_repo.remember_text_smart(conn, text=user_text, vault_root=vault_root, title=title, summary=summary)
                    save_voice_replies = ['记好啦~', '收到~', '好嘞~', '记下来了~']
                    response_text = random.choice(save_voice_replies)
                    saved = True
                except Exception as e:
                    response_text = f'保存失败: {str(e)}'

            # 生成语音回复
            try:
                audio_bytes = synthesize_with_qwen(response_text[:300])
                audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                return jsonify({
                    'user_text': user_text,
                    'response_text': response_text,
                    'audio': audio_b64,
                    'saved': saved
                })
            except Exception as e:
                return jsonify({
                    'user_text': user_text,
                    'response_text': response_text,
                    'error': f'语音合成失败: {str(e)}'
                }), 500

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
                        response_text = f"找到啦~ {r.get('title', '')}：{r.get('summary', r.get('body', ''))[:60]}"
                    else:
                        response_text = random.choice([
                            f'找到 {len(results)} 条相关记忆~',
                            f'找到了 {len(results)} 条~',
                        ])
            else:
                response_text = random.choice([
                    '没找到呢～换个说法试试？',
                    '好像没记过这个~',
                    '翻了一圈没找到~',
                ])

            # 生成语音回复
            try:
                audio_bytes = synthesize_with_qwen(response_text[:300])
                audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                return jsonify({
                    'user_text': user_text,
                    'response_text': response_text,
                    'audio': audio_b64,
                    'saved': saved,
                    'results': results[:8] if results else []
                })
            except Exception as e:
                return jsonify({
                    'user_text': user_text,
                    'response_text': response_text,
                    'error': f'语音合成失败: {str(e)}'
                }), 500

        # ===== 聊天意图 =====
        SESSION_CHAT_HISTORY.append({"role": "user", "content": user_text})
        response = call_llm_chat(user_text, SESSION_CHAT_HISTORY[-10:])
        if response:
            SESSION_CHAT_HISTORY.append({"role": "assistant", "content": response})
            response_text = response
        else:
            response_text = random.choice(["我在呢～想聊啥？", "嗯哼，说吧~", "听着呢~"])

        # 生成语音回复
        try:
            audio_bytes = synthesize_with_qwen(response_text[:300])
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            return jsonify({
                'user_text': user_text,
                'response_text': response_text,
                'audio': audio_b64,
                'saved': saved
            })
        except Exception as e:
            return jsonify({
                'user_text': user_text,
                'response_text': response_text,
                'error': f'语音合成失败: {str(e)}'
            }), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def call_llm_chat(user_query: str, history: list) -> str:
    """调用智谱大模型对话"""
    api_key = os.getenv("ZHIPU_API_KEY", "").strip()
    if not api_key:
        return None
    base_url = "https://open.bigmodel.cn/api/paas/v4"
    model = os.getenv("LOCAL_AGENT_MODEL", "GLM-4-Flash-250414")
    
    system_prompt = (
        "你是暖暖，用户身边一个真实、随性、有温度的好朋友。"
        "你们的关系很轻松，聊天不用拘束，就像平时微信唠嗑一样。"
        "你喜欢用口语、语气词，偶尔撒个娇、吐个槽、开开玩笑都很自然。"
        "不要客套，不要端着你好/请问/抱歉/谢谢理解这种客服腔。"
        "也不要提自己是人工智能、大模型或者AI，你就是暖暖本人。"
        "回复长短看心情、看场景：闲聊可以简短俏皮，用户让你讲故事、写作文、详细讲解、写代码之类的，就放开写，不用缩手缩脚，把内容说完说透。"
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_query})
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.85,
        "top_p": 0.92,
        "max_tokens": 4096,
        "tools": [{"type": "web_search", "web_search": {"enable": True}}]
    }
    try:
        from urllib import request as http_request
        req = http_request.Request(
            f"{base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with http_request.urlopen(req, timeout=90) as response:
            data = json.loads(response.read().decode("utf-8"))
        message = data["choices"][0]["message"]
        if "tool_calls" in message:
            messages.append(message)
            for tool_call in message["tool_calls"]:
                messages.append({"role": "tool", "tool_call_id": tool_call["id"], "content": "联网搜索已完成"})
            req2 = http_request.Request(
                f"{base_url.rstrip('/')}/chat/completions",
                data=json.dumps({"model": model, "messages": messages, "temperature": 0.7, "top_p": 0.9, "max_tokens": 4096}, ensure_ascii=False).encode("utf-8"),
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                method="POST",
            )
            with http_request.urlopen(req2, timeout=90) as response:
                data2 = json.loads(response.read().decode("utf-8"))
            final_message = data2["choices"][0]["message"]
            content = final_message.get("content", "")
            return str(content).strip() if content else "我查了一下，没找到相关信息呢。"
        content = message.get("content", "")
        return str(content).strip() if content else None
    except Exception as e:
        print(f"智谱大模型对话失败: {e}")
        return None

@app.route('/api/clear', methods=['POST'])
def clear_history():
    global SESSION_CHAT_HISTORY
    SESSION_CHAT_HISTORY = []
    return jsonify({'success': True})

# 图片
IMAGE_MIME_TYPES = {
    '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
    '.gif': 'image/gif', '.webp': 'image/webp', '.bmp': 'image/bmp', '.svg': 'image/svg+xml'
}

def _is_safe_path(file_path):
    try:
        p = Path(file_path).resolve()
        if '..' in file_path and str(p) != str(Path(file_path).absolute()):
            return False, "检测到路径遍历"
        ext = p.suffix.lower()
        if ext not in IMAGE_MIME_TYPES:
            return False, f"不允许的文件类型: {ext}"
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

@app.route('/api/memories/recent', methods=['GET'])
@rate_limit
@performance_monitor
@cached_response(lambda: f"recent_memories:{request.args.get('limit', '10')}", ttl=60)
def recent_memories():
    """获取最近的记忆（带缓存和优化）"""
    conn = get_db_conn()
    try:
        limit = int(request.args.get('limit', 10))
        if limit > 50:  # 限制最大数量
            limit = 50
        items = app_repo.list_recent(conn, limit=limit)
        return jsonify({'memories': items})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/memories/search', methods=['POST'])
@rate_limit
@performance_monitor
def search_memories_advanced():
    """高级记忆搜索"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        tags = data.get('tags', [])
        date_from = data.get('date_from')
        date_to = data.get('date_to')
        limit = min(int(data.get('limit', 20)), 100)  # 最大100条

        if not query and not tags:
            return jsonify({'error': '请提供搜索关键词或标签'}), 400

        conn = get_db_conn()

        # 构建搜索条件
        search_conditions = {
            'query': query,
            'tags': tags,
            'date_from': date_from,
            'date_to': date_to,
            'limit': limit
        }

        results = app_repo.search_advanced(conn, search_conditions)
        return jsonify({'results': results, 'total': len(results)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tags', methods=['GET'])
@rate_limit
@performance_monitor
@cached_response(lambda: "all_tags", ttl=300)  # 5分钟缓存
def get_all_tags():
    """获取所有标签"""
    try:
        conn = get_db_conn()
        # 如果没有get_all_tags方法，我们可以从现有记忆中提取
        tags = []
        try:
            tags = app_repo.get_all_tags(conn)
        except AttributeError:
            # 回退方案：从最近记忆中提取标签
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
@cached_response(lambda: "stats", ttl=60)  # 1分钟缓存
def get_stats():
    """获取统计信息"""
    try:
        conn = get_db_conn()
        # 获取总记忆数
        total_row = conn.execute("SELECT COUNT(*) as count FROM memories").fetchone()
        total_memories = int(total_row["count"]) if total_row else 0
        
        recent_memories = app_repo.list_recent(conn, limit=10)

        # 计算标签统计
        tag_counts = {}
        for memory in recent_memories:
            if 'tags' in memory and memory['tags']:
                for tag in memory['tags']:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1

        stats = {
            'total_memories': total_memories,
            'recent_count': len(recent_memories),
            'top_tags': sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            'last_updated': recent_memories[0].get('time') if recent_memories else None
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
    """更新记忆"""
    try:
        data = request.get_json()
        content = data.get('content', '').strip()
        if not content:
            return jsonify({'error': '内容不能为空'}), 400

        conn = get_db_conn()
        app_repo.update_memory(conn, memory_id, content)
        # 清除相关缓存
        response_cache.cache = {k: v for k, v in response_cache.cache.items()
                               if not k.startswith(f"recent_memories") and not k.startswith("stats")}
        return jsonify({'success': True, 'message': '记忆更新成功'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/memory/<int:memory_id>', methods=['DELETE'])
@rate_limit
@performance_monitor
def delete_memory(memory_id):
    """删除记忆"""
    try:
        conn = get_db_conn()
        app_repo.delete_memory(conn, memory_id)
        # 清除相关缓存
        response_cache.cache = {k: v for k, v in response_cache.cache.items()
                               if not k.startswith(f"recent_memories") and not k.startswith("stats")}
        return jsonify({'success': True, 'message': '记忆删除成功'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 启动
if __name__ == '__main__':
    load_env()
    init_vault()
    is_debug = os.environ.get('FLASK_DEBUG', '').lower() in ('true', '1', 'yes')
    print(f"""
╔════════════════════════════════════════╗
║         暖暖 - 记忆助手 Web 版          ║
╠════════════════════════════════════════╣
║  端口: {PORT}                              ║
║  数据目录: {DATA_DIR}                     ║
║  Vault目录: {VAULT_DIR}                   ║
╚════════════════════════════════════════╝
    """)
    app.run(host='0.0.0.0', port=PORT, debug=is_debug)
