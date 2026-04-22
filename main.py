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
    stripped = text.strip()
    if not stripped:
        return "chat"
    if stripped.startswith(("/聊", "/chat")):
        return "chat"
    if stripped.startswith(("/搜", "/search")):
        return "search"
    save_keywords = ("帮我记", "记一下", "记住", "记录一下", "保存一下",
        "帮我存", "存一下", "提醒我", "别忘了", "备忘",
        "把这个记下来", "收藏", "记下来", "要记住", "写下来")
    if any(k in stripped for k in save_keywords):
        return "save"
    search_keywords = ("帮我找", "帮我查", "查一下", "搜一下", "找一下",
        "查找", "找到", "搜索", "记得", "我记过",
        "之前记", "以前记", "我存过", "我的记忆", "记忆库",
        "保存过", "记录过", "查询", "检索", "有没有",
        "哪里", "在哪", "什么时候", "多少", "谁",
        "列出", "有哪些", "记了什么", "看看记忆")
    if any(k in stripped for k in search_keywords):
        return "search"
    memory_question_keywords = ("我记了什么", "记得什么", "存了什么", "保存了什么",
        "记忆里有", "记忆库有", "密码", "号码", "电话", "生日",
        "之前记的", "以前记的", "我记录过", "什么号", "什么密码")
    if any(k in stripped for k in memory_question_keywords):
        return "search"
    return "chat"

def check_save_pending(text: str) -> tuple:
    pure_intent_phrases = ["帮我记住", "帮我存", "记一下", "存一下", "保存", "好的", "好", "行", "嗯", "记住了"]
    placeholder_words = ["手机号", "电话", "地址", "密码", "账号", "生日", "日期", "时间", "金额", "价格",
        "邮箱", "银行卡", "卡号", "网址", "网站", "app", "软件", "会员", "到期", "有效期",
        "身份证", "驾驶证", "社保", "护照", "车牌", "订单号", "快递", "股票", "基金", "保险"]
    text_stripped = text.strip()
    is_pure_intent = text_stripped in pure_intent_phrases
    has_intent_word = any(phrase in text_stripped for phrase in ["帮我记住", "帮我存", "记一下", "存一下"])
    placeholder_after_content = False
    for word in placeholder_words:
        if word in text_stripped:
            idx = text_stripped.find(word)
            after_part = text_stripped[idx + len(word):]
            if len(after_part) > 2 or any(c.isdigit() for c in after_part):
                placeholder_after_content = True
                break
    has_content = len(text_stripped) > 15 or any(c.isdigit() for c in text_stripped)
    need_wait = is_pure_intent or (has_intent_word and not placeholder_after_content and not has_content)
    if need_wait:
        return (True, text_stripped)
    return (False, None)

def extract_search_query(text: str) -> str:
    query = text
    if query.startswith("/搜"):
        query = query[2:].strip(" ：,，,。")
    if query.startswith("/search"):
        query = query[len("/search"):].strip()
    return query

def build_memory_metadata(text: str) -> tuple:
    title = ""
    summary = ""
    try:
        plan = call_planning_model(text)
        title = (getattr(plan, "note_title", "") or "").strip()
        summary = (getattr(plan, "note_content", "") or "").strip()
    except Exception:
        pass
    fallback_title = (text[:18] + "…") if len(text) > 18 else (text or "记忆")
    fallback_summary = (text[:60] + "…") if len(text) > 60 else (text or "记忆")
    return (title or fallback_title, summary or fallback_summary)

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

    if intent == "search":
        query = extract_search_query(user_text)
        search_queries = [query]
        keywords = ["手机号", "电话", "密码", "地址", "生日"]
        for kw in keywords:
            if kw in query and kw not in search_queries:
                search_queries.append(kw)
        results = []
        seen_ids = set()
        for sq in search_queries:
            items = app_search.search(conn, query=sq, sort_mode="relevant", limit=20)
            for item in items:
                item_id = item.get('id')
                if item_id not in seen_ids:
                    seen_ids.add(item_id)
                    results.append(item)
        if not results:
            reply = call_llm_chat(user_text, SESSION_CHAT_HISTORY[-10:])
            SESSION_CHAT_HISTORY.append({"role": "user", "content": user_text})
            if reply:
                SESSION_CHAT_HISTORY.append({"role": "assistant", "content": reply})
            return jsonify({
                'type': 'assistant',
                'text': reply or '我没有找到相关内容，我们可以聊聊其他话题。',
                'results': []
            })
        ans = app_answer(user_text, results[:8])
        if ans.answer:
            top_time = str(results[0].get("time", "") or "").strip()
            response_text = ans.answer
            if top_time:
                response_text += f"\n（记忆时间：{top_time}）"
            return jsonify({
                'type': 'assistant',
                'text': response_text,
                'results': results[:8]
            })
        else:
            return jsonify({
                'type': 'assistant',
                'text': f'找到 {len(results)} 条相关记忆：',
                'results': results[:8]
            })

    if intent == "chat":
        need_wait, pending_text = check_save_pending(user_text)
        if need_wait:
            return jsonify({
                'type': 'assistant',
                'text': '好的，请说具体内容。',
                'pending_save': pending_text
            })
        SESSION_CHAT_HISTORY.append({"role": "user", "content": user_text})
        response = call_llm_chat(user_text, SESSION_CHAT_HISTORY[-10:])
        if response:
            SESSION_CHAT_HISTORY.append({"role": "assistant", "content": response})
        else:
            response = "我在。你想让我帮你查什么，还是帮我记住什么？"
        return jsonify({'type': 'assistant', 'text': response})

    if intent == "save":
        need_wait, pending_text = check_save_pending(user_text)
        if need_wait:
            return jsonify({
                'type': 'assistant',
                'text': '好的，请说具体内容。',
                'pending_save': pending_text
            })
        try:
            title, summary = build_memory_metadata(user_text)
            app_repo.remember_text_smart(conn, text=user_text, vault_root=vault_root, title=title, summary=summary)
            return jsonify({'type': 'assistant', 'text': '好的，我记住了。', 'saved': True})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({
        'type': 'assistant',
        'text': '嗯，我在。你想让我帮我查什么，还是帮我记住什么？'
    })

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

    if decision in {"记住", "保存", "要", "好", "好的", "是", "嗯", "ok"}:
        title, summary = build_memory_metadata(to_save)
        app_repo.remember_text_smart(conn, text=to_save, vault_root=vault_root, title=title, summary=summary)
        return jsonify({'type': 'assistant', 'text': '好的，我记住了。'})

    if decision in {"不用", "不", "不要", "取消", "算了"}:
        return jsonify({'type': 'assistant', 'text': '好的，不保存。'})

    final_text = to_save
    if text and text not in to_save:
        final_text = to_save + ' ' + text

    title, summary = build_memory_metadata(final_text)
    app_repo.remember_text_smart(conn, text=final_text, vault_root=vault_root, title=title, summary=summary)
    return jsonify({'type': 'assistant', 'text': '好的，已经帮你记住了。'})

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
                        except Exception:
                            pass
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
    except ImportError:
        raise RuntimeError("请先安装百度AIP: pip install baidu-aip")
    app_id = os.getenv("BAIDU_APP_ID", "").strip()
    api_key_baidu = os.getenv("BAIDU_API_KEY", "").strip()
    secret_key = os.getenv("BAIDU_SECRET_KEY", "").strip()
    if not all([app_id, api_key_baidu, secret_key]):
        raise RuntimeError("请配置百度语音 API")
    try:
        client = AipSpeech(app_id, api_key_baidu, secret_key)
        text = text.strip()[:300]
        result = client.synthesis(text, 'zh', 1, {'per': 5, 'spd': 5, 'pit': 5, 'vol': 7, 'aue': 6})
        if isinstance(result, dict):
            raise RuntimeError(f"百度语音合成失败: {result.get('err_msg', '未知错误')}")
        if not result or len(result) < 100:
            raise RuntimeError("百度语音合成返回空音频")
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(result)
        return wav_buffer.getvalue()
    except Exception as e:
        raise RuntimeError(f"百度语音合成失败: {e}")

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

def call_llm_chat(user_query: str, history: list) -> str:
    api_key = os.getenv("ZHIPU_API_KEY", "").strip()
    if not api_key:
        return None
    base_url = "https://open.bigmodel.cn/api/paas/v4"
    model = os.getenv("LOCAL_AGENT_MODEL", "glm-4-flash-250414")
    system_prompt = "你是一个友好的本地智能记忆助手，可以和用户闲聊、回答问题。请直接回答问题，不需要说明信息来源。"
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_query})
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.5,
        "max_tokens": 500,
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
                messages.append({"role": "tool", "tool_call_id": tool_call["id"], "content": "联网搜索已完成，模型将根据搜索结果生成回答。"})
            req2 = http_request.Request(
                f"{base_url.rstrip('/')}/chat/completions",
                data=json.dumps({"model": model, "messages": messages, "temperature": 0.5, "max_tokens": 500}, ensure_ascii=False).encode("utf-8"),
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                method="POST",
            )
            with http_request.urlopen(req2, timeout=90) as response:
                data2 = json.loads(response.read().decode("utf-8"))
            final_message = data2["choices"][0]["message"]
            content = final_message.get("content", "")
            return str(content).strip() if content else "我已联网搜索，但未能获取到有效信息。"
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
    except Exception as e:
        return Response('加载失败', status=500)

@app.route('/api/memories/recent', methods=['GET'])
def recent_memories():
    conn = get_db_conn()
    try:
        items = app_repo.list_recent(conn, limit=50)
        return jsonify({'memories': items})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/memory/<int:memory_id>', methods=['GET'])
def get_memory(memory_id):
    conn = get_db_conn()
    try:
        m = app_repo.get_memory(conn, memory_id)
        if not m:
            return jsonify({'error': '记忆不存在'}), 404
        return jsonify({'memory': m})
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
