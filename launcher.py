"""
暖暖记忆助手 - 桌面启动器
将 Flask Web 服务打包为桌面程序：后台运行 Flask，前台用 pywebview 显示原生窗口。

启动流程：
  1. 初始化用户数据目录（%APPDATA%/记忆助手）
  2. 首次启动时从模板生成 .env 配置
  3. 后台线程启动 Flask 服务
  4. 等待服务就绪后打开原生窗口
  5. 关闭窗口即退出程序

打包命令：python build_exe.py
"""
from __future__ import annotations

import os
import sys
import socket
import threading
import time
from pathlib import Path

APP_NAME = "记忆助手"
DEFAULT_WIDTH = 1180
DEFAULT_HEIGHT = 780
MIN_WIDTH = 960
MIN_HEIGHT = 640


# ============================================================
# 路径处理
# ============================================================

def is_frozen() -> bool:
    """是否运行在 PyInstaller 打包环境中"""
    return getattr(sys, "frozen", False)


def get_resource_root() -> Path:
    """获取资源根目录（打包后为 _MEIPASS，开发时为项目根目录）"""
    if is_frozen():
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent


def get_data_root() -> Path:
    """获取用户数据目录（%APPDATA%/记忆助手）"""
    base = os.environ.get("APPDATA") or str(Path.home())
    return Path(base) / APP_NAME


def ensure_data_dirs(data_root: Path) -> tuple[Path, Path]:
    """创建数据目录结构，返回 (data_dir, vault_dir)"""
    data_dir = data_root / "data"
    vault_dir = data_dir / "memory_vault"
    data_dir.mkdir(parents=True, exist_ok=True)
    vault_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, vault_dir


def ensure_env_file(data_root: Path) -> Path:
    """首次启动时创建空白 .env（用户需通过设置页面填写 API 密钥）"""
    env_path = data_root / ".env"
    if not env_path.exists():
        # 创建空白配置文件，所有密钥留空，强制用户通过设置页面填写
        env_path.write_text(
            "# 暖暖记忆助手配置文件\n"
            "# 请在程序设置页面（右上角齿轮图标）填写 API 密钥\n\n"
            "ZHIPU_API_KEY=\n"
            "LOCAL_AGENT_MODEL=glm-4-flash-250414\n"
            "LOCAL_AGENT_VISION_MODEL=glm-4v-flash\n"
            "BAIDU_APP_ID=\n"
            "BAIDU_API_KEY=\n"
            "BAIDU_SECRET_KEY=\n"
            "ADMIN_KEY=\n",
            encoding="utf-8",
        )
    return env_path


# ============================================================
# 端口管理
# ============================================================

def find_free_port(start: int = 5000, end: int = 5200) -> int:
    """在指定范围内寻找可用端口"""
    for port in range(start, end + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    # 兜底：让系统分配
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_port(port: int, timeout: float = 15.0) -> bool:
    """等待端口可连接"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.25)
    return False


# ============================================================
# Flask 服务
# ============================================================

def start_flask(port: int, resource_root: Path,
                data_dir: Path, vault_dir: Path, env_path: Path) -> None:
    """在当前线程启动 Flask 服务（应放入后台线程调用）"""

    # 设置环境变量，供 web/main.py 读取
    os.environ["PORT"] = str(port)
    os.environ["DATA_DIR"] = str(data_dir)
    os.environ["VAULT_DIR"] = str(vault_dir)
    os.environ["ENV_FILE"] = str(env_path)

    # 让 Python 能找到 app/ 和 web/ 下的模块
    web_dir = resource_root / "web"
    for p in (str(resource_root), str(web_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)

    # 切换工作目录到 web/，使 main.py 的相对路径正确
    os.chdir(str(web_dir))

    # 导入并运行 Flask 应用
    try:
        import main as web_main  # type: ignore[import-not-found]  # noqa: F401
        flask_app = web_main.app
    except Exception as e:
        print(f"[ERROR] Flask load failed: {e}", file=sys.stderr, flush=True)
        raise

    flask_app.run(
        host="127.0.0.1",
        port=port,
        debug=False,
        use_reloader=False,
        threaded=True,
    )


# ============================================================
# 主入口
# ============================================================

def main() -> None:
    resource_root = get_resource_root()
    data_root = get_data_root()

    # 初始化数据目录
    data_root.mkdir(parents=True, exist_ok=True)
    data_dir, vault_dir = ensure_data_dirs(data_root)
    env_path = ensure_env_file(data_root)

    port = find_free_port()

    # 后台启动 Flask 服务
    flask_thread = threading.Thread(
        target=start_flask,
        args=(port, resource_root, data_dir, vault_dir, env_path),
        name="flask-server",
        daemon=True,
    )
    flask_thread.start()

    # 等待服务就绪
    if not wait_for_port(port, timeout=20.0):
        msg = f"Flask server failed to start on port {port}"
        print(f"[ERROR] {msg}", file=sys.stderr, flush=True)
        _show_error_dialog(msg)
        sys.exit(1)

    url = f"http://127.0.0.1:{port}"

    # 打开原生窗口（使用 pywebview）
    try:
        import webview
    except ImportError:
        # 退化为系统浏览器
        import webbrowser
        webbrowser.open(url)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        return

    # 创建窗口 — 仅使用 pywebview 官方支持的参数
    webview.create_window(
        title="暖暖记忆助手",
        url=url,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        min_size=(MIN_WIDTH, MIN_HEIGHT),
        text_select=True,
    )
    webview.start(debug=False)


def _show_error_dialog(msg: str) -> None:
    """显示 Windows 错误对话框（无需 GUI 库）"""
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, msg, "暖暖记忆助手 - 启动错误", 0x10)
    except Exception:
        pass


if __name__ == "__main__":
    main()
