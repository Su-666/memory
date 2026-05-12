"""
暖暖记忆助手 - 桌面客户端 v6.0
PySide6 + PyQt-Fluent-Widgets 实现
"""
import os
import sys
from pathlib import Path

# PyInstaller 打包后 SSL 证书路径修复
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    pass

APP_VERSION = "6.0"

# 设置 PortAudio DLL 搜索路径（用于打包后）
if hasattr(sys, '_MEIPASS'):
    _base = sys._MEIPASS
    _dll_dir = os.path.join(_base, '_sounddevice_data', 'portaudio-binaries')
    if os.path.exists(_dll_dir):
        os.environ['PATH'] = _dll_dir + os.pathsep + os.environ.get('PATH', '')
    os.environ['PATH'] = _base + os.pathsep + os.environ.get('PATH', '')
    if _base not in sys.path:
        sys.path.insert(0, _base)

# 启动日志
_LOG_DIR = Path(os.environ.get("APPDATA", ".")) / "MemoryAssistant"
_LOG_DIR.mkdir(exist_ok=True)
_LOG_FILE = _LOG_DIR / "startup.log"
with open(_LOG_FILE, "w", encoding="utf-8") as _f:
    _f.write(f"Python: {sys.version}\n")
    _f.write(f"Executable: {sys.executable}\n")
    _f.write(f"MEIPASS: {getattr(sys, '_MEIPASS', 'N/A')}\n")

def _load_dotenv(dotenv_path: Path) -> None:
    DEFAULT_CONFIG = """
ZHIPU_API_KEY=
LOCAL_AGENT_MODEL=glm-4-flash-250414
BAIDU_APP_ID=
BAIDU_API_KEY=
BAIDU_SECRET_KEY=
"""
    if dotenv_path.exists():
        try:
            text = dotenv_path.read_text(encoding="utf-8")
        except Exception:
            text = DEFAULT_CONFIG
    else:
        text = DEFAULT_CONFIG
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)

_load_dotenv(Path(__file__).resolve().parent / ".env")

def main():
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt
    from qfluentwidgets import setTheme, Theme

    # 高 DPI 支持
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("暖暖")
    app.setApplicationVersion(APP_VERSION)

    from ui.main_window import AgentWindow
    window = AgentWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
