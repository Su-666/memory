"""
暖暖记忆助手 - 桌面客户端 v6.0
纯在线模式，通过 API 与服务端通信
"""
import base64
import html
import io
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any, TYPE_CHECKING
import uuid
import wave

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
    _internal_dir = _base
    if os.path.exists(_dll_dir):
        os.environ['PATH'] = _dll_dir + os.pathsep + os.environ.get('PATH', '')
    os.environ['PATH'] = _internal_dir + os.pathsep + os.environ.get('PATH', '')
    # 确保打包后的模块可以被导入
    if _base not in sys.path:
        sys.path.insert(0, _base)

# 启动日志（用于调试）
_LOG_DIR = Path(os.environ.get("APPDATA", ".")) / "MemoryAssistant"
_LOG_DIR.mkdir(exist_ok=True)
_LOG_FILE = _LOG_DIR / "startup.log"
with open(_LOG_FILE, "w", encoding="utf-8") as _f:
    _f.write(f"Python: {sys.version}\n")
    _f.write(f"Executable: {sys.executable}\n")
    _f.write(f"MEIPASS: {getattr(sys, '_MEIPASS', 'N/A')}\n")
    _f.write(f"sys.path[0]: {sys.path[0] if sys.path else 'N/A'}\n")
    try:
        import certifi
        _f.write(f"certifi: {certifi.where()}\n")
        _f.write(f"SSL_CERT_FILE env: {os.environ.get('SSL_CERT_FILE', 'N/A')}\n")
    except ImportError:
        _f.write("certifi: NOT INSTALLED\n")

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

try:
    import numpy as np
except ImportError:
    np = None

try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    from PyQt5.QtCore import (
        Qt, QThread, QSettings, pyqtSignal, QTimer, QObject, QUrl,
    )
    from PyQt5.QtGui import QFont, QFontMetrics, QPixmap, QDesktopServices, QTextOption
    from PyQt5.QtWidgets import (
        QApplication, QFileDialog, QFrame, QHBoxLayout, QLabel,
        QMainWindow, QMenu, QProgressBar, QPushButton, QScrollArea, QSizePolicy,
        QTextBrowser, QTextEdit, QVBoxLayout, QWidget, QDialog,
    )
except ImportError as exc:
    py = sys.executable
    raise SystemExit(
        "未安装 PyQt5。\n"
        f"当前 Python：{py}\n"
        f"  \"{py}\" -m pip install -U PyQt5"
    ) from exc

try:
    from ui.widgets.command_input import CommandInput, CommandSubmit, DropSubmit
except Exception:
    pass

from app.api_client import MemoryApiClient

if TYPE_CHECKING:
    import numpy as _np


# ============ 语音工具函数 ============

def init_sounddevice() -> bool:
    if sd is None:
        return False
    try:
        if hasattr(sys, '_MEIPASS'):
            dll_path = os.path.join(sys._MEIPASS, '_sounddevice_data', 'portaudio-binaries')
            if os.path.exists(dll_path):
                os.environ['PATH'] = dll_path + os.pathsep + os.environ.get('PATH', '')
        sd.query_devices()
        return True
    except Exception:
        return False


def play_wav_bytes(wav_bytes: bytes, stop_flag: list | None = None) -> None:
    import tempfile
    # 尝试 sounddevice
    if np is not None and sd is not None:
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                channels = wf.getnchannels()
                sample_rate = wf.getframerate()
                sample_width = wf.getsampwidth()
                raw = wf.readframes(wf.getnframes())
            if sample_width != 2:
                raise RuntimeError("不支持非 16-bit WAV")
            audio = np.frombuffer(raw, dtype=np.int16)
            if channels > 1:
                audio = audio.reshape(-1, channels)
            sd.play(audio, sample_rate)
            if stop_flag is not None:
                def wait_for_stop():
                    while sd.get_stream().active:
                        if stop_flag:
                            sd.stop()
                            return
                        time.sleep(0.1)
                threading.Thread(target=wait_for_stop, daemon=True).start()
            else:
                sd.wait()
            return
        except Exception:
            pass
    # 回退 PowerShell
    temp_wav = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(wav_bytes)
            temp_wav = f.name
        subprocess.run(['powershell', '-c', f'(New-Object System.Media.SoundPlayer("{temp_wav}")).PlaySync()'], check=True)
    except Exception as e:
        print(f"无法播放音频: {e}")
    finally:
        if temp_wav:
            try:
                os.unlink(temp_wav)
            except OSError:
                pass


# ============ 语音 Worker（纯在线，通过服务端 API） ============

class VoiceWorker(QObject):
    """语音处理 Worker - 录音本地，ASR/TTS 通过服务端 API"""
    status_changed = pyqtSignal(str, bool)
    listening_changed = pyqtSignal(bool)
    finished = pyqtSignal()
    dialogue_stopped = pyqtSignal()
    user_text_ready = pyqtSignal(str)
    voice_text_ready = pyqtSignal(str)

    def __init__(self, api_client: MemoryApiClient, dialogue_mode: bool = False) -> None:
        super().__init__()
        self._api = api_client
        self._stop_requested = False
        self._recorded_chunks: list[bytes] = []
        self._dialogue_mode = dialogue_mode
        self._is_speaking = False
        self._playback_stop_flag: list = []

    def stop(self) -> None:
        self._stop_requested = True
        if self._is_speaking:
            self._playback_stop_flag.append(True)

    def run(self) -> None:
        if np is None or sd is None:
            self.status_changed.emit("缺少语音依赖，请安装：pip install sounddevice numpy", True)
            self.finished.emit()
            return
        if not init_sounddevice():
            self.status_changed.emit("语音设备初始化失败", True)
            self.finished.emit()
            return

        self._stop_requested = False
        try:
            if self._dialogue_mode:
                self._run_dialogue()
            else:
                self._run_sync()
        except Exception as e:
            self.status_changed.emit(f"语音服务失败: {e}", True)
        finally:
            self.listening_changed.emit(False)
            self.finished.emit()
            self.dialogue_stopped.emit()

    def _run_sync(self) -> None:
        sample_rate = 16000
        self._recorded_chunks = []
        self.listening_changed.emit(True)
        self.status_changed.emit("正在录音，请说话。", False)

        with sd.InputStream(samplerate=sample_rate, channels=1, dtype="int16", callback=self._audio_callback):
            start = time.time()
            while not self._stop_requested and (time.time() - start < 20):
                sd.sleep(100)

        pcm = b"".join(self._recorded_chunks)
        if not pcm:
            self.status_changed.emit("没有录到声音，请重试。", True)
            return

        # 将 PCM 转为 WAV 发送到服务端
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm)
        wav_bytes = wav_buf.getvalue()

        try:
            result = self._api.speech_recognize(wav_bytes)
            text = result.get("text", "")
            if text:
                self.user_text_ready.emit(text)
                self.voice_text_ready.emit(text)
                self.status_changed.emit("语音识别完成。", False)
            else:
                self.status_changed.emit("没有识别到内容。", True)
        except Exception as e:
            self.status_changed.emit(f"语音识别失败：{e}", True)
        self.listening_changed.emit(False)

    def _run_dialogue(self) -> None:
        sample_rate = 16000
        max_rounds = 50

        for _ in range(max_rounds):
            if self._stop_requested:
                break

            self._recorded_chunks = []
            speech_started = False
            silence_start = None
            recording_start = None

            self.listening_changed.emit(True)
            self.status_changed.emit("请说话（说'退出'结束）...", False)

            try:
                with sd.InputStream(samplerate=sample_rate, channels=1, dtype="int16", callback=self._audio_callback):
                    while not self._stop_requested:
                        sd.sleep(50)
                        if not self._recorded_chunks:
                            continue
                        last_chunk = self._recorded_chunks[-1]
                        audio_data = np.frombuffer(last_chunk, dtype=np.int16)
                        rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))

                        if rms > 800:
                            if not speech_started:
                                speech_started = True
                                recording_start = time.time()
                                self.status_changed.emit("正在聆听...", False)
                            silence_start = None
                        elif speech_started:
                            if silence_start is None:
                                silence_start = time.time()
                            elif time.time() - silence_start > 2.0:
                                break
                        if recording_start and (time.time() - recording_start > 30):
                            break

                if self._stop_requested:
                    break

                self.listening_changed.emit(False)
                pcm = b"".join(self._recorded_chunks)
                if not pcm or len(pcm) < sample_rate * 2:
                    self.listening_changed.emit(True)
                    self.status_changed.emit("没听清，请再说一次...", False)
                    continue

                # PCM -> WAV -> 服务端 ASR
                wav_buf = io.BytesIO()
                with wave.open(wav_buf, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    wf.writeframes(pcm)
                wav_bytes = wav_buf.getvalue()

                self.status_changed.emit("正在识别...", False)
                try:
                    result = self._api.speech_recognize(wav_bytes)
                    user_text = result.get("text", "")
                except Exception as e:
                    self.status_changed.emit(f"语音识别失败：{e}", False)
                    continue

                if not user_text or not user_text.strip():
                    self.status_changed.emit("没听清，请再说一次...", False)
                    continue

                self.user_text_ready.emit(user_text)

                if any(w in user_text for w in ["退出", "不用了", "结束", "拜拜", "再见", "停"]):
                    self._speak_text_via_api("好的，对话结束。有什么需要再叫我。")
                    break

                self.voice_text_ready.emit(user_text)

            except Exception as exc:
                print(f"对话异常: {exc}")
                self._speak_text_via_api("出了点小问题，请再说一次。")

        self._speak_text_via_api("对话已结束。")
        self.dialogue_stopped.emit()

    def speak_text(self, text: str) -> None:
        self._speak_text_via_api(text)

    def stop_speaking(self) -> None:
        self._playback_stop_flag.append(True)

    def _speak_text_via_api(self, text: str) -> None:
        try:
            self._is_speaking = True
            self._playback_stop_flag = []
            self.status_changed.emit("正在回复...", False)
            result = self._api.speech_synthesize(text)
            audio_b64 = result.get("audio", "")
            if audio_b64:
                wav_bytes = base64.b64decode(audio_b64)
                play_wav_bytes(wav_bytes, self._playback_stop_flag)
        except Exception as e:
            print(f"语音合成失败: {e}")
        finally:
            self._is_speaking = False

    def _audio_callback(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        if sd is None:
            return
        if not self._stop_requested:
            try:
                self._recorded_chunks.append(indata.copy().tobytes())
            except Exception:
                pass
        else:
            raise sd.CallbackStop()


# ============ 统计面板对话框 ============

class StatsDialog(QDialog):
    _stats_result = pyqtSignal(dict)
    _stats_error = pyqtSignal(str)

    def __init__(self, api_client: MemoryApiClient, parent=None, dark_mode: bool = False):
        super().__init__(parent)
        self.setWindowTitle("记忆统计")
        self.setMinimumSize(380, 420)
        self._api = api_client
        self._dark_mode = dark_mode
        self._stats_result.connect(self._render_stats)
        self._stats_error.connect(lambda e: self._show_error(str(e)))
        self._build_ui()
        self._load_stats()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 28, 28, 28)
        layout.setSpacing(18)

        self._title_lbl = QLabel("记忆统计")
        if self._dark_mode:
            self._title_lbl.setStyleSheet("font-size:24px; font-weight:900; color:#6ee7c0; letter-spacing:-0.01em;")
        else:
            self._title_lbl.setStyleSheet("font-size:24px; font-weight:900; color:#0a5c56; letter-spacing:-0.01em;")
        layout.addWidget(self._title_lbl)

        self._stats_container = QVBoxLayout()
        self._stats_container.setSpacing(14)
        layout.addLayout(self._stats_container)

        self._loading_lbl = QLabel("加载中...")
        if self._dark_mode:
            self._loading_lbl.setStyleSheet("color:#5a9a8a; font-size:14px;")
        else:
            self._loading_lbl.setStyleSheet("color:#7a9a8e; font-size:14px;")
        self._stats_container.addWidget(self._loading_lbl)

        layout.addStretch(1)

        close_btn = QPushButton("关闭")
        close_btn.setCursor(Qt.PointingHandCursor)
        if self._dark_mode:
            close_btn.setStyleSheet(
                "QPushButton{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #0f766e,stop:1 #14a89a);"
                "color:#fff; border:none; border-radius:20px; padding:10px 28px; font-weight:700; font-size:14px;}"
                "QPushButton:hover{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #14a89a,stop:1 #20d4b8);}"
            )
        else:
            close_btn.setStyleSheet(
                "QPushButton{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #0d6b64,stop:1 #0f766e);"
                "color:#fff; border:none; border-radius:20px; padding:10px 28px; font-weight:700; font-size:14px;}"
                "QPushButton:hover{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #0a5c56,stop:1 #0d6b64);}"
            )
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn, 0, Qt.AlignRight)

    def _load_stats(self):
        def _do():
            try:
                stats = self._api.get_stats()
                self._stats_result.emit(stats)
            except Exception as e:
                self._stats_error.emit(str(e))
        threading.Thread(target=_do, daemon=True).start()

    def _render_stats(self, stats: dict):
        # 清空
        while self._stats_container.count():
            item = self._stats_container.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        items = [
            ("总记忆数", str(stats.get("total_memories", 0))),
            ("最近记忆", str(stats.get("recent_count", 0))),
            ("文件数量", str(stats.get("total_files", 0))),
            ("存储大小", stats.get("storage_formatted", "未知")),
            ("最后更新", stats.get("last_updated", "无")),
        ]
        for label, value in items:
            row = QHBoxLayout()
            lbl = QLabel(label)
            val = QLabel(value)
            if self._dark_mode:
                lbl.setStyleSheet("color:#7ab8a8; font-size:14px; font-weight:500;")
                val.setStyleSheet("color:#6ee7c0; font-size:17px; font-weight:800;")
            else:
                lbl.setStyleSheet("color:#5a8a7e; font-size:14px; font-weight:500;")
                val.setStyleSheet("color:#0a5c56; font-size:17px; font-weight:800;")
            row.addWidget(lbl)
            row.addStretch(1)
            row.addWidget(val)
            self._stats_container.addLayout(row)

        top_tags = stats.get("top_tags", [])
        if top_tags:
            sep = QLabel("热门标签")
            tags_layout = QHBoxLayout()
            tags_layout.setSpacing(8)
            if self._dark_mode:
                sep.setStyleSheet("color:#e8c878; font-size:12px; font-weight:700; margin-top:10px; letter-spacing:0.05em;")
            else:
                sep.setStyleSheet("color:#8a6a28; font-size:12px; font-weight:700; margin-top:10px; letter-spacing:0.05em;")
            self._stats_container.addWidget(sep)
            for tag, count in top_tags[:8]:
                chip = QLabel(f"{tag} ({count})")
                if self._dark_mode:
                    chip.setStyleSheet(
                        "background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 rgba(20,61,56,0.9),stop:1 rgba(30,70,60,0.9));"
                        "border:1px solid rgba(212,162,89,0.15); border-radius:16px;"
                        "padding:5px 12px; color:#e8c878; font-size:12px; font-weight:600;"
                    )
                else:
                    chip.setStyleSheet(
                        "background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #fff9f0,stop:1 #fff3e0);"
                        "border:1px solid rgba(218,165,32,0.15); border-radius:16px;"
                        "padding:5px 12px; color:#8a6a28; font-size:12px; font-weight:600;"
                    )
                tags_layout.addWidget(chip)
            tags_layout.addStretch(1)
            self._stats_container.addLayout(tags_layout)

    def _show_error(self, msg: str):
        while self._stats_container.count():
            item = self._stats_container.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        err = QLabel(f"加载失败: {msg}")
        if self._dark_mode:
            err.setStyleSheet("color:#e87a5a; font-size:14px;")
        else:
            err.setStyleSheet("color:#b6491d; font-size:14px;")
        self._stats_container.addWidget(err)


# ============ 图片放大对话框 ============

class ImageZoomDialog(QDialog):
    def __init__(self, pixmap: QPixmap, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setStyleSheet("background: rgba(0,0,0,0.92);")
        self._pix = pixmap

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._img_label = QLabel()
        self._img_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._img_label, 1)

        # 关闭按钮 - 右上角
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(40, 40)
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.setStyleSheet(
            "QPushButton{background:rgba(255,255,255,0.12); color:#fff; border:1px solid rgba(255,255,255,0.08);"
            "border-radius:20px; font-size:18px;}"
            "QPushButton:hover{background:rgba(255,255,255,0.25);}"
        )
        close_btn.clicked.connect(self.close)
        close_btn.setParent(self)
        close_btn.move(20, 20)

        # 底部提示标签
        self._hint_lbl = QLabel("点击任意位置关闭 · 滚轮缩放")
        self._hint_lbl.setStyleSheet("color:rgba(255,255,255,0.4); font-size:12px; padding:8px;")
        self._hint_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._hint_lbl)

        self._scale_factor = 1.0
        self._update_pixmap()

    def _update_pixmap(self):
        if self._pix.isNull():
            return
        sw = self.width() - 40
        sh = self.height() - 40
        scaled = self._pix.scaled(sw, sh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._img_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_pixmap()

    def mousePressEvent(self, event):
        self.close()


# ============ 主窗口 ============

class AgentWindow(QMainWindow):
    # 跨线程 UI 更新信号
    _health_result = pyqtSignal(bool)
    _chat_result_signal = pyqtSignal(dict)
    _chat_error_signal = pyqtSignal(str)
    _upload_result_signal = pyqtSignal(dict)
    _upload_error_signal = pyqtSignal(str)
    _voice_result_signal = pyqtSignal(str, object, object)
    _voice_error_signal = pyqtSignal(str)
    _update_bar_signal = pyqtSignal(str, str)

    def __init__(self) -> None:
        super().__init__()
        init_sounddevice()

        self.voice_thread: QThread | None = None
        self.voice_worker: VoiceWorker | None = None
        self._settings = QSettings("", "MemoryAssistant")
        self._pending_save_text: str | None = None
        self._refit_debounce_timer: QTimer | None = None
        self._chat_history: list[dict] = []
        self._chat_bubbles: list[QWidget] = []  # 跟踪气泡，避免 findChildren
        self._api: MemoryApiClient | None = None
        self._client_id: str = ""
        self._online: bool = False
        self._health_timer: QTimer | None = None
        self._version_timer: QTimer | None = None
        self._dark_mode: bool = False
        self._is_sending: bool = False  # 防止重复发送
        self._typing_widget: QWidget | None = None  # 打字指示器
        self._MAX_CHAT_MESSAGES = 120  # 最多保留消息数

        # 气泡参数
        self._bubble_radius = 17
        self._bubble_body_font_px = 18
        self._bubble_title_font_px = 12
        self._bubble_pad_x = 8
        self._bubble_pad_y = 8
        self._bubble_user_body_font_px = 18
        self._bubble_user_pad_x = 8
        self._bubble_user_pad_y = 8
        self._bubble_min_width_px = 40
        self._preview_base_w = 320
        self._preview_base_h = 200

        # 无边框窗口
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setAttribute(Qt.WA_TranslucentBackground, False)

        self._build_ui()
        self._apply_style()
        self._load_defaults()
        self._restore_window_state()
        self._init_backend()
        self._init_version_check()

    # ---- 自定义标题栏拖拽 ----
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and event.pos().y() < self._title_bar.height():
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if hasattr(self, '_drag_pos') and event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        self._drag_pos = None

    # ---- UI 构建 ----
    def _build_ui(self) -> None:
        self.setWindowTitle("暖暖")
        self.resize(820, 960)
        self.setMinimumSize(620, 780)

        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QVBoxLayout(root)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ---- 自定义标题栏 ----
        self._title_bar = QWidget()
        self._title_bar.setObjectName("titleBar")
        self._title_bar.setFixedHeight(48)
        tb_layout = QHBoxLayout(self._title_bar)
        tb_layout.setContentsMargins(16, 0, 12, 0)

        title_lbl = QLabel("暖暖")
        title_lbl.setObjectName("titleBarLabel")
        tb_layout.addWidget(title_lbl)
        tb_layout.addStretch(1)

        # 连接状态
        self._conn_status_label = QLabel("○ 离线")
        self._conn_status_label.setObjectName("connStatusLabel")
        tb_layout.addWidget(self._conn_status_label)
        tb_layout.addSpacing(12)

        # 最小化 / 最大化 / 关闭
        for text, slot in [("─", self.showMinimized), ("□", self._toggle_maximize), ("✕", self.close)]:
            btn = QPushButton(text)
            btn.setFixedSize(32, 32)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setObjectName("titleBtn")
            btn.clicked.connect(slot)
            tb_layout.addWidget(btn)

        main_layout.addWidget(self._title_bar)

        # ---- 顶部信息区 ----
        header_widget = QWidget()
        header_widget.setObjectName("headerWidget")
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(28, 18, 28, 18)
        header_layout.setSpacing(16)

        left_col = QVBoxLayout()
        left_col.setSpacing(4)
        header_badge = QLabel("LOCAL MEMORY STUDIO")
        header_badge.setObjectName("headerBadge")
        header_badge.setFixedSize(190, 26)
        header_badge.setAlignment(Qt.AlignCenter)
        header = QLabel("暖暖")
        header.setObjectName("headerTitle")
        subtitle = QLabel("把零散念头、语音和附件整理成可搜索的私人记忆库")
        subtitle.setObjectName("headerSubtitle")
        left_col.addWidget(header_badge, 0, Qt.AlignLeft)
        left_col.addWidget(header, 0, Qt.AlignLeft)
        left_col.addWidget(subtitle, 0, Qt.AlignLeft)

        right_col = QHBoxLayout()
        right_col.setSpacing(8)

        # 暗色模式切换
        self._theme_btn = QPushButton("🌙")
        self._theme_btn.setObjectName("headerThemeBtn")
        self._theme_btn.setCursor(Qt.PointingHandCursor)
        self._theme_btn.setFixedSize(36, 36)
        self._theme_btn.setToolTip("切换暗色/亮色模式")
        self._theme_btn.clicked.connect(self._toggle_theme)
        right_col.addWidget(self._theme_btn)

        # 统计按钮
        stats_btn = QPushButton("📊")
        stats_btn.setObjectName("headerStatsBtn")
        stats_btn.setCursor(Qt.PointingHandCursor)
        stats_btn.setFixedSize(36, 36)
        stats_btn.setToolTip("查看记忆统计")
        stats_btn.clicked.connect(self._show_stats)
        right_col.addWidget(stats_btn)

        self.clear_btn = QPushButton("清空对话")
        self.clear_btn.setObjectName("headerClearBtn")
        self.clear_btn.setCursor(Qt.PointingHandCursor)
        self.clear_btn.clicked.connect(self._clear_query)
        right_col.addWidget(self.clear_btn)

        header_layout.addLayout(left_col, 1)
        header_layout.addLayout(right_col, 0)
        main_layout.addWidget(header_widget)

        # ---- 版本更新通知条 ----
        self._update_bar = QWidget()
        self._update_bar.setObjectName("updateBar")
        self._update_bar.setVisible(False)
        update_layout = QHBoxLayout(self._update_bar)
        update_layout.setContentsMargins(16, 6, 16, 6)
        self._update_label = QLabel()
        self._update_label.setObjectName("updateLabel")
        self._update_label.setCursor(Qt.PointingHandCursor)
        self._update_label.mousePressEvent = lambda e: self._open_download_url()
        update_layout.addWidget(self._update_label, 1)
        close_update = QPushButton("✕")
        close_update.setObjectName("updateCloseBtn")
        close_update.setFixedSize(24, 24)
        close_update.clicked.connect(lambda: self._update_bar.setVisible(False))
        update_layout.addWidget(close_update)
        main_layout.addWidget(self._update_bar)

        # ---- 离线通知条 ----
        self._offline_bar = QWidget()
        self._offline_bar.setObjectName("offlineBar")
        self._offline_bar.setVisible(False)
        offline_layout = QHBoxLayout(self._offline_bar)
        offline_layout.setContentsMargins(16, 6, 16, 6)
        self._offline_label = QLabel("正在尝试重新连接...")
        self._offline_label.setObjectName("offlineLabel")
        offline_layout.addWidget(self._offline_label, 1)
        main_layout.addWidget(self._offline_bar)

        self._build_statusbar()

        # ---- 聊天主区域 ----
        chat_container = QWidget()
        chat_container.setObjectName("chatContainer")
        chat_layout = QVBoxLayout(chat_container)
        chat_layout.setContentsMargins(2, 4, 2, 4)
        chat_layout.setSpacing(6)

        self.chat_view = QScrollArea()
        self.chat_view.setObjectName("chatView")
        self.chat_view.setWidgetResizable(True)
        self.chat_view.setFrameShape(QFrame.NoFrame)
        self.chat_view.viewport().setStyleSheet("background:transparent;")

        self._chat_inner = QWidget()
        self._chat_inner.setObjectName("chatInner")
        self._chat_layout = QVBoxLayout(self._chat_inner)
        self._chat_layout.setContentsMargins(0, 0, 0, 0)
        self._chat_layout.setSpacing(12)
        self._chat_layout.addStretch(1)
        self._chat_layout.setAlignment(Qt.AlignTop)
        self.chat_view.setWidget(self._chat_inner)
        chat_layout.addWidget(self.chat_view, 1)

        # ---- 待保存指示条 ----
        self._pending_bar = QWidget()
        self._pending_bar.setObjectName("pendingBar")
        self._pending_bar.setVisible(False)
        pending_layout = QHBoxLayout(self._pending_bar)
        pending_layout.setContentsMargins(16, 4, 16, 4)
        pending_lbl = QLabel("⏳ 等待补充内容...")
        pending_lbl.setObjectName("pendingLabel")
        pending_layout.addWidget(pending_lbl, 1)
        chat_layout.addWidget(self._pending_bar)

        # ---- 输入区域 ----
        input_widget = QWidget()
        input_widget.setObjectName("inputWidget")
        input_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        input_layout = QHBoxLayout(input_widget)
        input_layout.setContentsMargins(12, 6, 8, 6)
        input_layout.setSpacing(10)

        # 附件按钮
        self._attach_btn = QPushButton("📎")
        self._attach_btn.setObjectName("attachBtn")
        self._attach_btn.setFixedSize(42, 42)
        self._attach_btn.setCursor(Qt.PointingHandCursor)
        self._attach_btn.setToolTip("选择文件上传")
        self._attach_btn.clicked.connect(self._choose_files)
        input_layout.addWidget(self._attach_btn)

        self.command_input = CommandInput()
        self.command_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.command_input.submitted_text.connect(self._handle_chat_submit_text)
        self.command_input.submitted_drop.connect(self._handle_chat_submit_drop)

        self.voice_btn = QPushButton("🎤")
        self.voice_btn.setObjectName("voiceBtn")
        self.voice_btn.setFixedSize(42, 42)
        self.voice_btn.setToolTip("语音对话")
        self.voice_btn.clicked.connect(self._toggle_voice_dialogue)

        self.send_btn = QPushButton("➜")
        self.send_btn.setObjectName("sendBtn")
        self.send_btn.setFixedSize(48, 48)
        self.send_btn.setToolTip("发送")
        self.send_btn.clicked.connect(self._submit_command_input)

        input_layout.addWidget(self.command_input, 1)
        input_layout.addWidget(self.voice_btn)
        input_layout.addWidget(self.send_btn)

        outer_input = QVBoxLayout()
        outer_input.setSpacing(0)
        outer_input.setContentsMargins(12, 2, 12, 2)
        outer_input.addWidget(input_widget)
        chat_layout.addLayout(outer_input)

        main_layout.addWidget(chat_container, 1)

        self._setup_shortcuts()
        self._append_assistant(
            "你好！我是暖暖，你的贴心记忆助手。\n\n"
            '你可以直接告诉我想记住什么，或像聊天一样问我"之前存过什么"。\n'
            "也支持语音输入、拖拽图片和文件。"
        )

    def _toggle_maximize(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def _build_statusbar(self) -> None:
        self.statusBar().setSizeGripEnabled(True)
        self._progress = QProgressBar(self)
        self._progress.setVisible(False)
        self._progress.setMaximumWidth(220)
        self._progress.setRange(0, 0)
        self.statusBar().addPermanentWidget(self._progress)
        self.statusBar().showMessage("准备就绪")

    def _setup_shortcuts(self) -> None:
        self.command_input.focus_input()
        self.chat_view.viewport().setContextMenuPolicy(Qt.CustomContextMenu)
        self.chat_view.viewport().customContextMenuRequested.connect(self._show_chat_context_menu)

    def _scroll_chat_to_bottom(self) -> None:
        bar = self.chat_view.verticalScrollBar()
        bar.setSliderPosition(bar.maximum())
        bar.setValue(bar.maximum())

    def eventFilter(self, obj, event) -> bool:
        target = obj
        if not isinstance(target, QTextBrowser):
            target = obj.parent()
        if isinstance(target, QTextBrowser):
            etype = event.type()
            # 只拦截键盘导航事件（防止用户方向键滚动气泡）
            if etype == 6:  # KeyPress
                key = event.key()
                if key in (Qt.Key_Up, Qt.Key_Down, Qt.Key_PageUp, Qt.Key_PageDown, Qt.Key_Left, Qt.Key_Right):
                    return True
        return super().eventFilter(obj, event)

    def _chat_available_width(self) -> int:
        try:
            return max(420, int(self.chat_view.viewport().width()))
        except Exception:
            return max(420, int(self.width()))

    def _responsive_params(self) -> dict[str, int | float]:
        w = self._chat_available_width()
        preview_w = int(min(self._preview_base_w, max(220, w * 0.28)))
        preview_h = int(preview_w * 0.62)
        return {"avail": w, "preview_w": preview_w, "preview_h": preview_h}

    def _tighten_bubble_document_layout(self, doc) -> None:
        try:
            doc.setDocumentMargin(0)
            root = doc.rootFrame()
            rfmt = root.frameFormat()
            rfmt.setMargin(0)
            rfmt.setPadding(0)
            root.setFrameFormat(rfmt)
        except Exception:
            pass

    def _fit_text_browser(self, tb: QTextBrowser) -> None:
        try:
            doc = tb.document()
            self._tighten_bubble_document_layout(doc)
            viewport_w = tb.viewport().width()
            if viewport_w <= 0:
                viewport_w = tb.width()
            doc.setTextWidth(max(0, viewport_w))
            h_doc = int(doc.size().height())
            fm = QFontMetrics(tb.font())
            plain = (tb.toPlainText() or "").rstrip("\n")
            if "\n" not in plain and viewport_w > 0 and fm.horizontalAdvance(plain) <= viewport_w + 2:
                h = fm.height()
            else:
                h = max(fm.height(), h_doc)
            tb.setFixedHeight(max(24, h + 4))
        except Exception:
            return

    def _schedule_refit_chat_bubbles(self) -> None:
        if self._refit_debounce_timer is None:
            self._refit_debounce_timer = QTimer(self)
            self._refit_debounce_timer.setSingleShot(True)
            self._refit_debounce_timer.timeout.connect(self._refit_chat_bubbles)
        self._refit_debounce_timer.start(200)  # 200ms 防抖，避免频繁触发

    def _refit_chat_bubbles(self) -> None:
        """只重新计算最近几个气泡的宽度，不遍历全部控件"""
        try:
            if not hasattr(self, "_chat_inner"):
                return
            rp = self._responsive_params()
            avail = int(rp["avail"])
            pad_m = self._bubble_pad_x

            # 只处理最后 8 个气泡（新消息通常在底部）
            recent_bubbles = self._chat_bubbles[-8:] if len(self._chat_bubbles) > 8 else self._chat_bubbles
            for outer in recent_bubbles:
                frame = outer.findChild(QFrame, "chatBubbleFrame")
                if not frame:
                    continue
                body = frame.findChild(QTextBrowser, "chatBubbleBody")
                plain = str(frame.property("plainText") or "")
                title = str(frame.property("bubbleTitle") or "")
                if body is not None and body.font():
                    fm_body = QFontMetrics(body.font())
                    lines = (plain or "").splitlines() or [""]
                    longest = max((ln.strip() for ln in lines), key=len, default="")
                    text_w = fm_body.horizontalAdvance(longest)
                    title_w = 0
                    if title:
                        fm_title = QFontMetrics(QFont(body.font().family(), self._bubble_title_font_px))
                        title_w = fm_title.horizontalAdvance(title)
                    content_w = max(text_w, title_w)
                    char_count = len((plain or "").strip())
                    dyn_pad = max(6, pad_m // 2) if char_count <= 2 else (max(8, pad_m - 4) if char_count <= 6 else pad_m)
                    new_frame_w = min(content_w + dyn_pad * 2 + 4, avail)
                    new_frame_w = max(new_frame_w, self._bubble_min_width_px)
                    body_w = max(40, new_frame_w - dyn_pad * 2)
                    body.setFixedWidth(body_w)
                    frame.setFixedWidth(new_frame_w)
                    self._fit_text_browser(body)
        except Exception:
            return

    def _add_chat_widget(self, w: QWidget) -> None:
        idx = max(0, self._chat_layout.count() - 1)
        if w.property("bubbleAlign") == "right":
            self._chat_layout.insertWidget(idx, w, 0, Qt.AlignRight)
        else:
            self._chat_layout.insertWidget(idx, w)
        self._chat_bubbles.append(w)

        # 超出上限时移除最早的消息（防止内存无限增长导致卡死）
        while len(self._chat_bubbles) > self._MAX_CHAT_MESSAGES:
            old = self._chat_bubbles.pop(0)
            self._chat_layout.removeWidget(old)
            old.setParent(None)
            old.deleteLater()

        self._schedule_refit_chat_bubbles()
        QTimer.singleShot(10, self._scroll_chat_to_bottom)

    def _make_bubble(
        self, *, align_right: bool, title: str, html_body: str, plain_text: str,
        bg: str, border: str, title_color: str, text_color: str,
        show_title: bool = True, pad_x: int | None = None,
        body_font_px: int | None = None,
    ) -> QWidget:
        pad_m = self._bubble_pad_x if pad_x is None else pad_x
        bfs = self._bubble_body_font_px if body_font_px is None else body_font_px
        rp = self._responsive_params()
        avail = int(rp["avail"])

        outer = QWidget()
        outer.setProperty("bubbleAlign", "right" if align_right else "left")
        row = QHBoxLayout(outer)
        row.setContentsMargins(0, 2, 0, 2)
        row.setSpacing(4)

        # 头像
        avatar = QLabel("暖" if not align_right else "我")
        avatar.setFixedSize(38, 38)
        avatar.setAlignment(Qt.AlignCenter)
        if not align_right:
            avatar_color = "qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #0d6b64,stop:1 #14a89a)"
            avatar.setStyleSheet(
                "background:qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #0d6b64,stop:1 #14a89a);"
                "color:#fff; border-radius:19px; font-size:15px; font-weight:800;"
            )
        else:
            avatar.setStyleSheet(
                "background:qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #5a7a72,stop:1 #7a9a92);"
                "color:#fff; border-radius:19px; font-size:15px; font-weight:800;"
            )

        frame = QFrame()
        frame.setObjectName("chatBubbleFrame")
        if align_right:
            frame.setStyleSheet(
                f"QFrame#chatBubbleFrame{{background:{bg}; border:1px solid {border};"
                f"border-radius:{self._bubble_radius + 2}px; padding:2px;}}"
            )
        else:
            frame.setStyleSheet(
                f"QFrame#chatBubbleFrame{{background:{bg}; border:1px solid {border};"
                f"border-radius:{self._bubble_radius + 2}px;}}"
            )
        frame.setProperty("plainText", plain_text or "")
        frame.setProperty("bubbleTitle", (title or "") if show_title else "")

        frame_layout = QVBoxLayout(frame)
        frame_layout.setSpacing(4)

        char_count = len((plain_text or "").strip())
        dyn_pad = max(6, pad_m // 2) if char_count <= 2 else (max(8, pad_m - 4) if char_count <= 6 else pad_m)
        frame_layout.setContentsMargins(dyn_pad, dyn_pad - 1, dyn_pad, dyn_pad - 1)

        title_lbl = None
        if show_title and title:
            title_lbl = QLabel(title)
            title_lbl.setWordWrap(True)
            title_lbl.setStyleSheet(f"color:{title_color}; font-size:{self._bubble_title_font_px}px; font-weight:700;")
            title_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)

        body = QTextBrowser()
        body.setObjectName("chatBubbleBody")
        bf = QFont("Microsoft YaHei UI", bfs)
        if not bf.exactMatch():
            bf = QFont("Microsoft YaHei", bfs)
        body.setFont(bf)
        body.setOpenExternalLinks(False)
        body.setOpenLinks(False)
        body.setStyleSheet(
            f"QTextBrowser#chatBubbleBody{{background:transparent; border:none; padding:0px; margin:0px; font-size:{bfs}px;}}"
        )
        body.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body.setFocusPolicy(Qt.NoFocus)
        body.setLineWrapMode(QTextEdit.WidgetWidth)
        body.setTextInteractionFlags(Qt.NoTextInteraction)
        body.installEventFilter(self)
        body.viewport().installEventFilter(self)

        fm_body = QFontMetrics(bf)
        lines = (plain_text or "").splitlines() or [""]
        longest = max((ln.strip() for ln in lines), key=len, default="")
        text_natural_w = fm_body.horizontalAdvance(longest)
        title_natural_w = 0
        if show_title and title:
            fm_title = QFontMetrics(QFont(bf.family(), self._bubble_title_font_px))
            title_natural_w = fm_title.horizontalAdvance(title or "")
        content_natural_w = max(text_natural_w, title_natural_w)
        frame_width = min(content_natural_w + dyn_pad * 2 + 4, avail)
        frame_width = max(frame_width, self._bubble_min_width_px)
        body_w = max(40, frame_width - dyn_pad * 2)

        body.setFixedWidth(body_w)
        body.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        body.setHtml(
            f"<div style=\"font-family:'Microsoft YaHei UI','Microsoft YaHei',sans-serif;"
            f"font-size:{bfs}px; line-height:1.45; color:{text_color}; word-wrap:break-word; word-break:break-all;\">"
            f"{html_body}</div>"
        )

        opt = QTextOption(Qt.AlignLeft)
        opt.setWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        body.document().setDefaultTextOption(opt)
        frame.setFixedWidth(frame_width)
        QTimer.singleShot(0, lambda tb=body: self._fit_text_browser(tb))

        if title_lbl is not None:
            frame_layout.addWidget(title_lbl)
        frame_layout.addWidget(body, 0, Qt.AlignLeft)

        # 时间戳
        ts = QLabel(time.strftime("%H:%M"))
        if align_right:
            ts.setStyleSheet(f"color:rgba(240,255,253,0.5); font-size:10px; margin-top:2px; font-weight:500;")
        else:
            if self._dark_mode:
                ts.setStyleSheet(f"color:rgba(122,184,168,0.4); font-size:10px; margin-top:2px; font-weight:500;")
            else:
                ts.setStyleSheet(f"color:rgba(90,138,126,0.35); font-size:10px; margin-top:2px; font-weight:500;")

        if align_right:
            col = QVBoxLayout()
            col.setSpacing(2)
            col.addWidget(frame, 0, Qt.AlignRight)
            col.addWidget(ts, 0, Qt.AlignRight)
            row.addStretch(1)
            row.addLayout(col)
            row.addWidget(avatar, 0, Qt.AlignTop)
        else:
            row.addWidget(avatar, 0, Qt.AlignTop)
            col = QVBoxLayout()
            col.setSpacing(2)
            col.addWidget(frame, 0, Qt.AlignLeft)
            col.addWidget(ts, 0, Qt.AlignLeft)
            row.addLayout(col)
            row.addStretch(1)

        outer.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        return outer

    def _make_related_item(self, item: dict[str, Any]) -> QWidget:
        title = str(item.get("title", "") or "").strip()
        time_text = str(item.get("time", "") or "").strip()
        summary = str(item.get("summary", "") or item.get("body_snippet", "") or "").strip()
        file_path = str(item.get("file_path", "") or "").strip()

        frame = QFrame()
        frame.setObjectName("relatedItemFrame")
        if self._dark_mode:
            frame.setStyleSheet(
                f"QFrame#relatedItemFrame{{background:rgba(22,38,35,0.85); border:1px solid rgba(20,168,154,0.08);"
                f"border-radius:{self._bubble_radius + 2}px;}}"
                f"QFrame#relatedItemFrame:hover{{border-color:rgba(20,168,154,0.2);}}"
            )
        else:
            frame.setStyleSheet(
                f"QFrame#relatedItemFrame{{background:rgba(255,255,255,0.9); border:1px solid rgba(15,118,110,0.06);"
                f"border-radius:{self._bubble_radius + 2}px;}}"
                f"QFrame#relatedItemFrame:hover{{border-color:rgba(15,118,110,0.15);}}"
            )
        frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        pad_m = self._bubble_pad_x
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(pad_m, pad_m - 2, pad_m, pad_m - 2)
        lay.setSpacing(4)

        header = QLabel(title or "（未命名）")
        header.setWordWrap(True)
        if self._dark_mode:
            header.setStyleSheet("font-weight:700; color:#c8e0d8; font-size:14px;")
        else:
            header.setStyleSheet("font-weight:700; color:#1a3330; font-size:14px;")
        header.setTextInteractionFlags(Qt.TextSelectableByMouse)

        meta = QLabel(time_text)
        if self._dark_mode:
            meta.setStyleSheet("color:#5a9a8a; font-size:11px; font-weight:600;")
        else:
            meta.setStyleSheet("color:#7a9a8e; font-size:11px; font-weight:600;")
        meta.setTextInteractionFlags(Qt.TextSelectableByMouse)
        meta.setVisible(bool(time_text))

        body = QLabel(summary)
        body.setWordWrap(True)
        if self._dark_mode:
            body.setStyleSheet("color:#9ab8a8; font-size:13px;")
        else:
            body.setStyleSheet("color:#4a6a62; font-size:13px;")
        body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        body.setVisible(bool(summary))

        lay.addWidget(header)
        if time_text:
            lay.addWidget(meta)
        lay.addWidget(body)

        # 图片预览（可点击放大）
        if file_path:
            try:
                p = Path(file_path)
                suf = p.suffix.lower()
                if suf in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"} and p.exists():
                    pix = QPixmap(str(p))
                    if not pix.isNull():
                        img = QLabel()
                        img.setObjectName("relatedPreviewImage")
                        img.setFixedSize(self._preview_base_w, self._preview_base_h)
                        img.setStyleSheet(f"border-radius:{self._bubble_radius}px; border:1px solid #c5d6ed;")
                        img.setScaledContents(True)
                        img.setPixmap(pix.scaled(self._preview_base_w, self._preview_base_h, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation))
                        img.setCursor(Qt.PointingHandCursor)
                        img.mousePressEvent = lambda e, p=pix: self._show_image_zoom(p)
                        lay.addWidget(img, 0, Qt.AlignLeft)
            except Exception:
                pass

        # 按钮区
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        copy_btn = QPushButton("复制记忆")
        copy_btn.setObjectName("copyMemBtn")
        copy_btn.setCursor(Qt.PointingHandCursor)
        copy_btn.setFixedHeight(36)
        if self._dark_mode:
            copy_btn.setStyleSheet(
                "QPushButton#copyMemBtn{background:rgba(15,50,44,0.9); color:#14a89a; border:1px solid rgba(20,168,154,0.15);"
                "border-radius:18px; padding:6px 16px; font-weight:600; font-size:12px;}"
                "QPushButton#copyMemBtn:hover{background:rgba(20,60,52,0.95); border-color:rgba(20,168,154,0.3);}"
            )
        else:
            copy_btn.setStyleSheet(
                "QPushButton#copyMemBtn{background:rgba(232,250,247,0.9); color:#0f766e; border:1px solid rgba(15,118,110,0.12);"
                "border-radius:18px; padding:6px 16px; font-weight:600; font-size:12px;}"
                "QPushButton#copyMemBtn:hover{background:rgba(210,245,238,0.95); border-color:rgba(15,118,110,0.25);}"
            )
        mem_text = f"【{title}】\n{summary}\n\n时间: {time_text}"
        copy_btn.clicked.connect(lambda _, t=mem_text: QApplication.clipboard().setText(t))
        btn_row.addWidget(copy_btn)

        if file_path:
            open_btn = QPushButton("打开文件")
            open_btn.setObjectName("openFileBtn")
            open_btn.setCursor(Qt.PointingHandCursor)
            open_btn.setFixedHeight(36)
            if self._dark_mode:
                open_btn.setStyleSheet(
                    "QPushButton#openFileBtn{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #0f766e,stop:1 #14a89a);"
                    "color:#fff; border:none; border-radius:18px; padding:6px 16px; font-weight:600; font-size:12px;}"
                    "QPushButton#openFileBtn:hover{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #14a89a,stop:1 #20d4b8);}"
                )
            else:
                open_btn.setStyleSheet(
                    "QPushButton#openFileBtn{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #0d6b64,stop:1 #0f766e);"
                    "color:#fff; border:none; border-radius:18px; padding:6px 16px; font-weight:600; font-size:12px;}"
                    "QPushButton#openFileBtn:hover{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #0a5c56,stop:1 #0d6b64);}"
                )
            open_btn.clicked.connect(lambda _, fp=file_path: self._open_file(fp))
            btn_row.addWidget(open_btn)

        btn_row.addStretch(1)
        lay.addLayout(btn_row)
        return frame

    def _show_image_zoom(self, pixmap: QPixmap):
        dlg = ImageZoomDialog(pixmap, self)
        dlg.resize(self.size())
        dlg.exec_()

    def _open_file(self, file_path: str):
        p = Path(file_path)
        if p.exists():
            if sys.platform == "win32":
                subprocess.Popen(f'explorer /select,"{p}"')
            else:
                os.startfile(str(p))
        else:
            self._append_assistant("文件不存在，可能已被移动或删除。")

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._schedule_refit_chat_bubbles()

    def _append_user(self, text: str) -> None:
        safe = html.escape(text).replace("\n", "<br>")
        if self._dark_mode:
            w = self._make_bubble(
                align_right=True, title="", html_body=safe, plain_text=text,
                bg="qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #0f766e,stop:0.5 #14a89a,stop:1 #0d6b64)",
                border="rgba(20,168,154,0.3)", title_color="#88dcc8", text_color="#f0fffd",
                show_title=False, pad_x=self._bubble_user_pad_x,
                body_font_px=self._bubble_user_body_font_px,
            )
        else:
            w = self._make_bubble(
                align_right=True, title="", html_body=safe, plain_text=text,
                bg="qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #0d6b64,stop:0.5 #0f766e,stop:1 #14a89a)",
                border="rgba(15,118,110,0.15)", title_color="#88dcc8", text_color="#f0fffd",
                show_title=False, pad_x=self._bubble_user_pad_x,
                body_font_px=self._bubble_user_body_font_px,
            )
        self._add_chat_widget(w)

    def _append_assistant(self, text: str) -> None:
        safe = html.escape(text).replace("\n", "<br>")
        if self._dark_mode:
            w = self._make_bubble(
                align_right=False, title="暖暖", html_body=safe, plain_text=text,
                bg="rgba(22,38,35,0.92)", border="rgba(20,168,154,0.08)",
                title_color="#7ab8a8", text_color="#c8e0d8",
                show_title=True,
            )
        else:
            w = self._make_bubble(
                align_right=False, title="暖暖", html_body=safe, plain_text=text,
                bg="rgba(255,255,255,0.85)", border="rgba(15,118,110,0.06)",
                title_color="#5a8a7e", text_color="#1a3330",
                show_title=True,
            )
        self._add_chat_widget(w)

    def _append_cards(self, items: list[dict[str, Any]]) -> None:
        if not items:
            return
        plain_lines = []
        for it in items[:8]:
            t = str(it.get("title", "") or "").strip()
            tm = str(it.get("time", "") or "").strip()
            if t and tm:
                plain_lines.append(f"- {t}（{tm}）")
            elif t:
                plain_lines.append(f"- {t}")

        container = QWidget()
        container.setObjectName("relatedContainer")
        v = QVBoxLayout(container)
        v.setContentsMargins(0, 2, 0, 2)
        v.setSpacing(6)

        title_lbl = QLabel("相关记忆")
        if self._dark_mode:
            title_lbl.setStyleSheet("color:#e8c878; font-size:12px; font-weight:700; letter-spacing:0.05em;")
        else:
            title_lbl.setStyleSheet("color:#8a6a28; font-size:12px; font-weight:700; letter-spacing:0.05em;")
        title_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        v.addWidget(title_lbl, 0, Qt.AlignLeft)

        for it in items[:8]:
            v.addWidget(self._make_related_item(it))

        container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        outer = QWidget()
        row = QHBoxLayout(outer)
        row.setContentsMargins(0, 4, 0, 4)
        row.setSpacing(6)
        row.addWidget(container, 0, Qt.AlignLeft)
        row.addStretch(1)
        self._add_chat_widget(outer)

    # ---- 后端初始化 ----
    def _init_backend(self) -> None:
        self._client_id = self._settings.value("client/id", "", type=str) or ""
        if not self._client_id:
            self._client_id = uuid.uuid4().hex[:16]
            self._settings.setValue("client/id", self._client_id)

        server_url = self._settings.value("server/url", "", type=str)
        if not server_url or "localhost" in server_url:
            server_url = "https://memory-n.ccwu.cc"
        self._api = MemoryApiClient(base_url=server_url, client_id=self._client_id, timeout=60)
        with open(_LOG_FILE, "a", encoding="utf-8") as _f:
            _f.write(f"Server URL: {server_url}\n")

        # 加载暗色模式偏好
        self._dark_mode = self._settings.value("theme/mode", "light", type=str) == "dark"
        if self._dark_mode:
            self._theme_btn.setText("☀️")
            self._apply_style()

        # 跨线程信号连接
        self._health_result.connect(self._handle_health_result)
        self._chat_result_signal.connect(self._process_chat_result)
        self._chat_error_signal.connect(self._handle_chat_error)
        self._upload_result_signal.connect(self._process_upload_result)
        self._upload_error_signal.connect(lambda e: self._append_assistant(f"文件上传失败了：{e}"))
        self._voice_result_signal.connect(self._process_voice_result)
        self._voice_error_signal.connect(lambda e: self._append_assistant(f"语音处理失败了：{e}"))
        self._update_bar_signal.connect(self._show_update_bar)

        # 健康检查（智能退避：在线时 15s，断线后逐步增长到 60s）
        self._health_fail_count = 0
        self._health_base_interval = 15000  # 15 秒基础间隔
        self._health_timer = QTimer(self)
        self._health_timer.timeout.connect(self._check_health)
        self._health_timer.start(self._health_base_interval)
        self._check_health()

    def _check_health(self):
        if self._api is None:
            return
        def _do():
            ok = self._api.health_fast()
            self._health_result.emit(ok)
        threading.Thread(target=_do, daemon=True).start()

    def _handle_health_result(self, ok: bool) -> None:
        was_online = self._online
        self._online = ok
        self._update_connection_status(ok)
        if ok and not was_online:
            self._set_status("已恢复连接 ✓")
            self._offline_bar.setVisible(False)
            self._health_fail_count = 0
            # 恢复正常检查间隔
            self._health_timer.setInterval(self._health_base_interval)
        elif ok:
            self._health_fail_count = 0
            self._health_timer.setInterval(self._health_base_interval)
        elif not ok:
            self._health_fail_count += 1
            # 逐步增加检查间隔：15s → 30s → 45s → 60s（最大）
            backoff = min(60000, self._health_base_interval * self._health_fail_count)
            self._health_timer.setInterval(backoff)
            if not was_online or self._health_fail_count == 1:
                self._offline_bar.setVisible(True)

    def _update_connection_status(self, online: bool) -> None:
        if online:
            self._conn_status_label.setText("● 已连接")
            self._conn_status_label.setProperty("connected", True)
        else:
            self._conn_status_label.setText("○ 离线")
            self._conn_status_label.setProperty("connected", False)
        # 刷新样式以响应属性变化
        self._conn_status_label.style().unpolish(self._conn_status_label)
        self._conn_status_label.style().polish(self._conn_status_label)

    # ---- 版本检查 ----
    def _init_version_check(self):
        self._check_version()
        self._version_timer = QTimer(self)
        self._version_timer.timeout.connect(self._check_version)
        self._version_timer.start(6 * 3600 * 1000)  # 每 6 小时

    def _check_version(self):
        if self._api is None:
            return
        def _do():
            try:
                info = self._api.get_version()
                remote_ver = info.get("version", "")
                download_url = info.get("download_url", "")
                if remote_ver and remote_ver != APP_VERSION and download_url:
                    self._update_bar_signal.emit(remote_ver, download_url)
            except Exception:
                pass
        threading.Thread(target=_do, daemon=True).start()

    def _show_update_bar(self, version: str, url: str):
        self._update_label.setText(f"发现新版本 v{version}，点击下载 →")
        self._update_label.setProperty("download_url", url)
        self._update_bar.setVisible(True)

    def _open_download_url(self):
        url = self._update_label.property("download_url") or ""
        if url:
            QDesktopServices.openUrl(QUrl(url))

    # ---- 聊天逻辑 ----
    def _submit_command_input(self) -> None:
        self.command_input.submit()

    def _handle_chat_submit_text(self, payload: CommandSubmit) -> None:
        text = (payload.text or "").strip()
        if not text:
            return
        # 防止重复发送
        if self._is_sending:
            return
        self._append_user(text)

        if self._api is None:
            self._append_assistant("后端未初始化，无法处理。")
            self.command_input.clear()
            return

        if not self._online:
            self._append_assistant("当前处于离线状态，无法执行此操作。请检查服务器连接。")
            self.command_input.clear()
            return

        pending_response = self._resolve_pending_save(text)
        if pending_response is not None:
            self._append_assistant(pending_response)
            self.command_input.clear()
            return

        def _do():
            try:
                result = self._api.chat(text)
                self._chat_result_signal.emit(result)
            except Exception as e:
                self._chat_error_signal.emit(str(e))

        self._is_sending = True
        self._show_typing_indicator()
        self._set_status("正在处理…")
        threading.Thread(target=_do, daemon=True).start()
        self.command_input.clear()

    def _handle_chat_error(self, error: str) -> None:
        """根据错误类型显示友好的提示"""
        self._hide_typing_indicator()
        self._is_sending = False
        err_lower = error.lower()
        if "超时" in error or "timeout" in err_lower:
            self._append_assistant("服务器响应有点慢，可能是网络波动，稍等一下再试试？")
        elif "连接" in error or "network" in err_lower or "connection" in err_lower:
            self._append_assistant("网络好像断了，我正在帮你重连。等网络恢复后直接发送就行～")
        elif "ssl" in err_lower or "certificate" in err_lower:
            self._append_assistant("安全连接出了问题，请检查系统时间是否正确。")
        elif "429" in error or "频繁" in error:
            self._append_assistant("操作太频繁了，休息几秒再试吧～")
        else:
            self._append_assistant(f"出了点小问题：{error}")

    def _show_typing_indicator(self) -> None:
        """显示 '正在思考...' 打字指示器"""
        self._hide_typing_indicator()
        w = QWidget()
        w.setProperty("bubbleAlign", "left")
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 2, 0, 2)
        row.setSpacing(4)

        avatar = QLabel("暖")
        avatar.setFixedSize(38, 38)
        avatar.setAlignment(Qt.AlignCenter)
        avatar.setStyleSheet(
            "background:qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #0d6b64,stop:1 #14a89a);"
            "color:#fff; border-radius:19px; font-size:15px; font-weight:800;"
        )

        dots = QLabel("● ● ●")
        dots.setObjectName("typingDots")
        if self._dark_mode:
            dots.setStyleSheet("color:rgba(122,184,168,0.5); font-size:14px; padding:12px 18px; font-weight:700; letter-spacing:4px;")
        else:
            dots.setStyleSheet("color:rgba(90,138,126,0.4); font-size:14px; padding:12px 18px; font-weight:700; letter-spacing:4px;")

        row.addWidget(avatar, 0, Qt.AlignTop)
        row.addWidget(dots)
        row.addStretch(1)

        idx = max(0, self._chat_layout.count() - 1)
        self._chat_layout.insertWidget(idx, w)
        self._typing_widget = w
        QTimer.singleShot(10, self._scroll_chat_to_bottom)

    def _hide_typing_indicator(self) -> None:
        """移除打字指示器"""
        if self._typing_widget is not None:
            self._chat_layout.removeWidget(self._typing_widget)
            self._typing_widget.setParent(None)
            self._typing_widget.deleteLater()
            self._typing_widget = None

    def _process_chat_result(self, result: dict) -> None:
        self._hide_typing_indicator()
        self._is_sending = False
        text = result.get("text", "")
        if text:
            self._append_assistant(text)
        results = result.get("results")
        if results:
            self._append_cards(results)
        pending = result.get("pending_save")
        if pending:
            self._pending_save_text = pending
            self._pending_bar.setVisible(True)
        self._set_status("准备就绪")

    def _resolve_pending_save(self, text: str) -> str | None:
        if self._pending_save_text is None:
            return None
        try:
            result = self._api.confirm_save(text, self._pending_save_text)
            self._pending_save_text = None
            self._pending_bar.setVisible(False)
            return result.get("text", "好的，已处理。")
        except Exception as e:
            self._pending_save_text = None
            self._pending_bar.setVisible(False)
            return f"处理失败：{e}"

    def _handle_chat_submit_drop(self, payload: DropSubmit) -> None:
        paths = payload.paths or []
        caption = (payload.caption or "").strip()
        shown = caption or ("拖拽附件：" + "；".join(Path(p).name for p in paths[:3]) + ("…" if len(paths) > 3 else ""))
        self._append_user(shown)
        if self._api is None:
            self._append_assistant("后端未初始化，无法保存附件。")
            self.command_input.clear()
            return
        if not self._online:
            self._append_assistant("当前处于离线状态，无法上传文件。")
            self.command_input.clear()
            return

        def _do():
            try:
                result = self._api.upload(paths, caption)
                self._upload_result_signal.emit(result)
            except Exception as e:
                self._upload_error_signal.emit(str(e))

        self._set_status("正在上传文件…")
        threading.Thread(target=_do, daemon=True).start()
        self.command_input.clear()

    def _process_upload_result(self, result: dict) -> None:
        if result.get("success"):
            self._append_assistant(result.get("message", "文件已保存。"))
        elif result.get("error"):
            self._append_assistant(f"文件上传失败：{result['error']}")
        self._set_status("准备就绪")

    def _choose_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "选择文件", "", "所有文件 (*)")
        if paths:
            self._handle_chat_submit_drop(DropSubmit(paths=paths, caption=""))

    # ---- 语音 ----
    def _toggle_voice_dialogue(self) -> None:
        if hasattr(self, "voice_worker") and self.voice_worker is not None:
            self.voice_worker.stop()
            return
        if not self._online:
            self._append_assistant("当前处于离线状态，无法使用语音功能。")
            return
        self._append_assistant("启动语音对话模式，说'退出'可结束对话。")
        self._start_voice_dialogue()

    def _start_voice_dialogue(self) -> None:
        if hasattr(self, "voice_worker") and self.voice_worker is not None:
            return
        self.voice_thread = QThread(self)
        self.voice_worker = VoiceWorker(api_client=self._api, dialogue_mode=True)
        self.voice_worker.moveToThread(self.voice_thread)
        self.voice_thread.started.connect(self.voice_worker.run)
        self.voice_worker.status_changed.connect(self._set_status)
        self.voice_worker.listening_changed.connect(self._update_voice_dialogue_state)
        self.voice_worker.user_text_ready.connect(self._append_user)
        self.voice_worker.voice_text_ready.connect(self._handle_voice_text)
        self.voice_worker.dialogue_stopped.connect(self._handle_dialogue_stopped)
        self.voice_worker.finished.connect(self.voice_thread.quit)
        self.voice_thread.finished.connect(self.voice_worker.deleteLater)
        self.voice_thread.finished.connect(self.voice_thread.deleteLater)
        self.voice_thread.finished.connect(self._cleanup_voice_worker)
        self.voice_thread.start()

    def _handle_voice_text(self, text: str) -> None:
        self._do_voice_text(text)

    def _do_voice_text(self, text: str) -> None:
        if self._api is None or not self._online:
            if self.voice_worker:
                self.voice_worker.speak_text("服务不可用，请稍后再试。")
            return

        if any(w in text for w in ["算了", "不要了", "取消", "不存了"]):
            if self._pending_save_text is not None:
                self._pending_save_text = None
                self._pending_bar.setVisible(False)
                response = "好的，已取消存储。"
                self._append_assistant(response)
                if self.voice_worker:
                    self.voice_worker.speak_text(response)
                return

        pending_response = self._resolve_pending_save(text)
        if pending_response is not None:
            self._append_assistant(pending_response)
            if self.voice_worker:
                self.voice_worker.speak_text(pending_response)
            return

        def _do():
            try:
                result = self._api.chat(text)
                response = result.get("text", "")
                results = result.get("results")
                pending = result.get("pending_save")
                self._voice_result_signal.emit(response, results, pending)
            except Exception as e:
                self._voice_error_signal.emit(str(e))

        threading.Thread(target=_do, daemon=True).start()

    def _process_voice_result(self, response: str, results: list | None, pending_save: str | None) -> None:
        if response:
            self._append_assistant(response)
            if hasattr(self, 'voice_worker') and self.voice_worker:
                self.voice_worker.speak_text(response)
        if results:
            self._append_cards(results)
        if pending_save:
            self._pending_save_text = pending_save
            self._pending_bar.setVisible(True)

    def _update_voice_dialogue_state(self, listening: bool) -> None:
        if hasattr(self, "voice_btn"):
            if listening:
                self.voice_btn.setText("■")
                self.voice_btn.setProperty("voiceState", "recording")
            else:
                self.voice_btn.setText("🎤")
                self.voice_btn.setProperty("voiceState", "idle")
            # 刷新样式以响应属性变化
            self.voice_btn.style().unpolish(self.voice_btn)
            self.voice_btn.style().polish(self.voice_btn)

    def _handle_dialogue_stopped(self) -> None:
        if hasattr(self, "voice_btn"):
            self.voice_btn.setText("🎤")
            self.voice_btn.setProperty("voiceState", "idle")
            self.voice_btn.style().unpolish(self.voice_btn)
            self.voice_btn.style().polish(self.voice_btn)
        self._set_status("语音对话已结束")

    def _cleanup_voice_worker(self) -> None:
        if hasattr(self, "voice_thread"):
            del self.voice_thread
        if hasattr(self, "voice_worker"):
            del self.voice_worker

    # ---- 暗色模式 ----
    def _toggle_theme(self):
        self._dark_mode = not self._dark_mode
        self._settings.setValue("theme/mode", "dark" if self._dark_mode else "light")
        self._theme_btn.setText("☀️" if self._dark_mode else "🌙")
        self._apply_style()

    # ---- 统计面板 ----
    def _show_stats(self):
        if self._api is None:
            return
        dlg = StatsDialog(self._api, self, dark_mode=self._dark_mode)
        dlg.exec_()

    # ---- 右键菜单 ----
    def _show_chat_context_menu(self, position) -> None:
        menu = QMenu(self)
        clear_action = menu.addAction("清空对话")
        clear_action.triggered.connect(self._clear_query)
        menu.exec_(self.chat_view.viewport().mapToGlobal(position))

    # ---- 其他 ----
    def _load_defaults(self) -> None:
        last_query = self._settings.value("query/last", "", type=str) or ""
        if last_query:
            self.command_input.set_text(last_query)

    def _set_status(self, text: str, is_error: bool = False) -> None:
        self.statusBar().showMessage(text, 8000 if not is_error else 15000)

    def _clear_query(self) -> None:
        self.command_input.clear()
        self._hide_typing_indicator()
        self._is_sending = False
        # 使用跟踪列表清空，比遍历 layout 更高效
        for w in self._chat_bubbles:
            self._chat_layout.removeWidget(w)
            w.setParent(None)
            w.deleteLater()
        self._chat_bubbles.clear()
        # 也清理可能遗漏的子控件
        for i in range(self._chat_layout.count() - 2, -1, -1):
            item = self._chat_layout.takeAt(i)
            if item is None:
                continue
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._append_assistant("内容已清空。")
        self._set_status("内容已清空。")
        self._settings.setValue("query/last", "")

    def _restore_window_state(self) -> None:
        geo = self._settings.value("window/geometry")
        state = self._settings.value("window/state")
        if geo is not None:
            self.restoreGeometry(geo)
        if state is not None:
            self.restoreState(state)

    def closeEvent(self, event) -> None:
        try:
            self._settings.setValue("query/last", self.command_input.text().strip())
            self._settings.setValue("window/geometry", self.saveGeometry())
            self._settings.setValue("window/state", self.saveState())
        finally:
            if self._health_timer:
                self._health_timer.stop()
            if self._version_timer:
                self._version_timer.stop()
            super().closeEvent(event)

    # ---- 样式 ----
    def _apply_style(self) -> None:
        bfs = self._bubble_body_font_px
        if self._dark_mode:
            css = self._dark_theme_css(bfs)
        else:
            css = self._light_theme_css(bfs)
        self.setStyleSheet(css)
        # 刷新样式
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def _light_theme_css(self, bfs: int) -> str:
        return f"""
            * {{ outline: none; }}
            QWidget {{
                font-family:"Microsoft YaHei UI","Microsoft YaHei","PingFang SC","SF Pro Display",sans-serif;
                font-size:15px; color:#1a2726;
            }}
            QMainWindow {{
                background:qlineargradient(x1:0,y1:0,x2:0.3,y2:1,stop:0 #f0f5f3,stop:0.5 #e8f0ec,stop:1 #dfe8e4);
            }}

            /* ---- 标题栏 ---- */
            QWidget#titleBar {{
                background:rgba(255,255,255,0.72);
                border-bottom:1px solid rgba(15,118,110,0.08);
            }}
            QPushButton#titleBtn {{
                background:transparent; border:none; color:#6b8a84; font-size:13px;
                border-radius:8px; padding:4px 8px;
            }}
            QPushButton#titleBtn:hover {{ background:rgba(15,118,110,0.08); color:#0f766e; }}

            /* ---- 头部 ---- */
            QWidget#headerWidget {{
                background:qlineargradient(x1:0,y1:0,x2:1,y2:0.3,
                    stop:0 rgba(255,255,255,0.85), stop:0.4 rgba(236,250,247,0.9),
                    stop:0.7 rgba(220,245,240,0.85), stop:1 rgba(200,238,232,0.8));
                border:none; border-bottom:1px solid rgba(15,118,110,0.06);
                border-bottom-left-radius:24px; border-bottom-right-radius:24px;
            }}
            QLabel#headerBadge {{
                background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #fff9f0,stop:1 #fff3e0);
                color:#b8860b; border:1px solid rgba(218,165,32,0.25);
                border-radius:20px; padding:4px 14px; font-size:10px;
                font-weight:700; letter-spacing:0.18em;
            }}
            QLabel#headerTitle {{
                font-size:36px; font-weight:900;
                color:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #0a5c56,stop:1 #0f766e);
                letter-spacing:-0.02em;
            }}
            QLabel#headerSubtitle {{
                font-size:15px; color:#4a7068; font-weight:500; letter-spacing:0.01em;
            }}

            /* ---- 通知条 ---- */
            QWidget#updateBar {{
                background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #fffbf0,stop:1 #fff7e0);
                border-bottom:1px solid rgba(218,165,32,0.15);
            }}
            QWidget#offlineBar {{
                background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #fff5f2,stop:1 #ffece6);
                border-bottom:1px solid rgba(182,73,29,0.12);
            }}

            /* ---- 聊天区域 ---- */
            QScrollArea#chatView {{
                background:qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #f8fcfb, stop:0.3 #f4faf8, stop:0.7 #f0f7f4, stop:1 #ecf4f0);
                border:1px solid rgba(15,118,110,0.06);
                border-radius:20px; padding:4px; font-size:15px;
            }}
            QScrollArea#chatView QWidget#chatInner {{ background:transparent; }}
            QScrollArea#chatView QWidget {{ background:transparent; }}

            /* ---- 输入区域 ---- */
            QWidget#inputWidget {{
                background:rgba(255,255,255,0.88);
                border:1px solid rgba(15,118,110,0.08);
                border-top:1px solid rgba(15,118,110,0.05);
                border-radius:20px;
            }}
            QLineEdit {{
                border:none; background:transparent; border-radius:20px;
                padding:12px 16px; font-size:15px; color:#1a2726;
            }}
            QLineEdit::placeholder {{ color:#9ab5ae; }}
            QLineEdit:focus {{ background:rgba(15,118,110,0.02); }}

            /* ---- 按钮通用 ---- */
            QPushButton {{ border:none; font-weight:600; font-size:14px; }}
            QPushButton:disabled {{ color:#c5d0cc !important; }}

            /* ---- 发送按钮 ---- */
            QPushButton#sendBtn {{
                background:qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #0d6b64,stop:0.5 #0f766e,stop:1 #14a89a);
                color:#fff; border-radius:26px; font-size:20px; font-weight:800;
                padding:2px 0 0 2px;
            }}
            QPushButton#sendBtn:hover {{
                background:qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #0a5c56,stop:0.5 #0d6b64,stop:1 #11968a);
            }}
            QPushButton#sendBtn:pressed {{ background:#0a5c56; }}
            QPushButton#sendBtn:disabled {{ background:#c8d8d4 !important; }}

            /* ---- 语音按钮 ---- */
            QPushButton#voiceBtn {{
                background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #e8faf7,stop:1 #d5f5f0);
                color:#0f766e; border:1.5px solid rgba(15,118,110,0.15);
                border-radius:22px; font-size:19px;
            }}
            QPushButton#voiceBtn:hover {{
                background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #d5f5f0,stop:1 #b8ece4);
                border-color:rgba(15,118,110,0.3);
            }}
            QPushButton#voiceBtn[voiceState="recording"] {{
                background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #ffe6d6,stop:1 #fdd5c0);
                color:#b6491d; border:1.5px solid rgba(182,73,29,0.25);
                font-weight:800; font-size:18px;
            }}

            /* ---- 附件按钮 ---- */
            QPushButton#attachBtn {{
                background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #f0f7f6,stop:1 #e4f2ef);
                color:#5a8a82; border:1.5px solid rgba(15,118,110,0.1);
                border-radius:22px; font-size:17px;
            }}
            QPushButton#attachBtn:hover {{
                color:#0f766e; border-color:rgba(15,118,110,0.25);
                background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #e0f5f2,stop:1 #d0efe9);
            }}

            /* ---- 头部按钮 ---- */
            QPushButton#headerThemeBtn, QPushButton#headerStatsBtn {{
                background:rgba(255,255,255,0.7); border:1px solid rgba(15,118,110,0.08);
                border-radius:20px; font-size:15px;
            }}
            QPushButton#headerThemeBtn:hover, QPushButton#headerStatsBtn:hover {{
                background:rgba(255,255,255,0.95); border-color:rgba(15,118,110,0.15);
            }}
            QPushButton#headerClearBtn {{
                background:rgba(255,247,235,0.8); border:1px solid rgba(236,216,184,0.5);
                border-radius:20px; color:#8a6a38; font-size:13px; font-weight:600;
                padding:7px 16px;
            }}
            QPushButton#headerClearBtn:hover {{
                color:#b84a2a; border-color:rgba(215,183,133,0.7);
                background:rgba(255,253,247,0.95);
            }}

            /* ---- 状态栏 ---- */
            QStatusBar {{
                background:rgba(255,255,255,0.6);
                border:none; border-top:1px solid rgba(15,118,110,0.05);
                border-top-left-radius:20px; border-top-right-radius:20px;
                color:#7a9a92; font-size:11px; padding:5px 20px; margin:0 2px;
            }}
            QStatusBar::item {{ border:none; }}
            QProgressBar {{
                border:none; border-radius:6px;
                background:rgba(15,118,110,0.06); height:4px; max-width:180px;
            }}
            QProgressBar::chunk {{ background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #0f766e,stop:1 #14a89a); border-radius:4px; }}

            /* ---- 滚动条 ---- */
            QScrollBar:vertical {{ background:transparent; width:5px; margin:4px 2px; }}
            QScrollBar::handle:vertical {{
                background:rgba(15,118,110,0.12); border-radius:4px;
                min-height:40px; margin:2px;
            }}
            QScrollBar::handle:vertical:hover {{ background:rgba(15,118,110,0.25); }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background:transparent; }}

            QTextBrowser#chatBubbleBody {{ font-size:{bfs}px; }}

            /* ---- 附件 badge ---- */
            QLabel#attachBadge {{
                background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #e8faf7,stop:1 #d5f5f0);
                color:#0f766e; border:1px solid rgba(15,118,110,0.15);
                border-radius:14px; padding:3px 10px; font-size:11px; font-weight:700;
            }}
            QPushButton#deleteAttachBtn {{
                background:rgba(182,73,29,0.08); color:#b6491d;
                border:none; border-radius:13px; font-size:14px; font-weight:700;
            }}
            QPushButton#deleteAttachBtn:hover {{ background:rgba(182,73,29,0.15); }}

            /* ---- 右键菜单 ---- */
            QMenu {{
                background:rgba(255,255,255,0.95);
                border:1px solid rgba(15,118,110,0.1);
                border-radius:12px;
                padding:6px 0px;
                font-size:13px;
            }}
            QMenu::item {{
                padding:8px 28px 8px 18px;
                color:#1a3330;
                border-radius:8px;
                margin:2px 6px;
            }}
            QMenu::item:selected {{
                background:rgba(15,118,110,0.08);
                color:#0f766e;
            }}
            QMenu::separator {{
                height:1px;
                background:rgba(15,118,110,0.06);
                margin:4px 12px;
            }}

            /* ---- 标题栏内部元素 ---- */
            QLabel#titleBarLabel {{ font-size:15px; font-weight:800; color:#0a5c56; letter-spacing:0.02em; }}
            QLabel#connStatusLabel {{ color:#b6491d; font-size:11px; font-weight:600; }}
            QLabel#connStatusLabel[connected="true"] {{ color:#0f766e; }}

            /* ---- 通知条内部元素 ---- */
            QLabel#updateLabel {{ color:#8a6a28; font-size:13px; font-weight:600; }}
            QPushButton#updateCloseBtn {{ background:transparent; border:none; color:#8a6a28; font-size:14px; border-radius:8px; padding:2px; }}
            QPushButton#updateCloseBtn:hover {{ color:#b6491d; background:rgba(182,73,29,0.06); }}
            QLabel#offlineLabel {{ color:#b6491d; font-size:13px; font-weight:600; }}
            QLabel#pendingLabel {{ color:#8a6a28; font-size:13px; font-weight:600; }}
        """

    def _dark_theme_css(self, bfs: int) -> str:
        return f"""
            * {{ outline: none; }}
            QWidget {{
                font-family:"Microsoft YaHei UI","Microsoft YaHei","PingFang SC","SF Pro Display",sans-serif;
                font-size:15px; color:#d8e8e4;
            }}
            QMainWindow {{
                background:qlineargradient(x1:0,y1:0,x2:0.3,y2:1,
                    stop:0 #0f1f1d, stop:0.5 #122825, stop:1 #0a1a18);
            }}

            /* ---- 标题栏 ---- */
            QWidget#titleBar {{
                background:rgba(13,38,34,0.92);
                border-bottom:1px solid rgba(20,168,154,0.08);
            }}
            QPushButton#titleBtn {{
                background:transparent; border:none; color:#6a9a92; font-size:13px;
                border-radius:8px; padding:4px 8px;
            }}
            QPushButton#titleBtn:hover {{ background:rgba(20,168,154,0.1); color:#14a89a; }}

            /* ---- 头部 ---- */
            QWidget#headerWidget {{
                background:qlineargradient(x1:0,y1:0,x2:1,y2:0.3,
                    stop:0 rgba(13,38,34,0.95), stop:0.4 rgba(18,50,46,0.95),
                    stop:0.7 rgba(20,58,52,0.92), stop:1 rgba(15,45,40,0.9));
                border:none; border-bottom:1px solid rgba(20,168,154,0.06);
                border-bottom-left-radius:24px; border-bottom-right-radius:24px;
            }}
            QLabel#headerBadge {{
                background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 rgba(20,61,56,0.9),stop:1 rgba(30,70,60,0.9));
                color:#e8c878; border:1px solid rgba(212,162,89,0.2);
                border-radius:20px; padding:4px 14px; font-size:10px;
                font-weight:700; letter-spacing:0.18em;
            }}
            QLabel#headerTitle {{
                font-size:36px; font-weight:900;
                color:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #6ee7c0,stop:1 #a8f0dc);
                letter-spacing:-0.02em;
            }}
            QLabel#headerSubtitle {{
                font-size:15px; color:#7ab8a8; font-weight:500; letter-spacing:0.01em;
            }}

            /* ---- 通知条 ---- */
            QWidget#updateBar {{
                background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 rgba(42,58,32,0.9),stop:1 rgba(50,65,35,0.9));
                border-bottom:1px solid rgba(74,90,48,0.3);
            }}
            QWidget#offlineBar {{
                background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 rgba(58,32,32,0.9),stop:1 rgba(65,35,35,0.9));
                border-bottom:1px solid rgba(90,48,48,0.3);
            }}

            /* ---- 聊天区域 ---- */
            QScrollArea#chatView {{
                background:qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #162623, stop:0.3 #132220, stop:0.7 #101e1c, stop:1 #0d1a18);
                border:1px solid rgba(20,168,154,0.06);
                border-radius:20px; padding:4px; font-size:15px;
            }}
            QScrollArea#chatView QWidget#chatInner {{ background:transparent; }}
            QScrollArea#chatView QWidget {{ background:transparent; }}

            /* ---- 输入区域 ---- */
            QWidget#inputWidget {{
                background:rgba(15,40,36,0.92);
                border:1px solid rgba(20,168,154,0.08);
                border-top:1px solid rgba(20,168,154,0.04);
                border-radius:20px;
            }}
            QLineEdit {{
                border:none; background:transparent; border-radius:20px;
                padding:12px 16px; font-size:15px; color:#d8e8e4;
            }}
            QLineEdit::placeholder {{ color:#4a7a70; }}
            QLineEdit:focus {{ background:rgba(20,168,154,0.03); }}

            /* ---- 按钮通用 ---- */
            QPushButton {{ border:none; font-weight:600; font-size:14px; }}
            QPushButton:disabled {{ color:#3a5a52 !important; }}

            /* ---- 发送按钮 ---- */
            QPushButton#sendBtn {{
                background:qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #0f766e,stop:0.5 #14a89a,stop:1 #20d4b8);
                color:#fff; border-radius:26px; font-size:20px; font-weight:800;
                padding:2px 0 0 2px;
            }}
            QPushButton#sendBtn:hover {{
                background:qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #14a89a,stop:0.5 #20d4b8,stop:1 #28e8c8);
            }}
            QPushButton#sendBtn:pressed {{ background:#0d6b64; }}
            QPushButton#sendBtn:disabled {{ background:#1a3a35 !important; }}

            /* ---- 语音按钮 ---- */
            QPushButton#voiceBtn {{
                background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 rgba(15,40,36,0.95),stop:1 rgba(20,50,44,0.95));
                color:#14a89a; border:1.5px solid rgba(20,168,154,0.15);
                border-radius:22px; font-size:19px;
            }}
            QPushButton#voiceBtn:hover {{
                background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 rgba(20,50,44,0.95),stop:1 rgba(25,60,52,0.95));
                border-color:rgba(20,168,154,0.3);
            }}
            QPushButton#voiceBtn[voiceState="recording"] {{
                background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 rgba(100,40,30,0.9),stop:1 rgba(80,30,20,0.9));
                color:#ff9a7a; border:1.5px solid rgba(255,120,80,0.3);
                font-weight:800; font-size:18px;
            }}

            /* ---- 附件按钮 ---- */
            QPushButton#attachBtn {{
                background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 rgba(15,40,36,0.95),stop:1 rgba(20,48,42,0.95));
                color:#7ab8a8; border:1.5px solid rgba(20,168,154,0.1);
                border-radius:22px; font-size:17px;
            }}
            QPushButton#attachBtn:hover {{
                color:#14a89a; border-color:rgba(20,168,154,0.25);
                background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 rgba(20,50,44,0.95),stop:1 rgba(25,58,50,0.95));
            }}

            /* ---- 头部按钮 ---- */
            QPushButton#headerThemeBtn, QPushButton#headerStatsBtn {{
                background:rgba(15,40,36,0.8); border:1px solid rgba(20,168,154,0.08);
                border-radius:20px; font-size:15px;
            }}
            QPushButton#headerThemeBtn:hover, QPushButton#headerStatsBtn:hover {{
                background:rgba(20,50,44,0.95); border-color:rgba(20,168,154,0.15);
            }}
            QPushButton#headerClearBtn {{
                background:rgba(15,40,36,0.8); border:1px solid rgba(20,168,154,0.08);
                color:#e8c878; border-radius:20px; font-size:13px; font-weight:600;
                padding:7px 16px;
            }}
            QPushButton#headerClearBtn:hover {{
                color:#f0d888; border-color:rgba(212,162,89,0.2);
                background:rgba(20,50,44,0.95);
            }}

            /* ---- 状态栏 ---- */
            QStatusBar {{
                background:rgba(12,32,28,0.9);
                border:none; border-top:1px solid rgba(20,168,154,0.05);
                border-top-left-radius:20px; border-top-right-radius:20px;
                color:#5a9a8a; font-size:11px; padding:5px 20px; margin:0 2px;
            }}
            QStatusBar::item {{ border:none; }}
            QProgressBar {{
                border:none; border-radius:6px;
                background:rgba(20,168,154,0.08); height:4px; max-width:180px;
            }}
            QProgressBar::chunk {{ background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #0f766e,stop:1 #14a89a); border-radius:4px; }}

            /* ---- 滚动条 ---- */
            QScrollBar:vertical {{ background:transparent; width:5px; margin:4px 2px; }}
            QScrollBar::handle:vertical {{
                background:rgba(20,168,154,0.12); border-radius:4px;
                min-height:40px; margin:2px;
            }}
            QScrollBar::handle:vertical:hover {{ background:rgba(20,168,154,0.25); }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background:transparent; }}

            QTextBrowser#chatBubbleBody {{ font-size:{bfs}px; }}

            /* ---- 附件 badge ---- */
            QLabel#attachBadge {{
                background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 rgba(15,50,44,0.9),stop:1 rgba(20,58,50,0.9));
                color:#14a89a; border:1px solid rgba(20,168,154,0.15);
                border-radius:14px; padding:3px 10px; font-size:11px; font-weight:700;
            }}
            QPushButton#deleteAttachBtn {{
                background:rgba(182,73,29,0.12); color:#e87a5a;
                border:none; border-radius:13px; font-size:14px; font-weight:700;
            }}
            QPushButton#deleteAttachBtn:hover {{ background:rgba(182,73,29,0.22); }}

            /* ---- 右键菜单 ---- */
            QMenu {{
                background:rgba(18,38,34,0.97);
                border:1px solid rgba(20,168,154,0.1);
                border-radius:12px;
                padding:6px 0px;
                font-size:13px;
            }}
            QMenu::item {{
                padding:8px 28px 8px 18px;
                color:#c8e0d8;
                border-radius:8px;
                margin:2px 6px;
            }}
            QMenu::item:selected {{
                background:rgba(20,168,154,0.12);
                color:#14a89a;
            }}
            QMenu::separator {{
                height:1px;
                background:rgba(20,168,154,0.08);
                margin:4px 12px;
            }}

            /* ---- 标题栏内部元素 ---- */
            QLabel#titleBarLabel {{ font-size:15px; font-weight:800; color:#6ee7c0; letter-spacing:0.02em; }}
            QLabel#connStatusLabel {{ color:#e87a5a; font-size:11px; font-weight:600; }}
            QLabel#connStatusLabel[connected="true"] {{ color:#14a89a; }}

            /* ---- 通知条内部元素 ---- */
            QLabel#updateLabel {{ color:#e8c878; font-size:13px; font-weight:600; }}
            QPushButton#updateCloseBtn {{ background:transparent; border:none; color:#e8c878; font-size:14px; border-radius:8px; padding:2px; }}
            QPushButton#updateCloseBtn:hover {{ color:#f0d888; background:rgba(232,200,120,0.1); }}
            QLabel#offlineLabel {{ color:#e87a5a; font-size:13px; font-weight:600; }}
            QLabel#pendingLabel {{ color:#e8c878; font-size:13px; font-weight:600; }}
        """


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("暖暖")
    app.setFont(QFont("Microsoft YaHei", 11))
    window = AgentWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
