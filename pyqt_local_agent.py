"""
暖暖记忆助手 - 桌面客户端 v6.0
纯在线模式，通过 API 与服务端通信
"""
import html
import io
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, TYPE_CHECKING
import wave

APP_VERSION = "6.0"


def get_documents_path() -> Path:
    if sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders")
            documents = winreg.QueryValueEx(key, "Personal")[0]
            winreg.CloseKey(key)
            return Path(documents)
        except Exception:
            pass
    return Path.home() / "Documents"


def get_default_vault_root() -> Path:
    return get_documents_path() / "记忆助手" / "memory_vault"


# 设置 PortAudio DLL 搜索路径（用于打包后）
if hasattr(sys, '_MEIPASS'):
    _base = sys._MEIPASS
    _dll_dir = os.path.join(_base, '_sounddevice_data', 'portaudio-binaries')
    _internal_dir = _base
    if os.path.exists(_dll_dir):
        os.environ['PATH'] = _dll_dir + os.pathsep + os.environ.get('PATH', '')
    os.environ['PATH'] = _internal_dir + os.pathsep + os.environ.get('PATH', '')

load_dotenv_file = None  # forward declaration

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
        Qt, QThread, QSettings, pyqtSignal, QTimer, QObject,
        QPropertyAnimation, QEasingCurve, QPoint,
    )
    from PyQt5.QtGui import QFont, QFontMetrics, QPixmap, QColor, QIcon, QDesktopServices
    from PyQt5.QtWidgets import (
        QApplication, QFileDialog, QFrame, QHBoxLayout, QLabel,
        QMainWindow, QProgressBar, QPushButton, QScrollArea, QSizePolicy,
        QTextBrowser, QTextEdit, QVBoxLayout, QWidget, QDialog, QGraphicsOpacityEffect,
    )
    from PyQt5.QtCore import QUrl
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

from app.api_client import MemoryApiClient, ApiError

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
        import subprocess
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
    voice_text_ready = pyqtSignal(str, bool)
    voice_speak = pyqtSignal(str)

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
                self.voice_text_ready.emit(text, False)
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

                has_pending = False
                self.voice_text_ready.emit(user_text, has_pending)

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
                import base64
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
    def __init__(self, api_client: MemoryApiClient, parent=None):
        super().__init__(parent)
        self.setWindowTitle("记忆统计")
        self.setMinimumSize(380, 420)
        self._api = api_client
        self._build_ui()
        self._load_stats()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        self._title_lbl = QLabel("记忆统计")
        self._title_lbl.setStyleSheet("font-size:22px; font-weight:800; color:#173834;")
        layout.addWidget(self._title_lbl)

        self._stats_container = QVBoxLayout()
        self._stats_container.setSpacing(12)
        layout.addLayout(self._stats_container)

        self._loading_lbl = QLabel("加载中...")
        self._loading_lbl.setStyleSheet("color:#8a7b6a; font-size:14px;")
        self._stats_container.addWidget(self._loading_lbl)

        layout.addStretch(1)

        close_btn = QPushButton("关闭")
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.setStyleSheet(
            "QPushButton{background:#0f766e; color:#fff; border:none; border-radius:17px; padding:10px 24px; font-weight:600;}"
            "QPushButton:hover{background:#0d6b64;}"
        )
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn, 0, Qt.AlignRight)

    def _load_stats(self):
        def _do():
            try:
                stats = self._api.get_stats()
                QTimer.singleShot(0, lambda: self._render_stats(stats))
            except Exception as e:
                QTimer.singleShot(0, lambda: self._show_error(str(e)))
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
            lbl.setStyleSheet("color:#5c736d; font-size:14px;")
            val = QLabel(value)
            val.setStyleSheet("color:#173834; font-size:16px; font-weight:700;")
            row.addWidget(lbl)
            row.addStretch(1)
            row.addWidget(val)
            self._stats_container.addLayout(row)

        top_tags = stats.get("top_tags", [])
        if top_tags:
            sep = QLabel("热门标签")
            sep.setStyleSheet("color:#7a4b25; font-size:13px; font-weight:700; margin-top:8px;")
            self._stats_container.addWidget(sep)
            tags_layout = QHBoxLayout()
            tags_layout.setSpacing(6)
            for tag, count in top_tags[:8]:
                chip = QLabel(f"{tag} ({count})")
                chip.setStyleSheet(
                    "background:#fff7eb; border:1px solid #ecd8b8; border-radius:12px;"
                    "padding:4px 10px; color:#6c5738; font-size:12px; font-weight:600;"
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
        err.setStyleSheet("color:#b6491d; font-size:14px;")
        self._stats_container.addWidget(err)


# ============ 图片放大对话框 ============

class ImageZoomDialog(QDialog):
    def __init__(self, pixmap: QPixmap, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setStyleSheet("background: rgba(0,0,0,0.85);")
        self._pix = pixmap

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._img_label = QLabel()
        self._img_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._img_label, 1)

        close_btn = QPushButton("✕")
        close_btn.setFixedSize(36, 36)
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.setStyleSheet(
            "QPushButton{background:rgba(255,255,255,0.2); color:#fff; border:none; border-radius:18px; font-size:18px;}"
            "QPushButton:hover{background:rgba(255,255,255,0.4);}"
        )
        close_btn.clicked.connect(self.close)
        close_btn.setParent(self)
        close_btn.move(20, 20)

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
    def __init__(self) -> None:
        super().__init__()
        init_sounddevice()

        self.voice_thread: QThread | None = None
        self.voice_worker: VoiceWorker | None = None
        self._voice_reply_pending = False
        self._settings = QSettings("", "MemoryAssistant")
        self._pending_save_text: str | None = None
        self._chat_plain_parts: list[str] = []
        self._chat_html_parts: list[str] = []
        self._refit_debounce_timer: QTimer | None = None
        self._chat_history: list[dict] = []
        self._api: MemoryApiClient | None = None
        self._client_id: str = ""
        self._online: bool = False
        self._cached_memories: list[dict] = []
        self._health_timer: QTimer | None = None
        self._version_timer: QTimer | None = None
        self._dark_mode: bool = False

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
        title_lbl.setStyleSheet("font-size:16px; font-weight:700; color:#173834;")
        tb_layout.addWidget(title_lbl)
        tb_layout.addStretch(1)

        # 连接状态
        self._conn_status_label = QLabel("○ 离线")
        self._conn_status_label.setStyleSheet("color: #b6491d; font-size: 12px;")
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
        self.clear_btn.setStyleSheet(
            "QPushButton#headerClearBtn{background:#fff7eb; border:1px solid #ecd8b8; border-radius:17px;"
            "padding:8px 16px; color:#6c5738; font-size:14px; font-weight:600;}"
            "QPushButton#headerClearBtn:hover{background:#fffdf7; border-color:#d7b785; color:#a63d2a;}"
        )
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
        self._update_label.setStyleSheet("color:#9a6a2b; font-size:13px; font-weight:600;")
        self._update_label.setCursor(Qt.PointingHandCursor)
        self._update_label.mousePressEvent = lambda e: self._open_download_url()
        update_layout.addWidget(self._update_label, 1)
        close_update = QPushButton("✕")
        close_update.setFixedSize(24, 24)
        close_update.setStyleSheet("QPushButton{border:none; color:#9a6a2b; font-size:14px;} QPushButton:hover{color:#b6491d;}")
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
        self._offline_label.setStyleSheet("color:#b6491d; font-size:13px; font-weight:600;")
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
        pending_lbl.setStyleSheet("color:#9a6a2b; font-size:13px; font-weight:600;")
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
            if etype == 31:
                return True
            if etype in (179, 180):
                return True
            if etype == 6:
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
        self._refit_debounce_timer.start(50)

    def _refit_chat_bubbles(self) -> None:
        try:
            if not hasattr(self, "_chat_inner"):
                return
            rp = self._responsive_params()
            avail = int(rp["avail"])
            preview_w = int(rp["preview_w"])
            preview_h = int(rp["preview_h"])
            pad_m = self._bubble_pad_x

            for frame in self._chat_inner.findChildren(QFrame, "chatBubbleFrame"):
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

            for w in self._chat_inner.findChildren(QWidget, "relatedContainer"):
                for img in w.findChildren(QLabel, "relatedPreviewImage"):
                    img.setFixedSize(preview_w, preview_h)
            for tb in self._chat_inner.findChildren(QTextBrowser):
                self._fit_text_browser(tb)
        except Exception:
            return

    def _add_chat_widget(self, w: QWidget) -> None:
        idx = max(0, self._chat_layout.count() - 1)
        if w.property("bubbleAlign") == "right":
            self._chat_layout.insertWidget(idx, w, 0, Qt.AlignRight)
        else:
            self._chat_layout.insertWidget(idx, w)
        self._schedule_refit_chat_bubbles()
        QTimer.singleShot(10, self._scroll_chat_to_bottom)

    def _make_bubble(
        self, *, align_right: bool, title: str, html_body: str, plain_text: str,
        bg: str, border: str, title_color: str, text_color: str,
        show_title: bool = True, pad_x: int | None = None,
        body_font_px: int | None = None, center_body: bool = False,
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
        avatar.setFixedSize(36, 36)
        avatar.setAlignment(Qt.AlignCenter)
        avatar_color = "#0f766e" if not align_right else "#6c5738"
        avatar.setStyleSheet(
            f"background:{avatar_color}; color:#fff; border-radius:18px; font-size:14px; font-weight:700;"
        )

        frame = QFrame()
        frame.setObjectName("chatBubbleFrame")
        frame.setStyleSheet(
            f"QFrame#chatBubbleFrame{{background:{bg}; border:1px solid {border}; border-radius:{self._bubble_radius}px;}}"
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
        from PyQt5.QtGui import QTextOption
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
        ts.setStyleSheet(f"color:{title_color}; font-size:10px; margin-top:2px;")

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
        frame.setStyleSheet(
            f"QFrame#relatedItemFrame{{background:#fef8f0; border:1px solid #e8ddd0; border-radius:{self._bubble_radius}px;}}"
        )
        frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        pad_m = self._bubble_pad_x
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(pad_m, pad_m - 2, pad_m, pad_m - 2)
        lay.setSpacing(4)

        header = QLabel(title or "（未命名）")
        header.setWordWrap(True)
        header.setStyleSheet("font-weight:700; color:#3d3028; font-size:14px;")
        header.setTextInteractionFlags(Qt.TextSelectableByMouse)

        meta = QLabel(time_text)
        meta.setStyleSheet("color:#8a7b6a; font-size:11px; font-weight:600;")
        meta.setTextInteractionFlags(Qt.TextSelectableByMouse)
        meta.setVisible(bool(time_text))

        body = QLabel(summary)
        body.setWordWrap(True)
        body.setStyleSheet("color:#24322f; font-size:13px;")
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
        copy_btn.setStyleSheet(
            "QPushButton#copyMemBtn{background:#e0f7f4; color:#0f766e; border:1px solid #0f766e; border-radius:18px; padding:6px 16px; font-weight:600; font-size:12px;}"
            "QPushButton#copyMemBtn:hover{background:#b3ece4;}"
        )
        mem_text = f"【{title}】\n{summary}\n\n时间: {time_text}"
        copy_btn.clicked.connect(lambda _, t=mem_text: QApplication.clipboard().setText(t))
        btn_row.addWidget(copy_btn)

        if file_path:
            open_btn = QPushButton("打开文件")
            open_btn.setObjectName("openFileBtn")
            open_btn.setCursor(Qt.PointingHandCursor)
            open_btn.setFixedHeight(36)
            open_btn.setStyleSheet(
                "QPushButton#openFileBtn{background:#0f766e; color:#fff; border:none; border-radius:18px; padding:6px 16px; font-weight:600; font-size:12px;}"
                "QPushButton#openFileBtn:hover{background:#11847c;}"
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
            import subprocess
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
        self._chat_plain_parts.append(f"你：{text}")
        self._chat_html_parts.append(f"<p><b>你</b><br/>{safe}</p>")
        w = self._make_bubble(
            align_right=True, title="", html_body=safe, plain_text=text,
            bg="#0f766e", border="#0d6560", title_color="#c9f1e6", text_color="#f8fffd",
            show_title=False, pad_x=self._bubble_user_pad_x,
            body_font_px=self._bubble_user_body_font_px, center_body=True,
        )
        self._add_chat_widget(w)

    def _append_assistant(self, text: str) -> None:
        safe = html.escape(text).replace("\n", "<br>")
        self._chat_plain_parts.append(f"暖暖：{text}")
        self._chat_html_parts.append(f"<p><b>暖暖</b><br/>{safe}</p>")
        w = self._make_bubble(
            align_right=False, title="暖暖", html_body=safe, plain_text=text,
            bg="#f5f5f5", border="#d8d8d8", title_color="#666666", text_color="#333333",
            show_title=True, center_body=True,
        )
        self._add_chat_widget(w)
        if self._voice_reply_pending:
            self._voice_reply_pending = False
            self._speak_text_async(text)

    def _speak_text_async(self, text: str) -> None:
        speak_text = (text or "").strip()
        if not speak_text or not self._api:
            return
        def _run():
            try:
                result = self._api.speech_synthesize(speak_text)
                audio_b64 = result.get("audio", "")
                if audio_b64:
                    import base64
                    wav_bytes = base64.b64decode(audio_b64)
                    play_wav_bytes(wav_bytes)
            except Exception as e:
                print(f"语音播放失败: {e}")
        threading.Thread(target=_run, daemon=True).start()

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
        if plain_lines:
            self._chat_plain_parts.append("相关记忆：\n" + "\n".join(plain_lines))

        container = QWidget()
        container.setObjectName("relatedContainer")
        v = QVBoxLayout(container)
        v.setContentsMargins(0, 2, 0, 2)
        v.setSpacing(6)

        title_lbl = QLabel("相关记忆")
        title_lbl.setStyleSheet("color:#7a4b25; font-size:12px; font-weight:700;")
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
            import uuid
            self._client_id = uuid.uuid4().hex[:16]
            self._settings.setValue("client/id", self._client_id)

        server_url = self._settings.value("server/url", "", type=str) or "http://localhost:5000"
        self._api = MemoryApiClient(base_url=server_url, client_id=self._client_id, timeout=30)

        # 加载暗色模式偏好
        self._dark_mode = self._settings.value("theme/mode", "light", type=str) == "dark"
        if self._dark_mode:
            self._theme_btn.setText("☀️")
            self._apply_style()

        # 健康检查
        self._health_timer = QTimer(self)
        self._health_timer.timeout.connect(self._check_health)
        self._health_timer.start(10000)
        self._check_health()

    def _check_health(self):
        if self._api is None:
            return
        def _do():
            try:
                result = self._api.health()
                ok = result.get("status") == "ok"
                was_online = self._online
                self._online = ok
                QTimer.singleShot(0, lambda: self._update_connection_status(ok))
                if ok and not was_online:
                    QTimer.singleShot(0, lambda: self._set_status("已恢复连接"))
                    QTimer.singleShot(0, lambda: self._offline_bar.setVisible(False))
                elif not ok:
                    QTimer.singleShot(0, lambda: self._offline_bar.setVisible(True))
            except Exception:
                was_online = self._online
                self._online = False
                QTimer.singleShot(0, lambda: self._update_connection_status(False))
                if was_online:
                    QTimer.singleShot(0, lambda: self._offline_bar.setVisible(True))
        threading.Thread(target=_do, daemon=True).start()

    def _update_connection_status(self, online: bool) -> None:
        if online:
            self._conn_status_label.setText("● 已连接")
            self._conn_status_label.setStyleSheet("color: #0f766e; font-size: 12px;")
        else:
            self._conn_status_label.setText("○ 离线")
            self._conn_status_label.setStyleSheet("color: #b6491d; font-size: 12px;")

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
                    QTimer.singleShot(0, lambda: self._show_update_bar(remote_ver, download_url))
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
                QTimer.singleShot(0, lambda: self._process_chat_result(result))
            except Exception as e:
                QTimer.singleShot(0, lambda: self._append_assistant(f"请求失败：{e}"))
                QTimer.singleShot(0, lambda: self._set_status("准备就绪", True))

        self._set_status("正在处理…")
        threading.Thread(target=_do, daemon=True).start()
        self.command_input.clear()

    def _process_chat_result(self, result: dict) -> None:
        text = result.get("text", "")
        if text:
            self._append_assistant(text)
        results = result.get("results")
        if results:
            self._append_cards(results)
            # 缓存最近记忆
            self._cached_memories = results[:30]
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
                QTimer.singleShot(0, lambda: self._process_upload_result(result))
            except Exception as e:
                QTimer.singleShot(0, lambda: self._append_assistant(f"上传失败：{e}"))
                QTimer.singleShot(0, lambda: self._set_status("准备就绪", True))

        self._set_status("正在上传文件…")
        threading.Thread(target=_do, daemon=True).start()
        self.command_input.clear()

    def _process_upload_result(self, result: dict) -> None:
        if result.get("success"):
            self._append_assistant(result.get("message", "文件已保存。"))
        elif result.get("error"):
            self._append_assistant(f"上传失败：{result['error']}")
        self._set_status("准备就绪")

    def _choose_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "选择文件", "", "所有文件 (*)")
        if paths:
            from ui.widgets.command_input import DropSubmit
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
        self.voice_worker.voice_speak.connect(lambda t: self.voice_worker.speak_text(t) if self.voice_worker else None)
        self.voice_worker.dialogue_stopped.connect(self._handle_dialogue_stopped)
        self.voice_worker.finished.connect(self.voice_thread.quit)
        self.voice_thread.finished.connect(self.voice_worker.deleteLater)
        self.voice_thread.finished.connect(self.voice_thread.deleteLater)
        self.voice_thread.finished.connect(self._cleanup_voice_worker)
        self.voice_thread.start()

    def _handle_voice_text(self, text: str, has_pending: bool) -> None:
        QTimer.singleShot(0, lambda: self._do_voice_text(text, has_pending))

    def _do_voice_text(self, text: str, has_pending: bool) -> None:
        _ = has_pending
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
                QTimer.singleShot(0, lambda: self._process_voice_result(response, results, pending))
            except Exception as e:
                QTimer.singleShot(0, lambda: self._append_assistant(f"请求失败：{e}"))

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
                self.voice_btn.setStyleSheet(
                    "QPushButton#voiceBtn{background:#ffe6d6; color:#b6491d; border:1px solid #f2b89a; border-radius:21px; font-weight:800; font-size:18px;}"
                )
            else:
                self.voice_btn.setText("🎤")
                self.voice_btn.setStyleSheet("")

    def _handle_dialogue_stopped(self) -> None:
        if hasattr(self, "voice_btn"):
            self.voice_btn.setText("🎤")
            self.voice_btn.setStyleSheet("")
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
        dlg = StatsDialog(self._api, self)
        dlg.exec_()

    # ---- 右键菜单 ----
    def _show_chat_context_menu(self, position) -> None:
        from PyQt5.QtWidgets import QMenu
        menu = QMenu(self)
        clear_action = menu.addAction("清空对话")
        clear_action.triggered.connect(self._clear_query)
        menu.exec_(self.chat_view.viewport().mapToGlobal(position))

    # ---- 其他 ----
    def _load_defaults(self) -> None:
        vault = self._settings.value("vault/root", "", type=str) or ""
        self._vault_root = Path(vault) if vault else get_default_vault_root()
        last_query = self._settings.value("query/last", "", type=str) or ""
        if last_query:
            self.command_input.set_text(last_query)

    def _get_vault_root(self) -> Path:
        root = getattr(self, "_vault_root", None)
        if root is None:
            root = get_default_vault_root()
            self._vault_root = root
        return Path(root)

    def _set_status(self, text: str, is_error: bool = False) -> None:
        self.statusBar().showMessage(text, 8000 if not is_error else 15000)

    def _set_busy(self, busy: bool) -> None:
        self.send_btn.setEnabled(not busy)
        self._progress.setVisible(busy)

    def _stop_all(self) -> None:
        if hasattr(self, "voice_worker") and self.voice_worker is not None:
            self.voice_worker.stop()

    def _clear_query(self) -> None:
        self.command_input.clear()
        for i in range(self._chat_layout.count() - 2, -1, -1):
            item = self._chat_layout.takeAt(i)
            if item is None:
                continue
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._chat_plain_parts.clear()
        self._chat_html_parts.clear()
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
            QWidget {{ font-family:"Microsoft YaHei UI","Microsoft YaHei","PingFang SC",sans-serif; font-size:17px; color:#24322f; }}
            QMainWindow {{ background:#f3eee7; }}
            QWidget#titleBar {{ background:#fefcf8; border-bottom:1px solid #e8ddd0; }}
            QPushButton#titleBtn {{ background:transparent; border:none; color:#5c736d; font-size:14px; }}
            QPushButton#titleBtn:hover {{ background:#e8ddd0; }}
            QWidget#headerWidget {{ background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #fefcf8,stop:1 #f8f2e8); border:none; border-bottom:1px solid #e8ddd0; border-bottom-left-radius:17px; border-bottom-right-radius:17px; }}
            QLabel#headerBadge {{ background:rgba(255,250,240,0.92); color:#9a6a2b; border:1px solid #edd9bd; border-radius:17px; padding:5px 12px; font-size:11px; font-weight:700; letter-spacing:0.16em; }}
            QLabel#headerTitle {{ font-size:34px; font-weight:800; color:#173834; }}
            QLabel#headerSubtitle {{ font-size:18px; color:#35524d; font-weight:600; }}
            QWidget#updateBar {{ background:#fff7eb; border-bottom:1px solid #ecd8b8; }}
            QWidget#offlineBar {{ background:#fff3ee; border-bottom:1px solid #f0c0b5; }}
            QScrollArea#chatView {{ background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #fffcf7,stop:1 #f8f4ee); border:1px solid #e8ddd0; border-radius:17px; padding:2px; font-size:17px; }}
            QScrollArea#chatView QWidget#chatInner {{ background:transparent; }}
            QScrollArea#chatView QWidget {{ background:transparent; }}
            QWidget#inputWidget {{ background:#fefcf8; border:1px solid #e8ddd0; border-top:2px solid #dccbb8; border-radius:17px; }}
            QLineEdit {{ border:none; background:transparent; border-radius:17px; padding:12px 14px; font-size:17px; color:#24322f; }}
            QLineEdit::placeholder {{ color:#b49d7b; }}
            QPushButton {{ border:none; font-weight:600; font-size:15px; }}
            QPushButton:disabled {{ color:#d1d5db !important; }}
            QPushButton#sendBtn {{ background:qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #145f5a,stop:1 #0f766e); color:#fff; border-radius:24px; font-size:20px; font-weight:800; }}
            QPushButton#sendBtn:hover {{ background:#0d6b64; }}
            QPushButton#sendBtn:disabled {{ background:#d9ddd9 !important; }}
            QPushButton#voiceBtn {{ background:#e0f7f4; color:#0f766e; border:1px solid #0f766e; border-radius:21px; font-size:20px; }}
            QPushButton#voiceBtn:hover {{ background:#b3ece4; }}
            QPushButton#attachBtn {{ background:#e0f7f4; color:#0f766e; border:1px solid #b3ece4; border-radius:21px; font-size:18px; }}
            QPushButton#attachBtn:hover {{ background:#b3ece4; }}
            QPushButton#headerThemeBtn, QPushButton#headerStatsBtn {{ background:#fff7eb; border:1px solid #ecd8b8; border-radius:18px; font-size:16px; }}
            QPushButton#headerThemeBtn:hover, QPushButton#headerStatsBtn:hover {{ background:#fffdf7; }}
            QPushButton#headerClearBtn:hover {{ color:#a63d2a; }}
            QStatusBar {{ background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #fefcf8,stop:1 #f8f2e8); border:none; border-top:1px solid #e8ddd0; border-top-left-radius:17px; border-top-right-radius:17px; color:#7a6b5a; font-size:12px; padding:4px 16px; margin:0 1px; }}
            QStatusBar::item {{ border:none; }}
            QProgressBar {{ border:none; border-radius:17px; background:#e8ddd0; height:3px; max-width:160px; }}
            QProgressBar::chunk {{ background:#145f5a; border-radius:3px; }}
            QScrollBar:vertical {{ background:transparent; width:6px; }}
            QScrollBar::handle:vertical {{ background:#d5cec5; border-radius:3px; min-height:36px; margin:2px; }}
            QScrollBar::handle:vertical:hover {{ background:#b8afa3; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background:transparent; }}
            QTextBrowser#chatBubbleBody {{ font-size:{bfs}px; }}
        """

    def _dark_theme_css(self, bfs: int) -> str:
        return f"""
            * {{ outline: none; }}
            QWidget {{ font-family:"Microsoft YaHei UI","Microsoft YaHei","PingFang SC",sans-serif; font-size:17px; color:#d4d4d4; }}
            QMainWindow {{ background:#1a2e2c; }}
            QWidget#titleBar {{ background:#0d3b36; border-bottom:1px solid #1a4a44; }}
            QPushButton#titleBtn {{ background:transparent; border:none; color:#8aaaa4; font-size:14px; }}
            QPushButton#titleBtn:hover {{ background:#1a4a44; }}
            QWidget#headerWidget {{ background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #0d3b36,stop:1 #143d38); border:none; border-bottom:1px solid #1a4a44; border-bottom-left-radius:17px; border-bottom-right-radius:17px; }}
            QLabel#headerBadge {{ background:rgba(20,61,56,0.9); color:#d4a259; border:1px solid #3a6a5a; border-radius:17px; padding:5px 12px; font-size:11px; font-weight:700; letter-spacing:0.16em; }}
            QLabel#headerTitle {{ font-size:34px; font-weight:800; color:#a8e6cf; }}
            QLabel#headerSubtitle {{ font-size:18px; color:#7fb8a8; font-weight:600; }}
            QWidget#updateBar {{ background:#2a3a20; border-bottom:1px solid #4a5a30; }}
            QWidget#offlineBar {{ background:#3a2020; border-bottom:1px solid #5a3030; }}
            QScrollArea#chatView {{ background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #1a2e2c,stop:1 #0d2622); border:1px solid #1a4a44; border-radius:17px; padding:2px; font-size:17px; }}
            QScrollArea#chatView QWidget#chatInner {{ background:transparent; }}
            QScrollArea#chatView QWidget {{ background:transparent; }}
            QWidget#inputWidget {{ background:#0d3b36; border:1px solid #1a4a44; border-top:2px solid #2a5a50; border-radius:17px; }}
            QLineEdit {{ border:none; background:transparent; border-radius:17px; padding:12px 14px; font-size:17px; color:#d4d4d4; }}
            QLineEdit::placeholder {{ color:#5a7a70; }}
            QPushButton {{ border:none; font-weight:600; font-size:15px; }}
            QPushButton:disabled {{ color:#4a4a4a !important; }}
            QPushButton#sendBtn {{ background:qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #0f766e,stop:1 #14a89a); color:#fff; border-radius:24px; font-size:20px; font-weight:800; }}
            QPushButton#sendBtn:hover {{ background:#14a89a; }}
            QPushButton#sendBtn:disabled {{ background:#2a3a38 !important; }}
            QPushButton#voiceBtn {{ background:#0d3b36; color:#14a89a; border:1px solid #1a4a44; border-radius:21px; font-size:20px; }}
            QPushButton#voiceBtn:hover {{ background:#1a4a44; }}
            QPushButton#attachBtn {{ background:#0d3b36; color:#14a89a; border:1px solid #1a4a44; border-radius:21px; font-size:18px; }}
            QPushButton#attachBtn:hover {{ background:#1a4a44; }}
            QPushButton#headerThemeBtn, QPushButton#headerStatsBtn {{ background:#0d3b36; border:1px solid #1a4a44; border-radius:18px; font-size:16px; }}
            QPushButton#headerThemeBtn:hover, QPushButton#headerStatsBtn:hover {{ background:#1a4a44; }}
            QPushButton#headerClearBtn {{ background:#0d3b36; border:1px solid #1a4a44; color:#d4a259; }}
            QPushButton#headerClearBtn:hover {{ color:#e8b85a; }}
            QStatusBar {{ background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #0d3b36,stop:1 #0a2e2a); border:none; border-top:1px solid #1a4a44; border-top-left-radius:17px; border-top-right-radius:17px; color:#7fb8a8; font-size:12px; padding:4px 16px; margin:0 1px; }}
            QStatusBar::item {{ border:none; }}
            QProgressBar {{ border:none; border-radius:17px; background:#1a4a44; height:3px; max-width:160px; }}
            QProgressBar::chunk {{ background:#14a89a; border-radius:3px; }}
            QScrollBar:vertical {{ background:transparent; width:6px; }}
            QScrollBar::handle:vertical {{ background:#2a4a44; border-radius:3px; min-height:36px; margin:2px; }}
            QScrollBar::handle:vertical:hover {{ background:#3a5a50; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background:transparent; }}
            QTextBrowser#chatBubbleBody {{ font-size:{bfs}px; }}
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
