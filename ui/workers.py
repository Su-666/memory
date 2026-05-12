"""语音处理 Worker 和音频工具函数"""

import base64
import io
import os
import subprocess
import sys
import threading
import time
import wave
from typing import Any

from PySide6.QtCore import QObject, Signal

from app.api_client import MemoryApiClient

try:
    import numpy as np
except ImportError:
    np = None

try:
    import sounddevice as sd
except ImportError:
    sd = None


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


class VoiceWorker(QObject):
    """语音处理 Worker - 录音本地，ASR/TTS 通过服务端 API"""
    status_changed = Signal(str, bool)
    listening_changed = Signal(bool)
    finished = Signal()
    dialogue_stopped = Signal()
    user_text_ready = Signal(str)
    voice_text_ready = Signal(str)

    def __init__(self, api_client: MemoryApiClient, dialogue_mode: bool = False) -> None:
        super().__init__()
        self._api = api_client
        self._stop_requested = False
        self._recorded_chunks: list[bytes] = []
        self._dialogue_mode = dialogue_mode
        self._is_speaking = False
        self._playback_stop_flag: list = []

    @staticmethod
    def _pcm_to_wav(pcm: bytes, sample_rate: int = 16000) -> bytes:
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm)
        return wav_buf.getvalue()

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

        wav_bytes = self._pcm_to_wav(pcm, sample_rate)
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
                        sd.sleep(30)
                        if not self._recorded_chunks:
                            continue
                        last_chunk = self._recorded_chunks[-1]
                        audio_data = np.frombuffer(last_chunk, dtype=np.int16)
                        rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))

                        if rms > 600:
                            if not speech_started:
                                speech_started = True
                                recording_start = time.time()
                                self.status_changed.emit("正在聆听...", False)
                            silence_start = None
                        elif speech_started:
                            if silence_start is None:
                                silence_start = time.time()
                            elif time.time() - silence_start > 1.2:
                                break
                        if recording_start and (time.time() - recording_start > 30):
                            break

                if self._stop_requested:
                    break

                self.listening_changed.emit(False)
                pcm = b"".join(self._recorded_chunks)
                if not pcm or len(pcm) < sample_rate:
                    self.listening_changed.emit(True)
                    self.status_changed.emit("没听清，请再说一次...", False)
                    continue

                wav_bytes = self._pcm_to_wav(pcm, sample_rate)
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
