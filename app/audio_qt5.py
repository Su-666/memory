"""
PyQt5 音频模块 - 替代 sounddevice 的录音和播放功能
"""
import io
import wave
import numpy as np
from typing import Callable, Any

# PyQt5 音频相关导入
try:
    from PyQt5.QtCore import QBuffer, QByteArray, QIODevice, QTimer
    from PyQt5.QtMultimedia import QAudioInput, QAudioOutput, QAudioDeviceInfo, QAudioFormat
    HAS_PYQT5_AUDIO = True
except ImportError:
    HAS_PYQT5_AUDIO = False
    print("[音频] PyQt5 多媒体模块未安装")


class PyQt5AudioPlayer:
    """使用 PyQt5 的音频播放器"""

    def __init__(self):
        self.output = None
        self.buffer = None
        self.device = None
        self._is_playing = False
        self._stop_flag = []

    def play_wav(self, wav_bytes: bytes, stop_flag: list = None) -> None:
        """播放 WAV 音频"""
        if not HAS_PYQT5_AUDIO:
            raise RuntimeError("PyQt5 音频模块不可用")

        try:
            # 解析 WAV
            with wave.open(io.BytesIO(wav_bytes), 'rb') as wf:
                channels = wf.getnchannels()
                sample_rate = wf.getframerate()
                sample_width = wf.getsampwidth()
                audio_data = wf.readframes(wf.getnframes())

            # 转换为 PCM 16-bit
            if sample_width == 2:
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
            elif sample_width == 1:
                audio_array = np.frombuffer(audio_data, dtype=np.uint8)
                audio_array = (audio_array - 128) * 256
            else:
                raise RuntimeError(f"不支持的采样宽度: {sample_width}")

            # 转换为字节
            if channels == 1:
                audio_bytes = audio_array.tobytes()
            else:
                # 转换为单声道（交错）
                audio_array = audio_array.reshape(-1, channels)
                audio_bytes = audio_array[:, 0].tobytes()

            # 设置音频格式
            format = QAudioFormat()
            format.setSampleRate(sample_rate)
            format.setChannelCount(1)
            format.setSampleSize(16)
            format.setCodec('audio/pcm')
            format.setByteOrder(QAudioFormat.LittleEndian)
            format.setSampleType(QAudioFormat.SignedInt)

            # 获取音频设备
            device_info = QAudioDeviceInfo.defaultOutputDevice()
            if not device_info.isFormatSupported(format):
                print(f"[音频] 默认格式不支持，尝试 nearest")
                format = device_info.nearestFormat(format)

            # 创建输出设备
            self.output = QAudioOutput(format)
            self._stop_flag = stop_flag or []

            # 创建缓冲区
            self.buffer = QBuffer()
            self.buffer.setBuffer(QByteArray(audio_bytes))
            self.buffer.open(QIODevice.ReadOnly)

            # 播放
            self.output.stateChanged.connect(self._on_state_changed)
            self.output.start(self.buffer)
            self._is_playing = True

            # 如果需要等待播放完成
            if stop_flag is None:
                # 等待播放完成
                while self.output.state() == QAudio.ActiveState:
                    QTimer.singleShot(50, lambda: None)
                    import time
                    time.sleep(0.05)

        except Exception as e:
            print(f"[音频] 播放失败: {e}")
            raise

    def _on_state_changed(self, state):
        if state == QAudio.IdleState:
            self._is_playing = False
        elif state == QAudio.StoppedState:
            self._is_playing = False

    def stop(self):
        """停止播放"""
        if self.output and self._is_playing:
            self.output.stop()
            self._is_playing = False


class PyQt5AudioRecorder:
    """使用 PyQt5 的音频录制器"""

    def __init__(self, sample_rate=16000, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.input = None
        self.buffer = None
        self.device = None
        self._is_recording = False
        self._recorded_data = bytearray()

    def start_recording(self) -> None:
        """开始录制"""
        if not HAS_PYQT5_AUDIO:
            raise RuntimeError("PyQt5 音频模块不可用")

        # 设置音频格式
        format = QAudioFormat()
        format.setSampleRate(self.sample_rate)
        format.setChannelCount(self.channels)
        format.setSampleSize(16)
        format.setCodec('audio/pcm')
        format.setByteOrder(QAudioFormat.LittleEndian)
        format.setSampleType(QAudioFormat.SignedInt)

        # 获取音频设备
        self.device = QAudioDeviceInfo.defaultInputDevice()
        if not self.device.isFormatSupported(format):
            print(f"[音频] 输入格式不支持，尝试 nearest")
            format = self.device.nearestFormat(format)

        # 创建输入设备
        self.input = QAudioInput(format)
        self._recorded_data = bytearray()

        # 创建内存缓冲区
        self.buffer = QBuffer()
        self.buffer.setBuffer(self._recorded_data)
        self.buffer.open(QIODevice.WriteOnly)

        # 开始录制
        self.input.start(self.buffer)
        self._is_recording = True
        print(f"[音频] 开始录制，格式: {format.sampleRate()}Hz, {format.channelCount()}通道")

    def stop_recording(self) -> bytes:
        """停止录制，返回 PCM 数据"""
        if not self._is_recording:
            return b''

        self.input.stop()
        self._is_recording = False

        # 从缓冲区获取数据
        audio_bytes = bytes(self._recorded_data)
        print(f"[音频] 录制完成，数据大小: {len(audio_bytes)} bytes")

        # 转换为 WAV 格式
        return self._make_wav(audio_bytes, self.sample_rate, self.channels)

    def _make_wav(self, pcm_data: bytes, sample_rate: int, channels: int) -> bytes:
        """将 PCM 数据封装为 WAV"""
        import struct

        # WAV 文件头
        data_size = len(pcm_data)
        wav_buffer = io.BytesIO()

        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_data)

        return wav_buffer.getvalue()

    def is_recording(self) -> bool:
        return self._is_recording


def check_audio_devices():
    """检查音频设备"""
    if not HAS_PYQT5_AUDIO:
        return False, "PyQt5 音频模块不可用"

    try:
        input_device = QAudioDeviceInfo.defaultInputDevice()
        output_device = QAudioDeviceInfo.defaultOutputDevice()

        if not input_device.isNull():
            print(f"[音频] 输入设备: {input_device.deviceName()}")
        if not output_device.isNull():
            print(f"[音频] 输出设备: {output_device.deviceName()}")

        return True, f"输入: {input_device.deviceName()}, 输出: {output_device.deviceName()}"
    except Exception as e:
        return False, f"检查失败: {e}"
