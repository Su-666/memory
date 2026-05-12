"""对话框组件：统计面板、图片预览、服务器设置"""

import threading
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QColor
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QVBoxLayout, QWidget, QFrame,
    QGraphicsDropShadowEffect,
)

from qfluentwidgets import (
    Dialog, LineEdit, PushButton, SmoothScrollArea, isDarkTheme,
)

from app.api_client import MemoryApiClient
from .styles import PRIMARY, PRIMARY_LIGHT, PRIMARY_DARK, ACCENT


class HoverChip(QLabel):
    """带 hover 高亮效果的标签芯片"""

    def __init__(self, text: str, normal_ss: str, hover_ss: str, parent=None):
        super().__init__(text, parent)
        self._normal_ss = normal_ss
        self._hover_ss = hover_ss
        self.setStyleSheet(normal_ss)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def enterEvent(self, event):
        self.setStyleSheet(self._hover_ss)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.setStyleSheet(self._normal_ss)
        super().leaveEvent(event)


class StatsDialog(Dialog):
    """记忆统计对话框"""
    _stats_result = Signal(dict)
    _stats_error = Signal(str)

    def __init__(self, api_client: MemoryApiClient, parent=None):
        super().__init__("记忆统计", "", parent)
        self._api = api_client
        self._stats_result.connect(self._render_stats)
        self._stats_error.connect(lambda e: self._show_error(str(e)))

        self.setFixedSize(440, 500)
        self._build_content()
        self._load_stats()

    def _build_content(self):
        self._stats_container = QVBoxLayout()
        self._stats_container.setSpacing(10)

        self._loading_lbl = QLabel("加载中...")
        dark = isDarkTheme()
        self._loading_lbl.setStyleSheet(
            f"color:{'#6aaa98' if dark else '#7a9a8e'}; font-size:14px;"
        )
        self._stats_container.addWidget(self._loading_lbl)

        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(24, 16, 24, 16)
        content_layout.setSpacing(16)
        content_layout.addLayout(self._stats_container)
        content_layout.addStretch(1)

        if self.vBoxLayout:
            self.vBoxLayout.addLayout(content_layout)

    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
            elif item.layout():
                self._clear_layout(item.layout())

    def _load_stats(self):
        def _do():
            try:
                stats = self._api.get_stats()
                self._stats_result.emit(stats)
            except Exception as e:
                self._stats_error.emit(str(e))
        threading.Thread(target=_do, daemon=True).start()

    def _render_stats(self, stats: dict):
        self._clear_layout(self._stats_container)
        dark = isDarkTheme()

        items = [
            ("总记忆数", str(stats.get("total_memories", 0))),
            ("最近记忆", str(stats.get("recent_count", 0))),
            ("文件数量", str(stats.get("total_files", 0))),
            ("存储大小", stats.get("storage_formatted", "未知")),
            ("最后更新", stats.get("last_updated", "无")),
        ]
        for label, value in items:
            card = QFrame()
            card.setObjectName("statRowCard")
            if dark:
                card.setStyleSheet(
                    "QFrame#statRowCard{background:rgba(16,30,26,0.85); "
                    "border:1px solid rgba(20,168,154,0.08); border-radius:12px;}"
                )
            else:
                card.setStyleSheet(
                    "QFrame#statRowCard{background:rgba(255,255,255,0.90); "
                    "border:1px solid rgba(15,118,110,0.06); border-radius:12px;}"
                )
            shadow = QGraphicsDropShadowEffect(card)
            shadow.setBlurRadius(16)
            shadow.setOffset(0, 2)
            shadow.setColor(QColor(0, 0, 0, 18 if dark else 12))
            card.setGraphicsEffect(shadow)
            row = QHBoxLayout(card)
            row.setContentsMargins(16, 12, 16, 12)

            lbl = QLabel(label)
            val = QLabel(value)
            if dark:
                lbl.setStyleSheet("color:#7ab8a8; font-size:13px; font-weight:500;")
                val.setStyleSheet("color:#6ee7c0; font-size:18px; font-weight:800; letter-spacing:0.5px;")
            else:
                lbl.setStyleSheet("color:#5a8a7e; font-size:13px; font-weight:500;")
                val.setStyleSheet("color:#0a5c56; font-size:18px; font-weight:800; letter-spacing:0.5px;")
            row.addWidget(lbl)
            row.addStretch(1)
            row.addWidget(val)
            self._stats_container.addWidget(card)

        top_tags = stats.get("top_tags", [])
        if top_tags:
            sep = QLabel("热门标签")
            sep.setStyleSheet(
                f"color:{'#e8c878' if dark else '#8a6a28'}; "
                f"font-size:12px; font-weight:700; margin-top:12px; letter-spacing:0.06em;"
            )
            self._stats_container.addWidget(sep)

            tags_layout = QHBoxLayout()
            tags_layout.setSpacing(8)
            for tag, count in top_tags[:8]:
                if dark:
                    normal_ss = (
                        "background:rgba(20,61,56,0.85); border:1px solid rgba(232,200,120,0.15);"
                        "border-radius:14px; padding:5px 14px; color:#e8c878; "
                        "font-size:12px; font-weight:600;"
                    )
                    hover_ss = (
                        "background:rgba(232,200,120,0.18); border:1px solid rgba(232,200,120,0.30);"
                        "border-radius:14px; padding:5px 14px; color:#f0d888; "
                        "font-size:12px; font-weight:600;"
                    )
                else:
                    normal_ss = (
                        "background:rgba(255,249,240,0.90); border:1px solid rgba(218,165,32,0.15);"
                        "border-radius:14px; padding:5px 14px; color:#8a6a28; "
                        "font-size:12px; font-weight:600;"
                    )
                    hover_ss = (
                        "background:rgba(218,165,32,0.15); border:1px solid rgba(218,165,32,0.30);"
                        "border-radius:14px; padding:5px 14px; color:#6a5020; "
                        "font-size:12px; font-weight:600;"
                    )
                chip = HoverChip(f"{tag} ({count})", normal_ss, hover_ss)
                tags_layout.addWidget(chip)
            tags_layout.addStretch(1)
            self._stats_container.addLayout(tags_layout)

    def _show_error(self, msg: str):
        self._clear_layout(self._stats_container)
        err = QLabel(f"加载失败: {msg}")
        err.setStyleSheet("color:#ef4444; font-size:14px;")
        self._stats_container.addWidget(err)


class ImageZoomDialog(QDialog):
    """图片放大预览对话框"""

    def __init__(self, pixmap: QPixmap, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setStyleSheet("background: rgba(0,0,0,0.92);")
        self._pix = pixmap

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._img_label = QLabel()
        self._img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._img_label, 1)

        close_btn = PushButton("✕", self)
        close_btn.setFixedSize(40, 40)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.clicked.connect(self.close)
        close_btn.move(20, 20)
        close_btn.setStyleSheet(
            "QPushButton{background:rgba(255,255,255,0.10); border:1px solid rgba(255,255,255,0.08);"
            "border-radius:20px; color:#fff; font-size:18px;}"
            "QPushButton:hover{background:rgba(255,255,255,0.25); border-color:rgba(255,255,255,0.20);}"
        )

        self._hint_lbl = QLabel("点击任意位置关闭")
        self._hint_lbl.setStyleSheet("color:rgba(255,255,255,0.35); font-size:12px; padding:10px;")
        self._hint_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._hint_lbl)

        self._scale_factor = 1.0
        self._update_pixmap()

    def _update_pixmap(self):
        if self._pix.isNull():
            return
        sw = self.width() - 40
        sh = self.height() - 40
        scaled = self._pix.scaled(
            sw, sh,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._img_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_pixmap()

    def mousePressEvent(self, event):
        self.close()


class ServerSettingsDialog(Dialog):
    """服务器地址设置对话框"""

    def __init__(self, current_url: str, parent=None):
        super().__init__("服务器设置", "输入服务器地址后点击确认", parent)
        self._result_url = ""
        self.setFixedSize(480, 240)
        dark = isDarkTheme()

        content = QVBoxLayout()
        content.setContentsMargins(28, 18, 28, 18)
        content.setSpacing(14)

        hint = QLabel("本地服务器: http://127.0.0.1:5000")
        hint.setStyleSheet(
            f"color:{'#6aaa98' if dark else '#7a9a8e'}; font-size:12px; "
            f"background:{'rgba(20,168,154,0.06)' if dark else 'rgba(15,118,110,0.04)'}; "
            f"padding:10px 14px; border-radius:10px;"
        )
        content.addWidget(hint)

        self._url_edit = LineEdit()
        self._url_edit.setText(current_url)
        self._url_edit.setPlaceholderText("输入服务器地址...")
        content.addWidget(self._url_edit)

        content.addStretch(1)

        if self.vBoxLayout:
            self.vBoxLayout.addLayout(content)

        # 隐藏默认的 content label
        if hasattr(self, 'contentLabel'):
            self.contentLabel.hide()

    def result_url(self) -> str:
        return self._result_url

    def accept(self):
        self._result_url = self._url_edit.text().strip().rstrip("/")
        super().accept()
