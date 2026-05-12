"""聊天气泡和记忆卡片组件

气泡直接用 QLabel 实现（背景+文字同一层），避免 QFrame 套 QLabel 导致文字模糊。
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QFont, QFontMetrics, QPixmap
from PySide6.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QSizePolicy,
    QVBoxLayout, QWidget,
)

from qfluentwidgets import isDarkTheme, PushButton

from ui.styles import (
    BUBBLE_RADIUS, BUBBLE_FONT_PX, BUBBLE_TITLE_FONT_PX,
    BUBBLE_PAD, AVATAR_SIZE,
)

# 气泡最大宽度占可用宽度的比例
_BUBBLE_MAX_RATIO = 0.82
_GAP = 10


def _make_avatar(text: str, is_user: bool, size: int = AVATAR_SIZE) -> QLabel:
    lbl = QLabel(text)
    lbl.setFixedSize(size, size)
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    bg = "#5a8a7a" if is_user else "#0f766e"
    lbl.setStyleSheet(
        f"background:{bg}; color:#ffffff; border-radius:{size // 2}px;"
        f"font-size:15px; font-weight:800;"
    )
    return lbl


class _BubbleLabel(QLabel):
    """自带背景和内边距的气泡标签，sizeHint 根据文字自适应"""

    def __init__(self, bg: str, text_color: str, radius: int, pad: int,
                 font: QFont, max_w: int, border: str = "none", parent=None):
        super().__init__(parent)
        self._pad = pad
        self._font = font
        self._max_w = max_w
        self._font_px = font.pointSize()
        self._radius = radius
        self._border = border
        # 解析 border 宽度（QSS border 会占额外空间）
        self._border_w = self._parse_border_w(border)
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self.setFont(font)
        self.setStyleSheet(
            f"background:{bg}; color:{text_color}; border:{border};"
            f"border-radius:{radius}px;"
            f"font-size:{self._font_px}px; font-weight:bold;"
        )
        # 内边距：上下减少一点让文字更贴近中心
        self.setContentsMargins(pad - 2, pad - 2, pad - 2, pad - 2)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

    @staticmethod
    def _parse_border_w(border: str) -> int:
        """从 border QSS 字符串中提取宽度像素值"""
        if not border or border == "none":
            return 0
        parts = border.strip().split()
        if parts:
            try:
                return int(parts[0].replace("px", ""))
            except (ValueError, IndexError):
                pass
        return 0

    def setMaxWidth(self, w: int):
        self._max_w = w
        self.setMaximumWidth(w)

    def set_style(self, bg: str, text_color: str, radius: int, border: str = "none"):
        """更新气泡样式（用于主题切换）"""
        self._border_w = self._parse_border_w(border)
        self.setStyleSheet(
            f"background:{bg}; color:{text_color}; border:{border};"
            f"border-radius:{radius}px;"
            f"font-size:{self._font_px}px; font-weight:bold;"
        )

    def _total_extra(self) -> tuple[int, int]:
        """返回水平和垂直方向的额外占用（contentsMargins + border）"""
        m = self.contentsMargins()
        bw = self._border_w * 2
        return m.left() + m.right() + bw, m.top() + m.bottom() + bw

    def sizeHint(self) -> QSize:
        extra_w, extra_h = self._total_extra()
        fm = self.fontMetrics()
        text = self.text()
        text_avail = max(20, self._max_w - extra_w)
        if not text:
            return QSize(extra_w + 20, extra_h + fm.height())
        rect = fm.boundingRect(0, 0, text_avail, 99999,
                                Qt.TextFlag.TextWordWrap, text)
        w = min(rect.width() + extra_w + 2, self._max_w)
        h = rect.height() + extra_h
        return QSize(max(w, extra_w + 20), max(h, extra_h + fm.height()))

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        extra_w, extra_h = self._total_extra()
        text_avail = max(20, width - extra_w)
        rect = self.fontMetrics().boundingRect(
            0, 0, text_avail, 99999, Qt.TextFlag.TextWordWrap, self.text())
        return rect.height() + extra_h


class ChatBubble(QWidget):
    """聊天气泡组件"""

    def __init__(
        self,
        *,
        align_right: bool,
        title: str,
        plain_text: str,
        bg: str,
        border: str,
        title_color: str,
        text_color: str,
        show_title: bool = True,
        pad_x: int | None = None,
        body_font_px: int | None = None,
        parent=None,
        **kwargs,
    ):
        super().__init__(parent)
        self._pad = pad_x if pad_x is not None else BUBBLE_PAD
        self._bfs = body_font_px if body_font_px is not None else BUBBLE_FONT_PX
        self._plain_text = plain_text or ""
        self._last_avail = 0

        self.setProperty("bubbleAlign", "right" if align_right else "left")

        font = QFont("Microsoft YaHei UI", self._bfs)
        font.setBold(True)
        font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias
                               | QFont.StyleStrategy.PreferQuality)

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 4, 0, 4)
        row.setSpacing(_GAP)

        avatar = _make_avatar("我" if align_right else "暖", is_user=align_right)

        self._name_lbl = None
        if show_title and title:
            self._name_lbl = QLabel(title)
            self._name_lbl.setStyleSheet(
                f"color:{title_color}; font-size:{BUBBLE_TITLE_FONT_PX}px;"
                f"font-weight:700; margin-bottom:3px;"
            )

        # 气泡 = QLabel 直接带背景，不需要 QFrame 包裹
        self._body = _BubbleLabel(bg=bg, text_color=text_color,
                                  radius=BUBBLE_RADIUS, pad=self._pad,
                                  font=font, max_w=600, border=border)
        self._body.setText(self._plain_text)

        col = QVBoxLayout()
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)
        if self._name_lbl:
            col.addWidget(self._name_lbl, 0, Qt.AlignmentFlag.AlignLeft)
        col.addWidget(self._body, 0, Qt.AlignmentFlag.AlignLeft)

        if align_right:
            row.addStretch(1)
            row.addLayout(col)
            row.addWidget(avatar, 0, Qt.AlignmentFlag.AlignTop)
        else:
            row.addWidget(avatar, 0, Qt.AlignmentFlag.AlignTop)
            row.addLayout(col)
            row.addStretch(1)

        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

    def body(self):
        return self._body

    def refit(self, avail_width: int) -> None:
        if avail_width == self._last_avail:
            return
        self._last_avail = avail_width

        avatar_total = AVATAR_SIZE + _GAP + 8
        max_w = max(80, int((avail_width - avatar_total) * _BUBBLE_MAX_RATIO))
        self._body.setMaxWidth(max_w)


class RelatedCard(QFrame):
    """搜索结果记忆卡片（与 Web 端 result-item 风格一致）"""

    def __init__(self, item: dict[str, Any], preview_w: int = 280, preview_h: int = 180, parent=None):
        super().__init__(parent)

        dark = isDarkTheme()
        if dark:
            card_bg = "rgba(22,38,35,0.92)"
            card_border = "rgba(20,168,154,0.08)"
            card_border_hover = "rgba(20,184,166,0.15)"
            title_color = "#d8e8e4"
            meta_color = "#5a8a78"
            body_color = "#8ab8a8"
        else:
            card_bg = "rgba(255,255,255,0.9)"
            card_border = "rgba(15,118,110,0.06)"
            card_border_hover = "rgba(20,184,166,0.15)"
            title_color = "#1a2726"
            meta_color = "#7a9a8e"
            body_color = "#4a6a62"

        self.setObjectName("relatedItemFrame")
        self.setStyleSheet(
            f"QFrame#relatedItemFrame{{background:{card_bg}; border:1px solid {card_border};"
            f"border-radius:14px;}}"
            f"QFrame#relatedItemFrame:hover{{border-color:{card_border_hover};}}"
        )
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        title = str(item.get("title", "") or "").strip()
        time_text = str(item.get("time", "") or "").strip()
        summary = str(item.get("summary", "") or item.get("body_snippet", "") or "").strip()
        file_path = str(item.get("file_path", "") or "").strip()

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(6)

        header = QLabel(title or "（未命名）")
        header.setWordWrap(True)
        header.setStyleSheet(f"font-weight:700; color:{title_color}; font-size:14px; margin-bottom:6px;")
        header.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        meta = QLabel(time_text)
        meta.setStyleSheet(f"color:{meta_color}; font-size:11px; font-weight:600;")
        meta.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        meta.setVisible(bool(time_text))

        body = QLabel(summary)
        body.setWordWrap(True)
        body.setStyleSheet(f"color:{body_color}; font-size:13px; line-height:1.5;")
        body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        body.setVisible(bool(summary))

        lay.addWidget(header)
        if time_text:
            lay.addWidget(meta)
        if summary:
            lay.addWidget(body)

        # 图片预览
        if file_path:
            try:
                p = Path(file_path)
                suf = p.suffix.lower()
                if suf in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"} and p.exists():
                    pix = QPixmap(str(p))
                    if not pix.isNull():
                        img = QLabel()
                        img.setFixedSize(preview_w, preview_h)
                        img.setStyleSheet(
                            f"border-radius:10px; "
                            f"border:1px solid {'rgba(20,168,154,0.12)' if dark else 'rgba(15,118,110,0.08)'};"
                        )
                        img.setScaledContents(True)
                        img.setPixmap(pix.scaled(
                            preview_w, preview_h,
                            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                            Qt.TransformationMode.SmoothTransformation,
                        ))
                        img.setCursor(Qt.CursorShape.PointingHandCursor)
                        img.mousePressEvent = lambda e, p=pix: self._show_zoom(p)
                        lay.addWidget(img, 0, Qt.AlignmentFlag.AlignLeft)
            except Exception:
                pass

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        copy_btn = PushButton("复制记忆")
        copy_btn.setFixedHeight(30)
        mem_text = f"【{title}】\n{summary}\n\n时间: {time_text}"
        copy_btn.clicked.connect(lambda: QApplication.clipboard().setText(mem_text))
        btn_row.addWidget(copy_btn)

        if file_path:
            open_btn = PushButton("打开文件")
            open_btn.setFixedHeight(30)
            open_btn.clicked.connect(lambda: self._open_file(file_path))
            btn_row.addWidget(open_btn)

        btn_row.addStretch(1)
        lay.addLayout(btn_row)

    def _show_zoom(self, pixmap: QPixmap):
        from ui.dialogs import ImageZoomDialog
        dlg = ImageZoomDialog(pixmap, self.window())
        dlg.resize(self.window().size())
        dlg.exec()

    @staticmethod
    def _open_file(file_path: str):
        p = Path(file_path)
        if p.exists():
            if sys.platform == "win32":
                os.startfile(str(p))
            else:
                subprocess.Popen(["xdg-open", str(p)])
