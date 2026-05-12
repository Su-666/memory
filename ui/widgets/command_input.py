from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import QCompleter, QHBoxLayout, QLabel, QWidget
from qfluentwidgets import isDarkTheme

from qfluentwidgets import LineEdit, PushButton


@dataclass(frozen=True)
class CommandSubmit:
    text: str


@dataclass(frozen=True)
class DropSubmit:
    paths: list[str]
    caption: str


class CommandInput(QWidget):
    submitted_text = Signal(object)
    submitted_drop = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._pending_paths: list[str] = []
        self._drag_over = False

        self._edit = LineEdit()
        self._edit.setPlaceholderText("输入一句话，或拖入图片 / 文件后添加说明…")
        self._edit.setClearButtonEnabled(True)
        self._edit.setObjectName("commandLineEdit")
        self._edit.returnPressed.connect(self.submit)

        completer = QCompleter(["/search ", "/搜 ", "帮我记住 ", "帮我查找 "], self._edit)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self._edit.setCompleter(completer)
        self._edit.textChanged.connect(self._update_badge)

        self._attach_badge = QLabel("")
        self._attach_badge.setObjectName("attachBadge")
        self._attach_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._attach_badge.setVisible(False)

        self._delete_btn = PushButton("×")
        self._delete_btn.setObjectName("deleteAttachBtn")
        self._delete_btn.setFixedSize(24, 24)
        self._delete_btn.setToolTip("移除当前附件")
        self._delete_btn.setVisible(False)
        self._delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._delete_btn.clicked.connect(self.clear_pending_attachments)
        self._delete_btn.setStyleSheet(
            "QPushButton{background:rgba(239,68,68,0.12); border:none; border-radius:12px; "
            "color:#ef4444; font-size:14px; font-weight:700;}"
            "QPushButton:hover{background:rgba(239,68,68,0.25);}"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(self._edit, 1)
        layout.addWidget(self._attach_badge)
        layout.addWidget(self._delete_btn)

        self._update_badge("")

    def has_pending_attachments(self) -> bool:
        return bool(self._pending_paths)

    def pending_attachments_count(self) -> int:
        return len(self._pending_paths)

    def clear_pending_attachments(self) -> None:
        self._pending_paths.clear()
        self._update_badge(self._edit.text())

    def text(self) -> str:
        return self._edit.text()

    def set_text(self, text: str) -> None:
        self._edit.setText(text)

    def clear(self) -> None:
        self._edit.clear()
        self.clear_pending_attachments()

    def focus_input(self) -> None:
        self._edit.setFocus()

    def _update_badge(self, _: str) -> None:
        if self._pending_paths:
            count = len(self._pending_paths)
            self._attach_badge.setText(f"附件 {count}")
            filenames = [Path(path).name for path in self._pending_paths[:5]]
            if count > 5:
                filenames.append(f"... 共 {count} 个文件")
            tooltip = "已添加附件：\n" + "\n".join(filenames)
            self._attach_badge.setToolTip(tooltip)
            self._attach_badge.setVisible(True)
            self._delete_btn.setVisible(True)
        else:
            self._attach_badge.setVisible(False)
            self._delete_btn.setVisible(False)

    def submit(self) -> None:
        text = self._edit.text().strip()
        if self._pending_paths:
            paths = list(dict.fromkeys(self._pending_paths))
            self.submitted_drop.emit(DropSubmit(paths=paths, caption=text))
            return
        if not text:
            return
        self.submitted_text.emit(CommandSubmit(text=text))

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._set_drag_highlight(True)
            return
        event.ignore()

    def dragLeaveEvent(self, event) -> None:
        self._set_drag_highlight(False)
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        self._set_drag_highlight(False)
        if not event.mimeData().hasUrls():
            event.ignore()
            return
        paths: list[str] = []
        for url in event.mimeData().urls():
            if url.isLocalFile():
                paths.append(str(Path(url.toLocalFile())))
        if not paths:
            event.ignore()
            return
        self._pending_paths.extend(paths)
        self._update_badge(self._edit.text())
        event.acceptProposedAction()

    def _set_drag_highlight(self, on: bool):
        self._drag_over = on
        dark = isDarkTheme()
        if on:
            border_c = "rgba(90,232,184,0.50)" if not dark else "rgba(90,232,184,0.40)"
            bg_c = "rgba(15,118,110,0.06)" if not dark else "rgba(90,232,184,0.06)"
            self.setStyleSheet(
                f"CommandInput{{border:2px dashed {border_c}; "
                f"background:{bg_c}; border-radius:12px;}}"
            )
        else:
            self.setStyleSheet("")
