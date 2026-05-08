from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal  # pyright: ignore[reportMissingImports]
from PyQt5.QtGui import QDragEnterEvent, QDropEvent  # pyright: ignore[reportMissingImports]
from PyQt5.QtWidgets import QCompleter, QHBoxLayout, QLabel, QLineEdit, QPushButton, QWidget  # pyright: ignore[reportMissingImports]


@dataclass(frozen=True)
class CommandSubmit:
    text: str


@dataclass(frozen=True)
class DropSubmit:
    paths: list[str]
    caption: str


class CommandInput(QWidget):
    submitted_text = pyqtSignal(object)
    submitted_drop = pyqtSignal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._pending_paths: list[str] = []

        self._edit = QLineEdit()
        self._edit.setPlaceholderText("输入一句话，或拖入图片 / 文件后添加说明…")
        self._edit.setClearButtonEnabled(True)
        self._edit.setStyleSheet("font-size: 17px;")
        self._edit.returnPressed.connect(self.submit)

        completer = QCompleter(["/search ", "/搜 ", "帮我记住 ", "帮我查找 "], self._edit)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setFilterMode(Qt.MatchContains)
        self._edit.setCompleter(completer)
        self._edit.textChanged.connect(self._update_badge)

        self._attach_badge = QLabel("")
        self._attach_badge.setObjectName("attachBadge")
        self._attach_badge.setAlignment(Qt.AlignCenter)
        self._attach_badge.setVisible(False)

        self._delete_btn = QPushButton("×")
        self._delete_btn.setObjectName("deleteAttachBtn")
        self._delete_btn.setFixedSize(26, 26)
        self._delete_btn.setToolTip("移除当前附件")
        self._delete_btn.setVisible(False)
        self._delete_btn.clicked.connect(self.clear_pending_attachments)

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
            return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
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
