"""暖暖记忆助手 - 主窗口 (PySide6 + PyQt-Fluent-Widgets)"""

import threading
import time
import uuid
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QTimer, QSettings, QThread, Signal
from PySide6.QtGui import QDesktopServices, QShortcut, QKeySequence, QColor
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QFrame, QHBoxLayout, QLabel,
    QMenu, QSizePolicy, QVBoxLayout, QWidget,
    QGraphicsDropShadowEffect,
)

from qfluentwidgets import (
    FluentWindow, FluentIcon as FIF, PushButton, ToolButton,
    TransparentPushButton, SmoothScrollArea, ProgressBar, RoundMenu,
    setTheme, setThemeColor, Theme, isDarkTheme,
)

from app.api_client import MemoryApiClient
from ui.widgets.chat_bubble import ChatBubble, RelatedCard
from ui.workers import VoiceWorker, init_sounddevice
from ui.dialogs import StatsDialog, ImageZoomDialog
from ui.styles import (
    bubble_styles, BUBBLE_FONT_PX, BUBBLE_TITLE_FONT_PX,
    BUBBLE_PAD, BUBBLE_RADIUS, PRIMARY,
    apply_content_stylesheet,
)

APP_VERSION = "6.0"


class AgentWindow(FluentWindow):
    """暖暖记忆助手主窗口"""

    # 跨线程 UI 更新信号
    _health_result = Signal(bool)
    _chat_result_signal = Signal(dict)
    _chat_error_signal = Signal(str)
    _upload_result_signal = Signal(dict)
    _upload_error_signal = Signal(str)
    _confirm_save_result_signal = Signal(str)
    _confirm_save_error_signal = Signal(str)
    _voice_result_signal = Signal(str, object, object)
    _voice_error_signal = Signal(str)
    _update_bar_signal = Signal(str, str)

    def __init__(self) -> None:
        # 必须在 super().__init__() 之前初始化所有实例变量，
        # 因为 Qt 事件（resizeEvent 等）会在 super().__init__() 中触发
        self.voice_thread: QThread | None = None
        self.voice_worker: VoiceWorker | None = None
        self._settings = QSettings("", "MemoryAssistant")
        self._pending_save_text: str | None = None
        self._refit_debounce_timer: QTimer | None = None
        self._chat_bubbles: list[QWidget] = []
        self._api: MemoryApiClient | None = None
        self._client_id: str = ""
        self._online: bool = False
        self._health_timer: QTimer | None = None
        self._version_timer: QTimer | None = None
        self._is_sending: bool = False
        self._typing_widget: QWidget | None = None
        self._typing_anim_timer: QTimer | None = None
        self._typing_dots_lbl: QLabel | None = None
        self._typing_anim_idx: int = 0
        self._tw_timer: QTimer | None = None
        self._tw_body: QLabel | None = None
        self._tw_plain: str = ""
        self._tw_idx: int = 0
        self._tw_speed: int = 30
        self._tw_text_color: str = ""
        self._last_msg_time: float = 0
        self._MAX_CHAT_MESSAGES = 120
        self._bubble_user_body_font_px = BUBBLE_FONT_PX
        self._bubble_user_pad_x = BUBBLE_PAD
        self._preview_base_w = 320
        self._preview_base_h = 200

        super().__init__()

        init_sounddevice()
        setThemeColor(QColor(PRIMARY))

        self._init_window()
        self._build_content()
        self._load_defaults()
        self._restore_window_state()
        self._init_backend()
        self._init_version_check()

    def _init_window(self):
        """初始化窗口基本属性"""
        self.setWindowTitle("暖暖")
        self.resize(960, 720)
        self.setMinimumSize(640, 480)

        # 居中显示在屏幕上
        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            x = (geo.width() - self.width()) // 2 + geo.x()
            y = (geo.height() - self.height()) // 2 + geo.y()
            self.move(x, y)

        # 隐藏侧边导航栏（单页面应用不需要）
        self.navigationInterface.hide()

    def _build_content(self):
        """构建主内容区域"""
        # 创建主内容 widget
        content = QWidget()
        content.setObjectName("mainContent")
        main_layout = QVBoxLayout(content)
        main_layout.setContentsMargins(16, 10, 16, 10)
        main_layout.setSpacing(0)

        # ---- 顶部信息区 ----
        main_layout.addWidget(self._build_header())

        # ---- 版本更新通知条 ----
        self._update_bar = QWidget()
        self._update_bar.setObjectName("updateBar")
        self._update_bar.setVisible(False)
        update_layout = QHBoxLayout(self._update_bar)
        update_layout.setContentsMargins(16, 6, 16, 6)
        self._update_label = QLabel()
        self._update_label.setObjectName("updateLabel")
        self._update_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self._update_label.mousePressEvent = lambda e: self._open_download_url()
        update_layout.addWidget(self._update_label, 1)
        close_update = TransparentPushButton("✕")
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

        # ---- 聊天主区域 ----
        main_layout.addWidget(self._build_chat_area(), 1)

        # ---- 自定义状态栏 ----
        main_layout.addWidget(self._build_status_bar())

        # 添加到 FluentWindow 的堆栈
        self.stackedWidget.addWidget(content)
        self.stackedWidget.setCurrentWidget(content)

        # 应用内容区样式（只对 content 生效，不覆盖 FluentWindow 标题栏）
        apply_content_stylesheet(content)

        # 欢迎消息
        self._append_assistant(
            "你好！我是暖暖，你的贴心记忆助手。\n\n"
            '你可以直接告诉我想记住什么，或像聊天一样问我"之前存过什么"。\n'
            "也支持语音输入、拖拽图片和文件。"
        )

        # 快捷键
        QShortcut(QKeySequence("Ctrl+T"), self, self._toggle_theme)
        QShortcut(QKeySequence("Ctrl+L"), self, self._clear_query)

    @staticmethod
    def _add_shadow(widget, blur=24, offset_y=4, alpha=30):
        """为 widget 添加柔和投影"""
        shadow = QGraphicsDropShadowEffect(widget)
        shadow.setBlurRadius(blur)
        shadow.setOffset(0, offset_y)
        shadow.setColor(QColor(0, 0, 0, alpha))
        widget.setGraphicsEffect(shadow)

    def _build_header(self) -> QWidget:
        """构建顶部信息区"""
        header_card = QFrame()
        header_card.setObjectName("headerCard")
        self._add_shadow(header_card, blur=28, offset_y=4, alpha=25)
        header_layout = QHBoxLayout(header_card)
        header_layout.setContentsMargins(24, 18, 24, 18)
        header_layout.setSpacing(16)

        # 左侧：标题信息
        left_col = QVBoxLayout()
        left_col.setSpacing(6)

        badge = QLabel("LOCAL MEMORY STUDIO")
        badge.setObjectName("headerBadge")
        badge.setFixedSize(190, 22)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title = QLabel("暖暖")
        title.setObjectName("headerTitle")

        subtitle = QLabel("把零散念头、语音和附件整理成可搜索的私人记忆库")
        subtitle.setObjectName("headerSubtitle")

        left_col.addWidget(badge, 0, Qt.AlignmentFlag.AlignLeft)
        left_col.addWidget(title, 0, Qt.AlignmentFlag.AlignLeft)
        left_col.addWidget(subtitle, 0, Qt.AlignmentFlag.AlignLeft)

        # 右侧：功能按钮
        right_col = QHBoxLayout()
        right_col.setSpacing(8)

        # 主题切换
        self._theme_btn = ToolButton(FIF.CONSTRACT)
        self._theme_btn.setFixedSize(36, 36)
        self._theme_btn.setToolTip("切换暗色/亮色模式 (Ctrl+T)")
        self._theme_btn.clicked.connect(self._toggle_theme)
        right_col.addWidget(self._theme_btn)

        # 统计
        stats_btn = ToolButton(FIF.CHAT)
        stats_btn.setFixedSize(36, 36)
        stats_btn.setToolTip("查看记忆统计")
        stats_btn.clicked.connect(self._show_stats)
        right_col.addWidget(stats_btn)

        # 清空对话
        self.clear_btn = TransparentPushButton("清空对话")
        self.clear_btn.setObjectName("headerClearBtn")
        self.clear_btn.clicked.connect(self._clear_query)
        right_col.addWidget(self.clear_btn)

        header_layout.addLayout(left_col, 1)
        header_layout.addLayout(right_col, 0)
        return header_card

    def _build_chat_area(self) -> QWidget:
        """构建聊天消息区域"""
        chat_container = QWidget()
        chat_container.setObjectName("chatContainer")
        chat_layout = QVBoxLayout(chat_container)
        chat_layout.setContentsMargins(4, 6, 4, 6)
        chat_layout.setSpacing(10)

        self.chat_view = SmoothScrollArea()
        self.chat_view.setObjectName("chatView")
        self.chat_view.setStyleSheet("background:transparent;")
        self.chat_view.setWidgetResizable(True)
        self.chat_view.setFrameShape(QFrame.Shape.NoFrame)
        self.chat_view.viewport().setStyleSheet("background:transparent;")

        self._chat_inner = QWidget()
        self._chat_inner.setObjectName("chatInner")
        self._chat_layout = QVBoxLayout(self._chat_inner)
        self._chat_layout.setContentsMargins(10, 10, 10, 10)
        self._chat_layout.setSpacing(14)
        self._chat_layout.addStretch(1)
        self._chat_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.chat_view.setWidget(self._chat_inner)
        chat_layout.addWidget(self.chat_view, 1)

        # 待保存指示条
        self._pending_bar = QWidget()
        self._pending_bar.setObjectName("pendingBar")
        self._pending_bar.setVisible(False)
        pending_layout = QHBoxLayout(self._pending_bar)
        pending_layout.setContentsMargins(16, 6, 16, 6)
        pending_lbl = QLabel("⏳ 等待补充内容...")
        pending_lbl.setObjectName("pendingLabel")
        pending_layout.addWidget(pending_lbl, 1)
        chat_layout.addWidget(self._pending_bar)

        # ---- 输入区域 ----
        chat_layout.addWidget(self._build_input_area())

        # 右键菜单
        self.chat_view.viewport().setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.chat_view.viewport().customContextMenuRequested.connect(self._show_chat_context_menu)

        # 事件过滤器（拦截气泡内滚轮事件）
        self.chat_view.viewport().installEventFilter(self)

        return chat_container

    def _build_input_area(self) -> QWidget:
        """构建输入区域"""
        outer = QFrame()
        outer.setObjectName("inputCard")
        outer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._add_shadow(outer, blur=20, offset_y=3, alpha=20)
        input_layout = QHBoxLayout(outer)
        input_layout.setContentsMargins(16, 10, 12, 10)
        input_layout.setSpacing(10)

        # 附件按钮
        self._attach_btn = ToolButton(FIF.ADD)
        self._attach_btn.setFixedSize(40, 40)
        self._attach_btn.setToolTip("选择文件上传")
        self._attach_btn.clicked.connect(self._choose_files)
        input_layout.addWidget(self._attach_btn)

        # 命令输入框
        from ui.widgets.command_input import CommandInput, CommandSubmit, DropSubmit
        self.command_input = CommandInput()
        self.command_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.command_input.submitted_text.connect(self._handle_chat_submit_text)
        self.command_input.submitted_drop.connect(self._handle_chat_submit_drop)
        input_layout.addWidget(self.command_input, 1)

        # 语音按钮
        self.voice_btn = ToolButton(FIF.MICROPHONE)
        self.voice_btn.setFixedSize(40, 40)
        self.voice_btn.setToolTip("语音对话")
        self.voice_btn.clicked.connect(self._toggle_voice_dialogue)
        input_layout.addWidget(self.voice_btn)

        # 发送按钮
        self.send_btn = PushButton("发送")
        self.send_btn.setObjectName("sendBtn")
        self.send_btn.setFixedSize(68, 40)
        self.send_btn.setToolTip("发送 (Enter)")
        self.send_btn.clicked.connect(self._submit_command_input)
        input_layout.addWidget(self.send_btn)

        return outer

    def _build_status_bar(self) -> QWidget:
        """构建状态栏"""
        bar = QWidget()
        bar.setObjectName("customStatusBar")
        bar.setFixedHeight(36)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(20, 0, 20, 0)

        self._status_label = QLabel("准备就绪")
        self._status_label.setObjectName("statusLabel")
        layout.addWidget(self._status_label, 1)

        self._progress = ProgressBar()
        self._progress.setVisible(False)
        self._progress.setMaximumWidth(200)
        self._progress.setRange(0, 0)
        layout.addWidget(self._progress)

        return bar

    # ========== 窗口事件 ==========

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, '_refit_debounce_timer'):
            self._schedule_refit_chat_bubbles()

    def closeEvent(self, event) -> None:
        try:
            if hasattr(self, 'command_input'):
                self._settings.setValue("query/last", self.command_input.text().strip())
            self._settings.setValue("window/geometry", self.saveGeometry())
        finally:
            if self._health_timer:
                self._health_timer.stop()
            if self._version_timer:
                self._version_timer.stop()
            super().closeEvent(event)

    def eventFilter(self, obj, event) -> bool:
        # Only filter events for the chat viewport
        if hasattr(self, 'chat_view') and obj is self.chat_view.viewport():
            etype = event.type()
            if etype == 31:  # Wheel
                return False  # Let SmoothScrollArea handle it
        return super().eventFilter(obj, event)

    # ========== 主题切换 ==========

    def _toggle_theme(self):
        if isDarkTheme():
            setTheme(Theme.LIGHT)
            self._settings.setValue("theme/mode", "light")
        else:
            setTheme(Theme.DARK)
            self._settings.setValue("theme/mode", "dark")
        # 刷新内容区样式
        content = self.stackedWidget.currentWidget()
        if content:
            apply_content_stylesheet(content)
        self._refresh_bubble_styles()

    def _refresh_bubble_styles(self):
        """主题切换后刷新所有已有气泡的颜色"""
        s = bubble_styles()
        for bubble in self._chat_bubbles:
            if isinstance(bubble, ChatBubble):
                is_user = bubble.property("bubbleAlign") == "right"
                # 更新气泡体颜色
                body = bubble.body()
                if is_user:
                    body.set_style(s["user_bg"], s["user_text_color"], BUBBLE_RADIUS, s["user_border"])
                else:
                    body.set_style(s["asst_bg"], s["asst_text_color"], BUBBLE_RADIUS, s["asst_border"])
                # 更新标题颜色
                name_lbl = getattr(bubble, "_name_lbl", None)
                if name_lbl:
                    name_lbl.setStyleSheet(
                        f"color:{s['asst_title_color']}; font-size:{BUBBLE_TITLE_FONT_PX}px;"
                        f"font-weight:700; margin-bottom:3px;"
                    )
                # 更新头像颜色
                row_layout = bubble.layout()
                if row_layout:
                    for i in range(row_layout.count()):
                        item = row_layout.itemAt(i)
                        if item and item.widget():
                            w = item.widget()
                            ss = w.styleSheet()
                            if "border-radius:19px" in ss:
                                bg = "#5a8a7a" if is_user else "#0f766e"
                                w.setStyleSheet(
                                    f"background:{bg}; color:#ffffff; border-radius:19px;"
                                    f"font-size:15px; font-weight:800;"
                                )
                                break
            elif isinstance(bubble, QLabel):
                # 时间分隔标签
                bubble.setStyleSheet(
                    f"color:{s['separator_color']}; font-size:11px; padding:6px 0; font-weight:500;"
                )

    # ========== 聊天消息渲染 ==========

    def _scroll_chat_to_bottom(self) -> None:
        bar = self.chat_view.verticalScrollBar()
        bar.setSliderPosition(bar.maximum())
        bar.setValue(bar.maximum())

    def _responsive_params(self) -> dict[str, int | float]:
        try:
            w = max(420, int(self.chat_view.viewport().width()))
        except Exception:
            w = max(420, int(self.width()))
        preview_w = int(min(self._preview_base_w, max(220, w * 0.28)))
        preview_h = int(preview_w * 0.62)
        return {"avail": w, "preview_w": preview_w, "preview_h": preview_h}

    def _add_chat_widget(self, w: QWidget) -> None:
        self._chat_inner.setUpdatesEnabled(False)

        idx = max(0, self._chat_layout.count() - 1)
        if w.property("bubbleAlign") == "right":
            row = QWidget()
            row.setProperty("bubbleAlign", "right")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(0)
            row_layout.addStretch(1)
            row_layout.addWidget(w, 0)
            w.setProperty("_chat_wrapper", row)
            self._chat_layout.insertWidget(idx, row)
        else:
            self._chat_layout.insertWidget(idx, w)
        self._chat_bubbles.append(w)

        # 超出上限时移除最早的消息
        while len(self._chat_bubbles) > self._MAX_CHAT_MESSAGES:
            old = self._chat_bubbles.pop(0)
            wrapper = old.property("_chat_wrapper")
            remove_target = wrapper if wrapper else old
            self._chat_layout.removeWidget(remove_target)
            remove_target.setParent(None)
            remove_target.deleteLater()

        self._chat_inner.setUpdatesEnabled(True)
        QTimer.singleShot(0, self._scroll_chat_to_bottom)

    def _maybe_add_time_separator(self) -> None:
        now = time.time()
        if now - self._last_msg_time < 300 and self._last_msg_time > 0:
            return
        self._last_msg_time = now
        t = time.localtime(now)
        if time.localtime().tm_mday == t.tm_mday and time.localtime().tm_mon == t.tm_mon:
            label = time.strftime("%H:%M", t)
        else:
            label = time.strftime("%m月%d日 %H:%M", t)
        sep = QLabel(label)
        sep.setAlignment(Qt.AlignmentFlag.AlignCenter)
        s = bubble_styles()
        sep.setStyleSheet(f"color:{s['separator_color']}; font-size:11px; padding:6px 0; font-weight:500;")
        sep.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        sep.setFixedHeight(24)
        idx = max(0, self._chat_layout.count() - 1)
        self._chat_layout.insertWidget(idx, sep)
        self._chat_bubbles.append(sep)

    def _append_user(self, text: str) -> None:
        s = bubble_styles()
        self._maybe_add_time_separator()
        rp = self._responsive_params()
        w = ChatBubble(
            align_right=True, title="", plain_text=text,
            bg=s["user_bg"], border=s["user_border"],
            title_color=s["user_title_color"], text_color=s["user_text_color"],
            show_title=False, pad_x=self._bubble_user_pad_x,
            body_font_px=self._bubble_user_body_font_px,
        )
        w.refit(int(rp["avail"]))
        self._add_chat_widget(w)

    def _append_assistant(self, text: str, typewriter: bool = False) -> None:
        s = bubble_styles()
        self._maybe_add_time_separator()
        rp = self._responsive_params()

        w = ChatBubble(
            align_right=False, title="暖暖", plain_text=text,
            bg=s["asst_bg"], border=s["asst_border"],
            title_color=s["asst_title_color"], text_color=s["asst_text_color"],
            show_title=True,
        )
        w.refit(int(rp["avail"]))

        if typewriter and len(text) > 1:
            body = w.body()
            self._start_typewriter(body, text, s["asst_text_color"])

        self._add_chat_widget(w)

    def _append_cards(self, items: list[dict[str, Any]]) -> None:
        if not items:
            return

        s = bubble_styles()
        container = QWidget()
        container.setObjectName("relatedContainer")
        v = QVBoxLayout(container)
        v.setContentsMargins(0, 2, 0, 2)
        v.setSpacing(6)

        title_lbl = QLabel("相关记忆")
        title_lbl.setStyleSheet(
            f"color:{s['section_title_color']}; font-size:12px; font-weight:700; letter-spacing:0.05em;"
        )
        title_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(title_lbl, 0, Qt.AlignmentFlag.AlignLeft)

        rp = self._responsive_params()
        for it in items[:8]:
            v.addWidget(RelatedCard(it, int(rp["preview_w"]), int(rp["preview_h"])))

        container.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        outer = QWidget()
        row = QHBoxLayout(outer)
        row.setContentsMargins(0, 4, 0, 4)
        row.setSpacing(6)
        row.addWidget(container, 0, Qt.AlignmentFlag.AlignLeft)
        row.addStretch(1)
        self._add_chat_widget(outer)

    # ========== 打字机效果 ==========

    def _start_typewriter(self, body: QLabel, plain: str, text_color: str) -> None:
        self._stop_typewriter()
        self._tw_body = body
        self._tw_plain = plain
        self._tw_text_color = text_color
        self._tw_idx = 0
        body.setText("")
        self._tw_timer = QTimer(self)
        self._tw_timer.setInterval(self._tw_speed)
        self._tw_timer.timeout.connect(self._typewriter_tick)
        self._tw_timer.start()

    def _typewriter_tick(self) -> None:
        if self._tw_body is None or self._tw_idx >= len(self._tw_plain):
            self._stop_typewriter()
            return
        step = 3 if len(self._tw_plain) > 100 else (2 if len(self._tw_plain) > 30 else 1)
        self._tw_idx = min(self._tw_idx + step, len(self._tw_plain))
        shown = self._tw_plain[:self._tw_idx]
        self._tw_body.setText(shown)
        QTimer.singleShot(0, self._scroll_chat_to_bottom)

    def _stop_typewriter(self) -> None:
        if self._tw_timer:
            self._tw_timer.stop()
            self._tw_timer.deleteLater()
            self._tw_timer = None
        if self._tw_body and self._tw_plain:
            self._tw_body.setText(self._tw_plain)
        self._tw_body = None
        self._tw_plain = ""

    # ========== 气泡重排 ==========

    def _schedule_refit_chat_bubbles(self) -> None:
        if self._refit_debounce_timer is None:
            self._refit_debounce_timer = QTimer(self)
            self._refit_debounce_timer.setSingleShot(True)
            self._refit_debounce_timer.timeout.connect(self._refit_chat_bubbles)
        self._refit_debounce_timer.start(200)

    def _refit_chat_bubbles(self) -> None:
        try:
            if not hasattr(self, "_chat_inner"):
                return
            rp = self._responsive_params()
            avail = int(rp["avail"])
            recent = self._chat_bubbles[-8:] if len(self._chat_bubbles) > 8 else self._chat_bubbles
            for bubble in recent:
                if isinstance(bubble, ChatBubble):
                    bubble.refit(avail)
        except Exception:
            return

    # ========== 输入处理 ==========

    def _submit_command_input(self) -> None:
        self.command_input.submit()

    def _handle_chat_submit_text(self, payload) -> None:
        text = (payload.text or "").strip()
        if not text:
            return
        if self._is_sending:
            return
        self._stop_typewriter()
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

    def _handle_chat_submit_drop(self, payload) -> None:
        paths = payload.paths or []
        caption = (payload.caption or "").strip()
        shown = caption or (
            "拖拽附件：" + "；".join(Path(p).name for p in paths[:3])
            + ("…" if len(paths) > 3 else "")
        )
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

    def _choose_files(self):
        from ui.widgets.command_input import DropSubmit
        paths, _ = QFileDialog.getOpenFileNames(self, "选择文件", "", "所有文件 (*)")
        if paths:
            self._handle_chat_submit_drop(DropSubmit(paths=paths, caption=""))

    # ========== 聊天结果处理 ==========

    def _process_chat_result(self, result: dict) -> None:
        self._hide_typing_indicator()
        self._is_sending = False
        text = result.get("text", "")
        if text:
            self._append_assistant(text, typewriter=True)
        results = result.get("results")
        if results:
            self._append_cards(results)
        pending = result.get("pending_save")
        if pending:
            self._pending_save_text = pending
            self._pending_bar.setVisible(True)
        self._set_status("准备就绪")

    def _handle_chat_error(self, error: str) -> None:
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

    def _process_upload_result(self, result: dict) -> None:
        if result.get("success"):
            self._append_assistant(result.get("message", "文件已保存。"))
        elif result.get("error"):
            self._append_assistant(f"文件上传失败：{result['error']}")
        self._set_status("准备就绪")

    # ========== 待确认保存 ==========

    def _resolve_pending_save(self, text: str) -> str | None:
        if self._pending_save_text is None:
            return None
        pending = self._pending_save_text
        self._pending_save_text = None
        self._pending_bar.setVisible(False)

        def _do():
            try:
                result = self._api.confirm_save(text, pending)
                self._confirm_save_result_signal.emit(result.get("text", "好的，已处理。"))
            except Exception as e:
                self._confirm_save_error_signal.emit(str(e))

        self._show_typing_indicator()
        self._set_status("正在处理保存确认…")
        threading.Thread(target=_do, daemon=True).start()
        return ""

    def _on_confirm_save_result(self, text: str) -> None:
        self._hide_typing_indicator()
        self._append_assistant(text, typewriter=True)
        self._set_status("准备就绪")
        if self.voice_thread and self.voice_thread.isRunning():
            self.voice_worker.speak_text(text)

    def _on_confirm_save_error(self, error: str) -> None:
        self._hide_typing_indicator()
        self._append_assistant(f"处理失败：{error}")
        self._set_status("准备就绪")

    # ========== 打字指示器 ==========

    def _show_typing_indicator(self) -> None:
        self._hide_typing_indicator()
        from ui.widgets.chat_bubble import _make_avatar
        w = QWidget()
        w.setProperty("bubbleAlign", "left")
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 4, 0, 4)
        row.setSpacing(10)

        avatar = _make_avatar("暖", is_user=False)

        dark = isDarkTheme()
        dots = QLabel("● ● ●")
        dots.setObjectName("typingDots")
        dots.setStyleSheet(
            f"background:{'rgba(22,38,35,0.92)' if dark else 'rgba(255,255,255,0.85)'};"
            f"color:{'#14a89a' if dark else '#0f766e'};"
            f"border:none; border-radius:18px;"
            f"padding:16px 20px; font-size:14px; font-weight:700; letter-spacing:5px;"
        )

        row.addWidget(avatar, 0, Qt.AlignmentFlag.AlignTop)
        row.addWidget(dots)
        row.addStretch(1)

        idx = max(0, self._chat_layout.count() - 1)
        self._chat_layout.insertWidget(idx, w)
        self._typing_widget = w
        QTimer.singleShot(10, self._scroll_chat_to_bottom)

        # 脉冲动画：循环切换点的亮度
        self._typing_dots_lbl = dots
        self._typing_anim_idx = 0
        self._typing_anim_timer = QTimer(self)
        self._typing_anim_timer.setInterval(400)
        self._typing_anim_timer.timeout.connect(self._animate_typing_dots)
        self._typing_anim_timer.start()

    def _animate_typing_dots(self) -> None:
        if not hasattr(self, '_typing_dots_lbl') or self._typing_dots_lbl is None:
            return
        self._typing_anim_idx = (self._typing_anim_idx + 1) % 4
        patterns = ["● ● ●", "◉ ● ●", "● ◉ ●", "● ● ◉"]
        self._typing_dots_lbl.setText(patterns[self._typing_anim_idx])

    def _hide_typing_indicator(self) -> None:
        if hasattr(self, '_typing_anim_timer') and self._typing_anim_timer:
            self._typing_anim_timer.stop()
            self._typing_anim_timer.deleteLater()
            self._typing_anim_timer = None
        self._typing_dots_lbl = None
        if self._typing_widget is not None:
            self._chat_layout.removeWidget(self._typing_widget)
            self._typing_widget.setParent(None)
            self._typing_widget.deleteLater()
            self._typing_widget = None

    # ========== 语音 ==========

    def _toggle_voice_dialogue(self) -> None:
        if self.voice_worker is not None:
            self.voice_worker.stop()
            return
        if not self._online:
            self._append_assistant("当前处于离线状态，无法使用语音功能。")
            return
        self._append_assistant("启动语音对话模式，说'退出'可结束对话。")
        self._start_voice_dialogue()

    def _start_voice_dialogue(self) -> None:
        if self.voice_worker is not None:
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
            if self.voice_worker:
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
                self.voice_btn.setToolTip("停止录音")
            else:
                self.voice_btn.setText("")
                self.voice_btn.setIcon(FIF.MICROPHONE)
                self.voice_btn.setToolTip("语音对话")

    def _handle_dialogue_stopped(self) -> None:
        if hasattr(self, "voice_btn"):
            self.voice_btn.setText("")
            self.voice_btn.setIcon(FIF.MICROPHONE)
            self.voice_btn.setToolTip("语音对话")
        self._set_status("语音对话已结束")

    def _cleanup_voice_worker(self) -> None:
        self.voice_thread = None
        self.voice_worker = None

    # ========== 右键菜单 ==========

    def _show_chat_context_menu(self, position) -> None:
        menu = RoundMenu(self)
        menu.addAction("清空对话", self._clear_query)
        menu.exec(self.chat_view.viewport().mapToGlobal(position))

    # ========== 统计和设置 ==========

    def _show_stats(self):
        if self._api is None:
            return
        dlg = StatsDialog(self._api, self)
        dlg.exec()

    # ========== 状态栏 ==========

    def _set_status(self, text: str, is_error: bool = False) -> None:
        if hasattr(self, '_status_label'):
            self._status_label.setText(text)

    def _clear_query(self) -> None:
        self.command_input.clear()
        self._hide_typing_indicator()
        self._stop_typewriter()
        self._is_sending = False
        self._last_msg_time = 0
        for w in self._chat_bubbles:
            wrapper = w.property("_chat_wrapper")
            remove_target = wrapper if wrapper else w
            self._chat_layout.removeWidget(remove_target)
            remove_target.setParent(None)
            remove_target.deleteLater()
        self._chat_bubbles.clear()
        self._append_assistant("内容已清空。")
        self._set_status("内容已清空。")
        self._settings.setValue("query/last", "")

    def _load_defaults(self) -> None:
        last_query = self._settings.value("query/last", "", type=str) or ""
        if last_query:
            self.command_input.set_text(last_query)

    def _restore_window_state(self) -> None:
        geo = self._settings.value("window/geometry")
        if geo is not None:
            self.restoreGeometry(geo)

    # ========== 后端初始化 ==========

    def _init_backend(self) -> None:
        self._client_id = self._settings.value("client/id", "", type=str) or ""
        if not self._client_id:
            self._client_id = uuid.uuid4().hex[:16]
            self._settings.setValue("client/id", self._client_id)

        server_url = "http://127.0.0.1:5000"
        self._api = MemoryApiClient(base_url=server_url, client_id=self._client_id, timeout=60)

        # 加载暗色模式偏好
        saved_theme = self._settings.value("theme/mode", "light", type=str)
        if saved_theme == "dark" and not isDarkTheme():
            setTheme(Theme.DARK)
        elif saved_theme == "light" and isDarkTheme():
            setTheme(Theme.LIGHT)

        # 跨线程信号连接
        self._health_result.connect(self._handle_health_result)
        self._chat_result_signal.connect(self._process_chat_result)
        self._chat_error_signal.connect(self._handle_chat_error)
        self._upload_result_signal.connect(self._process_upload_result)
        self._upload_error_signal.connect(lambda e: self._append_assistant(f"文件上传失败了：{e}"))
        self._voice_result_signal.connect(self._process_voice_result)
        self._voice_error_signal.connect(lambda e: self._append_assistant(f"语音处理失败了：{e}"))
        self._update_bar_signal.connect(self._show_update_bar)
        self._confirm_save_result_signal.connect(self._on_confirm_save_result)
        self._confirm_save_error_signal.connect(self._on_confirm_save_error)

        # 健康检查
        self._health_fail_count = 0
        self._health_base_interval = 15000
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
        if ok and not was_online:
            self._set_status("已恢复连接 ✓")
            self._offline_bar.setVisible(False)
            self._health_fail_count = 0
            self._health_timer.setInterval(self._health_base_interval)
        elif ok:
            self._health_fail_count = 0
            self._health_timer.setInterval(self._health_base_interval)
        elif not ok:
            self._health_fail_count += 1
            backoff = min(120000, self._health_base_interval * (2 ** (self._health_fail_count - 1)))
            self._health_timer.setInterval(backoff)
            if not was_online or self._health_fail_count == 1:
                self._offline_bar.setVisible(True)

    # ========== 版本检查 ==========

    def _init_version_check(self):
        self._check_version()
        self._version_timer = QTimer(self)
        self._version_timer.timeout.connect(self._check_version)
        self._version_timer.start(6 * 3600 * 1000)

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
            QDesktopServices.openUrl(url)
