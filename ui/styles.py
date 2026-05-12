"""主题颜色常量、气泡样式和内容区 QSS

QSS 只应用到 content widget，不应用到 FluentWindow 本身。
"""

from qfluentwidgets import isDarkTheme

# ========== 主色调 - Teal 系列 ==========
PRIMARY = "#0f766e"
PRIMARY_LIGHT = "#14a89a"
PRIMARY_DARK = "#0d6b64"
ACCENT = "#5ae8b8"

# ========== 语义色 ==========
SUCCESS = "#22c55e"
WARNING = "#eab308"
ERROR = "#ef4444"

# ========== 聊天气泡常量（与 Web 端一致） ==========
BUBBLE_RADIUS = 18
BUBBLE_FONT_PX = 15
BUBBLE_TITLE_FONT_PX = 11
BUBBLE_PAD = 16
AVATAR_SIZE = 38


def bubble_styles() -> dict:
    if isDarkTheme():
        return {
            "user_bg": "#0f766e",
            "user_border": "none",
            "user_title_color": "#88dcc8",
            "user_text_color": "#ffffff",
            "asst_bg": "rgba(22,38,35,0.92)",
            "asst_border": "1px solid rgba(90,138,126,0.15)",
            "asst_title_color": "#8ab8a8",
            "asst_text_color": "#d8e8e4",
            "separator_color": "rgba(90,138,126,0.30)",
            "card_bg": "rgba(22,38,35,0.92)",
            "card_border": "rgba(20,168,154,0.08)",
            "card_border_hover": "rgba(20,184,166,0.15)",
            "card_title_color": "#d8e8e4",
            "card_meta_color": "#5a8a78",
            "card_body_color": "#8ab8a8",
            "section_title_color": "#e8c878",
        }
    else:
        return {
            "user_bg": "#0f766e",
            "user_border": "none",
            "user_title_color": "#a0f0dc",
            "user_text_color": "#ffffff",
            "asst_bg": "rgba(248,248,248,0.88)",
            "asst_border": "1px solid rgba(200,200,200,0.45)",
            "asst_title_color": "#7a9a8e",
            "asst_text_color": "#1a2726",
            "separator_color": "rgba(90,138,126,0.28)",
            "card_bg": "rgba(255,255,255,0.9)",
            "card_border": "rgba(15,118,110,0.06)",
            "card_border_hover": "rgba(20,184,166,0.15)",
            "card_title_color": "#1a2726",
            "card_meta_color": "#7a9a8e",
            "card_body_color": "#4a6a62",
            "section_title_color": "#b8860b",
        }


# ========== 内容区域 QSS ==========

CONTENT_QSS_LIGHT = """
/* 主内容区背景 - 唯一使用渐变的地方 */
QWidget#mainContent {
    background: qlineargradient(x1:0.5,y1:0,x2:0.5,y2:1,
        stop:0 #e8f8f5, stop:0.2 #f0fbf8, stop:0.5 #f5fdfc,
        stop:0.8 #f0faf8, stop:1 #e4f5f0);
}

/* Header 卡片 */
QFrame#headerCard {
    background: rgba(255,255,255,0.80);
    border: 1px solid rgba(15,118,110,0.08);
    border-radius: 18px;
}

QLabel#headerBadge {
    background: #0f766e;
    color: #ffffff;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.2px;
    padding: 3px 14px;
    border-radius: 10px;
}

QLabel#headerTitle {
    color: #0a4a44;
    font-size: 26px;
    font-weight: 800;
    letter-spacing: 2px;
    padding: 0;
    margin: 0;
}

QLabel#headerSubtitle {
    color: #4a8074;
    font-size: 12px;
    font-weight: 400;
    letter-spacing: 0.3px;
    padding: 0;
    margin: 0;
}

/* 通知条 */
QWidget#updateBar {
    background: rgba(234,179,8,0.10);
    border: 1px solid rgba(234,179,8,0.18);
    border-radius: 12px;
    margin: 4px 0;
}

QLabel#updateLabel {
    color: #92600a;
    font-size: 12px;
    font-weight: 600;
}

QWidget#offlineBar {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.15);
    border-radius: 12px;
    margin: 4px 0;
}

QLabel#offlineLabel {
    color: #b91c1c;
    font-size: 12px;
    font-weight: 500;
}

/* 聊天区域 */
QWidget#chatContainer {
    background: transparent;
}

QWidget#chatInner {
    background: transparent;
}

/* 待保存条 */
QWidget#pendingBar {
    background: rgba(15,118,110,0.08);
    border: 1px solid rgba(15,118,110,0.12);
    border-radius: 12px;
    margin: 4px 0;
}

QLabel#pendingLabel {
    color: #0f766e;
    font-size: 12px;
    font-weight: 600;
}

/* 输入区域卡片 */
QFrame#inputCard {
    background: rgba(255,255,255,0.90);
    border: 1px solid rgba(15,118,110,0.08);
    border-radius: 18px;
}

/* 发送按钮 */
QPushButton#sendBtn {
    background: #0f766e;
    color: #ffffff;
    border: none;
    border-radius: 14px;
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 0.5px;
}

QPushButton#sendBtn:hover {
    background: #14a89a;
}

QPushButton#sendBtn:pressed {
    background: #0d6b64;
}

/* 状态栏 */
QWidget#customStatusBar {
    border-top: 1px solid rgba(15,118,110,0.06);
    background: rgba(232,248,245,0.50);
}

QLabel#statusLabel {
    color: #5a8a7e;
    font-size: 11px;
    font-weight: 500;
}

/* 清空按钮 */
QPushButton#headerClearBtn {
    color: #5a8a7e;
    font-size: 12px;
    font-weight: 500;
    padding: 4px 12px;
    border-radius: 8px;
    background: transparent;
    border: none;
}

QPushButton#headerClearBtn:hover {
    background: rgba(239,68,68,0.08);
    color: #dc2626;
}

/* 附件徽章 */
QLabel#attachBadge {
    background: rgba(15,118,110,0.10);
    color: #0f766e;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 8px;
}

"""

CONTENT_QSS_DARK = """
/* 主内容区背景 - 唯一使用渐变的地方 */
QWidget#mainContent {
    background: qlineargradient(x1:0.5,y1:0,x2:0.5,y2:1,
        stop:0 #0a1816, stop:0.2 #0d201d, stop:0.5 #102825,
        stop:0.8 #0d2220, stop:1 #0a1a17);
}

/* Header 卡片 */
QFrame#headerCard {
    background: rgba(16,36,32,0.88);
    border: 1px solid rgba(90,232,184,0.08);
    border-radius: 18px;
}

QLabel#headerBadge {
    background: #0f766e;
    color: #ffffff;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.2px;
    padding: 3px 14px;
    border-radius: 10px;
}

QLabel#headerTitle {
    color: #e0f5f0;
    font-size: 26px;
    font-weight: 800;
    letter-spacing: 2px;
    padding: 0;
    margin: 0;
}

QLabel#headerSubtitle {
    color: #60a898;
    font-size: 12px;
    font-weight: 400;
    letter-spacing: 0.3px;
    padding: 0;
    margin: 0;
}

/* 通知条 */
QWidget#updateBar {
    background: rgba(234,179,8,0.12);
    border: 1px solid rgba(234,179,8,0.20);
    border-radius: 12px;
    margin: 4px 0;
}

QLabel#updateLabel {
    color: #fbbf24;
    font-size: 12px;
    font-weight: 600;
}

QWidget#offlineBar {
    background: rgba(239,68,68,0.12);
    border: 1px solid rgba(239,68,68,0.18);
    border-radius: 12px;
    margin: 4px 0;
}

QLabel#offlineLabel {
    color: #fca5a5;
    font-size: 12px;
    font-weight: 500;
}

/* 聊天区域 */
QWidget#chatContainer {
    background: transparent;
}

QWidget#chatInner {
    background: transparent;
}

/* 待保存条 */
QWidget#pendingBar {
    background: rgba(90,232,184,0.08);
    border: 1px solid rgba(90,232,184,0.12);
    border-radius: 12px;
    margin: 4px 0;
}

QLabel#pendingLabel {
    color: #5ae8b8;
    font-size: 12px;
    font-weight: 600;
}

/* 输入区域卡片 */
QFrame#inputCard {
    background: rgba(16,36,32,0.90);
    border: 1px solid rgba(90,232,184,0.08);
    border-radius: 18px;
}

/* 发送按钮 */
QPushButton#sendBtn {
    background: #0f766e;
    color: #ffffff;
    border: none;
    border-radius: 14px;
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 0.5px;
}

QPushButton#sendBtn:hover {
    background: #14a89a;
}

QPushButton#sendBtn:pressed {
    background: #0d6b64;
}

/* 状态栏 */
QWidget#customStatusBar {
    border-top: 1px solid rgba(90,232,184,0.06);
    background: rgba(10,24,22,0.60);
}

QLabel#statusLabel {
    color: #4a8a7a;
    font-size: 11px;
    font-weight: 500;
}

/* 清空按钮 */
QPushButton#headerClearBtn {
    color: #4a8a7a;
    font-size: 12px;
    font-weight: 500;
    padding: 4px 12px;
    border-radius: 8px;
    background: transparent;
    border: none;
}

QPushButton#headerClearBtn:hover {
    background: rgba(239,68,68,0.12);
    color: #fca5a5;
}

/* 附件徽章 */
QLabel#attachBadge {
    background: rgba(90,232,184,0.10);
    color: #5ae8b8;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 8px;
}

/* 打字指示器 */
QLabel#typingDots {
    color: #14a89a;
    font-size: 14px;
    padding: 0;
    font-weight: 700;
    letter-spacing: 5px;
}
"""


def apply_content_stylesheet(content_widget) -> None:
    """只对内容区域应用 QSS，不影响 FluentWindow 的标题栏"""
    if isDarkTheme():
        content_widget.setStyleSheet(CONTENT_QSS_DARK)
    else:
        content_widget.setStyleSheet(CONTENT_QSS_LIGHT)
