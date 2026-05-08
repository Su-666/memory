# 暖暖记忆助手 v6.0

一个基于大语言模型的个人记忆管理助手，支持桌面端和 Web 端两种使用模式。用户可以通过文字或语音与"暖暖"对话，保存想法、联系人、备忘等内容到本地数据库，并随时搜索回忆。

## 功能特性

### 核心能力
- **智能意图识别** - 自动判断用户是要保存记忆、搜索记忆还是闲聊
- **记忆管理** - 保存、搜索、编辑、删除记忆，支持标签分类
- **全文检索** - 基于 SQLite FTS5 的中文全文搜索，带相关性评分
- **图片理解** - 上传图片自动生成描述、标签和 OCR 文字提取
- **语音交互** - 语音识别 + 语音合成，支持语音对话模式
- **联网搜索** - 对话模式下大模型可联网获取实时信息
- **文件保险库** - 附件和文本记忆统一归档到 `memory_vault` 目录
- **暗色模式** - 桌面端和 Web 端均支持亮色/暗色主题切换
- **在线更新检查** - 桌面端启动时自动检查新版本

### 两种运行模式

| 模式 | 入口文件 | 说明 |
|------|----------|------|
| 桌面 GUI | `pyqt_local_agent.py` | PyQt5 桌面应用，连接 Railway 服务器 |
| Web 服务 | `web/main.py` | Flask 后端 + 单页 HTML 前端，部署到 Railway |

## 技术栈

| 组件 | 技术 |
|------|------|
| 桌面端 | PyQt5（无边框窗口、暗色模式、动画） |
| Web 后端 | Flask + Flask-CORS + Gunicorn |
| Web 前端 | 原生 HTML/CSS/JS 单页应用（响应式） |
| 数据库 | SQLite + FTS5 全文搜索 |
| 大模型 | 智谱 AI（GLM-4-Flash 对话，GLM-4V-Flash 图片理解） |
| 语音 | 百度 AIP（服务端 ASR + TTS） |
| 通信 | HTTP API（urllib） |

## 项目结构

```
记忆助手/
├── pyqt_local_agent.py      # 桌面端主程序（纯在线模式，连接 Railway 服务器）
├── build_exe.py              # PyInstaller 打包脚本
├── requirements.txt          # Python 依赖
├── .env                      # API 密钥（不提交到 Git）
├── Procfile                  # Railway 部署入口
├── railway.toml              # Railway 部署配置
│
├── app/                      # 核心业务逻辑（桌面端和 Web 端共用）
│   ├── db.py                 #   数据库初始化（memories 表、FTS5、触发器）
│   ├── repo.py               #   记忆 CRUD、标签提取、保险库导入
│   ├── search.py             #   FTS5 全文搜索 + LIKE 回退 + 评分排序
│   ├── answer.py             #   基于记忆的回答生成（LLM + 本地规则）
│   ├── intent.py             #   LLM 意图规划
│   ├── intent_chat.py        #   统一意图路由（保存/搜索/聊天）
│   ├── llm.py                #   智谱大模型调用（支持联网搜索）
│   ├── vision.py             #   图片理解（描述、标签、OCR）
│   ├── vault.py              #   文件保险库（附件导入、Markdown 写入）
│   └── api_client.py         #   HTTP API 客户端（桌面端→服务端通信）
│
├── ui/                       # 桌面端 UI 组件
│   └── widgets/
│       └── command_input.py  #   输入框组件（拖拽、自动补全）
│
└── web/                      # Web 部署目录
    ├── main.py               #   Flask 后端（全部 API 路由）
    ├── index.html            #   单页前端（聊天、搜索、暗色模式）
    ├── import_setup.py       #   路径修复（Railway 部署用）
    └── requirements.txt      #   Web 端依赖
```

## 快速开始

### 环境要求

- Python 3.11+
- Windows / macOS / Linux

### 1. 安装依赖

```bash
# Web 端依赖
pip install -r requirements.txt

# 桌面端额外依赖
pip install PyQt5
```

### 2. 配置 API 密钥

在项目根目录创建 `.env` 文件：

```env
# 智谱 AI（必填 - 对话和图片理解）
ZHIPU_API_KEY=your_zhipu_api_key
LOCAL_AGENT_MODEL=glm-4-flash-250414

# 百度语音（选填 - 语音功能需要）
BAIDU_APP_ID=your_app_id
BAIDU_API_KEY=your_baidu_api_key
BAIDU_SECRET_KEY=your_baidu_secret_key
```

获取方式：
- 智谱 AI：https://open.bigmodel.cn/ 注册后在 API Keys 页面获取
- 百度语音：https://ai.baidu.com/ 创建应用后获取

### 3. 启动运行

**Web 模式（本地开发）：**

```bash
cd web
python main.py
```

浏览器打开 http://127.0.0.1:5000 即可使用。

**桌面 GUI 模式：**

```bash
python pyqt_local_agent.py
```

桌面端连接 Railway 服务器（`https://memory-n.ccwu.cc`），也可本地启动 Web 服务后连接 `http://localhost:5000`。

## 使用方式

### 保存记忆

直接用自然语言告诉暖暖要记住的内容：

- "帮我记住 WiFi 密码是 12345678"
- "记一下：张三的电话是 13800138000"
- "提醒我明天下午3点开会"

### 搜索记忆

- "帮我找一下 WiFi 密码"
- "查一下张三的联系方式"
- "之前记过的那个会议是什么时候"

### 闲聊对话

- "你好"
- "今天天气怎么样"
- "给我讲个故事"

### 图片理解

拖拽、粘贴或点击📎上传图片，暖暖会自动识别图片内容并生成可搜索的描述和标签。

## v6.0 新特性

- **全新 UI 设计** - 参考主流聊天应用，消息气泡带头像和时间戳
- **无边框窗口** - 桌面端自定义标题栏，支持拖拽移动
- **暗色模式** - 亮色/暗色主题一键切换，设置持久化
- **统计面板** - 查看记忆总数、热门标签、存储信息
- **图片放大** - 搜索结果中的图片可点击查看大图
- **复制记忆** - 每条搜索结果支持一键复制
- **在线更新检查** - 启动时自动检查新版本通知
- **附件上传** - 点击📎按钮选择文件上传
- **待保存指示** - 输入区下方显示等待补充内容提示

### 已移除功能

- 离线模式（桌面端必须连接服务器）
- 命令行模式（`--cli` / `--once`）
- 唤醒词检测（"你好暖暖"）
- 本地语音处理（ASR/TTS 改为服务端处理）
- 记忆库文件夹操作
- 聊天记录导出

## 打包为 EXE

```bash
python build_exe.py
```

生成的 `dist/记忆助手.exe` 为单文件可执行程序，需与 `data/` 目录一起分发。

## 部署到 Railway

1. 将代码推送到 Git 仓库
2. 在 Railway 中连接仓库
3. 在 Variables 页面添加环境变量：
   - `ZHIPU_API_KEY` - 智谱 AI API Key
   - `BAIDU_APP_ID` / `BAIDU_API_KEY` / `BAIDU_SECRET_KEY` - 百度语音（可选）
4. Railway 自动检测 `Procfile` 并部署

### 可选环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LOCAL_AGENT_MODEL` | 对话模型 | `GLM-4-Flash-250414` |
| `LOCAL_AGENT_VISION_MODEL` | 图片理解模型 | `glm-4v-flash` |
| `PORT` | 服务端口 | `5000` |
| `DATA_DIR` | 数据目录 | `./data` |

## API 接口

Web 模式提供以下 REST API：

### 对话与记忆

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/chat` | 发送聊天消息（自动判断意图） |
| POST | `/api/chat/confirm_save` | 确认保存记忆 |
| POST | `/api/save` | 直接保存记忆 |
| POST | `/api/search` | 搜索记忆 |
| POST | `/api/upload` | 上传文件/图片 |

### 记忆管理

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/memories` | 获取记忆列表 |
| GET | `/api/memories/recent` | 获取最近记忆 |
| POST | `/api/memories/search` | 高级搜索（标签、日期范围） |
| GET | `/api/memory/<id>` | 获取记忆详情 |
| PUT | `/api/memory/<id>` | 更新记忆 |
| DELETE | `/api/memory/<id>` | 删除记忆 |
| GET | `/api/tags` | 获取所有标签 |
| GET | `/api/stats` | 获取统计信息 |

### 语音

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/speech_recognize` | 语音识别（ASR） |
| POST | `/api/speech_synthesize` | 语音合成（TTS） |
| POST | `/api/voice_dialogue` | 语音对话（识别 + 回复 + 合成） |

### 其他

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查 |
| GET | `/api/vault/path` | 获取保险库路径 |
| POST | `/api/clear` | 清空对话历史 |
| GET | `/api/file/image` | 图片预览 |

## 架构设计

```
用户输入（文字/语音/图片）
        │
        ▼
   意图判断 (intent_chat.py)
        │
   ┌────┼────┐
   ▼    ▼    ▼
 保存  搜索  聊天
   │    │    │
   ▼    ▼    ▼
 repo  search  llm.py
   │    │    │
   ▼    ▼    │
 SQLite+FTS5  智谱 AI API
   │         │
   ▼         ▼
 回答生成 (answer.py) ← 记忆上下文
        │
        ▼
   Flask API 服务端 (web/main.py)
        │
   ┌────┴────┐
   ▼         ▼
 桌面端     Web 前端
(API客户端)  (浏览器)
```

### 关键设计

- **纯在线架构**：桌面端通过 `MemoryApiClient` 调用 Flask 后端 API，断线时自动重连
- **共享业务逻辑** (`app/`)：桌面端和 Web 端共用同一套核心代码，避免逻辑重复
- **FTS5 + LIKE 双重搜索**：优先使用 FTS5 全文索引，索引不可用时回退到 LIKE 模糊匹配
- **记忆保险库** (`memory_vault/`)：所有记忆以 Markdown 文件归档，便于备份和迁移
- **服务端语音处理**：ASR/TTS 通过百度 AIP 在服务端完成，桌面端无需安装音频库

## 许可证

私人项目，仅供个人使用。
