# 暖暖记忆助手 v6.0

一个基于大语言模型的个人记忆管理助手，支持 Web 端和桌面端两种使用方式。用户可以通过文字、语音或图片与"暖暖"对话，保存想法、联系人、备忘等内容，并随时搜索回忆。

## 核心能力

- **智能意图识别** — 自动判断保存、搜索、闲聊三种意图
- **全文检索** — SQLite FTS5 中文全文搜索 + BM25 相关性排序，自动回退 LIKE 模糊匹配
- **图片理解** — 上传图片自动生成描述、标签和 OCR 文字提取（GLM-4V）
- **语音交互** — 百度 ASR/TTS，支持连续语音对话模式
- **文件保险库** — 所有记忆以 Markdown 文件归档到 `memory_vault/` 目录
- **管理员后台** — 独立 SPA 管理页面，支持数据查看、搜索、删除、导出和备份
- **暗色模式** — Web 端和桌面端均支持亮色/暗色主题切换

## 技术栈

| 组件 | 技术 |
|------|------|
| Web 后端 | Flask + Gunicorn + Flask-CORS |
| Web 前端 | 原生 HTML/CSS/JS 单页应用 |
| 桌面端 | PyQt5（无边框窗口、暗色模式、动画） |
| 数据库 | SQLite + FTS5 全文索引 |
| 大模型 | 智谱 AI（GLM-4-Flash 对话，GLM-4V-Flash 图片理解） |
| 语音 | 百度 AIP（服务端 ASR + TTS） |

## 项目结构

```
记忆助手/
├── pyqt_local_agent.py      # 桌面端主程序
├── build_exe.py              # PyInstaller 打包脚本
├── requirements.txt          # Python 依赖
├── Procfile                  # Railway 部署入口
├── .env                      # API 密钥（不提交到 Git）
│
├── app/                      # 核心业务逻辑（桌面端和 Web 端共用）
│   ├── db.py                 #   数据库连接与表结构初始化
│   ├── repo.py               #   记忆 CRUD、标签提取、保险库导入
│   ├── search.py             #   FTS5 全文搜索 + LIKE 回退 + 评分排序
│   ├── answer.py             #   基于记忆的回答生成（LLM + 本地规则）
│   ├── intent.py             #   LLM 意图规划
│   ├── intent_chat.py        #   统一意图路由（保存/搜索/聊天）
│   ├── llm.py                #   智谱大模型调用
│   ├── vision.py             #   图片理解（描述、标签、OCR）
│   ├── vault.py              #   文件保险库（附件导入、Markdown 写入）
│   ├── zhipu_client.py       #   智谱 API 统一 HTTP 客户端
│   ├── api_client.py         #   HTTP API 客户端（桌面端→服务端通信）
│   └── utils.py              #   工具函数
│
├── ui/                       # 桌面端 UI 组件
│   └── widgets/
│       └── command_input.py  #   输入框组件（拖拽、自动补全）
│
└── web/                      # Web 部署目录
    ├── main.py               #   Flask 后端（全部 API 路由）
    ├── index.html            #   Web 前端 SPA
    ├── admin.html            #   管理员后台 SPA
    ├── import_setup.py       #   路径修复（Railway 部署用）
    └── requirements.txt      #   Web 端依赖
```

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

**关键设计决策：**

- **纯在线架构** — 桌面端通过 `MemoryApiClient` 调用 Flask 后端 API，支持自动重连和指数退避
- **共享业务逻辑** — `app/` 目录下的代码被桌面端和 Web 端共用，避免逻辑重复
- **FTS5 + LIKE 双重搜索** — 优先使用 FTS5 全文索引，索引不可用时回退到 LIKE 模糊匹配
- **服务端语音处理** — ASR/TTS 通过百度 AIP 在服务端完成，桌面端无需安装音频库
- **连接池与缓存** — 服务端使用 LRU 缓存（tags 300s、stats 60s、recent 30s），减少重复查询

## 快速开始

### 环境要求

- Python 3.11+
- Windows / macOS / Linux

### 安装依赖

```bash
pip install -r requirements.txt

# 桌面端额外依赖
pip install PyQt5
```

### 配置 API 密钥

复制 `.env.example` 为 `.env` 并填入实际值：

```bash
cp .env.example .env
```

获取方式：
- 智谱 AI：https://open.bigmodel.cn/ — 注册后在 API Keys 页面获取
- 百度语音：https://ai.baidu.com/ — 创建应用后获取

### 启动运行

**Web 模式（本地）：**

```bash
cd web
python main.py
```

浏览器打开 http://127.0.0.1:5000 ，管理后台在 http://127.0.0.1:5000/admin

**桌面 GUI 模式：**

```bash
python pyqt_local_agent.py
```

桌面端启动时会自动检测可用服务器：优先连接本地 `http://127.0.0.1:5000`，不可用则连接远程。也可点击标题栏右侧的设置按钮手动切换。

**本地完整运行（Web + 桌面 + 管理）：**

```bash
# 1. 启动 Web 服务
cd web && python main.py &

# 2. 启动桌面端（自动连接本地服务）
cd .. && python pyqt_local_agent.py
```

在 `.env` 中设置 `SERVER_URL=http://127.0.0.1:5000` 可让 API 客户端默认使用本地服务。

## 部署到 Railway

1. 将代码推送到 Git 仓库
2. 在 Railway 中连接仓库，自动检测 `Procfile` 并部署
3. 在 Variables 页面添加环境变量：

| 变量 | 必填 | 说明 | 默认值 |
|------|------|------|--------|
| `ZHIPU_API_KEY` | 是 | 智谱 AI API Key | — |
| `ADMIN_KEY` | 否 | 管理员密码 | — |
| `BAIDU_APP_ID` | 否 | 百度语音 APP ID | — |
| `BAIDU_API_KEY` | 否 | 百度语音 API Key | — |
| `BAIDU_SECRET_KEY` | 否 | 百度语音 Secret Key | — |
| `LOCAL_AGENT_MODEL` | 否 | 对话模型 | `glm-4-flash-250414` |
| `LOCAL_AGENT_VISION_MODEL` | 否 | 图片理解模型 | `glm-4v-flash` |
| `PORT` | 否 | 服务端口 | `5000` |
| `DATA_DIR` | 否 | 数据目录 | `./data` |

部署完成后访问 `https://your-app.up.railway.app` 即可使用。

## API 接口

详见 [使用文档](docs/USAGE.md#api-接口) 。

## 许可证

私人项目，仅供个人使用。
