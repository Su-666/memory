# 暖暖记忆助手 v6.0

一个基于大语言模型的个人记忆管理助手，支持 Web 端使用方式。用户可以通过文字、语音或图片与"暖暖"对话，保存想法、联系人、备忘等内容，并随时搜索回忆。

## 核心能力

- **智能意图识别** — 自动判断保存、搜索、闲聊三种意图
- **全文检索** — SQLite FTS5 中文全文搜索 + BM25 相关性排序，自动回退 LIKE 模糊匹配
- **图片理解** — 上传图片自动生成描述、标签和 OCR 文字提取（GLM-4V）
- **语音交互** — 百度 ASR/TTS，支持连续语音对话模式
- **文件保险库** — 所有记忆以 Markdown 文件归档到 `memory_vault/` 目录
- **管理员后台** — 独立 SPA 管理页面，支持数据查看、搜索、删除、导出和备份
- **暗色模式** — Web 端支持亮色/暗色主题切换

## 技术栈

| 组件 | 技术 |
|------|------|
| Web 后端 | Flask + Gunicorn + Flask-CORS |
| Web 前端 | 原生 HTML/CSS/JS 单页应用 |
| 数据库 | SQLite + FTS5 全文索引 |
| 大模型 | 智谱 AI（GLM-4-Flash 对话，GLM-4V-Flash 图片理解） |
| 语音 | 百度 AIP（服务端 ASR + TTS） |

## 项目结构

```
记忆助手/
├── requirements.txt          # Python 依赖
├── .env.example              # 环境变量模板
├── .env                      # API 密钥（不提交到 Git）
├── launcher.py               # 桌面程序入口（Flask + pywebview）
├── build_exe.py              # PyInstaller 打包脚本
│
├── app/                      # 核心业务逻辑
│   ├── db.py                 #   数据库连接与表结构初始化
│   ├── repo.py               #   记忆 CRUD、标签提取、保险库导入
│   ├── search.py             #   FTS5 全文搜索 + LIKE 回退 + 评分排序
│   ├── answer.py             #   基于记忆的回答生成（LLM + 本地规则）
│   ├── intent_chat.py        #   统一意图路由（保存/搜索/聊天）
│   ├── llm.py                #   智谱大模型调用
│   ├── vision.py             #   图片理解（描述、标签、OCR）
│   ├── vault.py              #   文件保险库（附件导入、Markdown 写入）
│   ├── zhipu_client.py       #   智谱 API 统一 HTTP 客户端
│   └── utils.py              #   工具函数
│
└── web/                      # Web 部署目录
    ├── main.py               #   Flask 后端（全部 API 路由）
    ├── index.html            #   Web 前端 SPA
    ├── admin.html            #   管理员后台 SPA
    └── import_setup.py       #   路径修复（Web 部署用）
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
        └─── Web 前端
               (浏览器)
```

**关键设计决策：**

- **纯 Web 架构** — 无需安装客户端，浏览器直接访问
- **轻量设计** — 所有代码在服务器端运行，客户端仅需浏览器
- **FTS5 + LIKE 双重搜索** — 优先使用 FTS5 全文索引，索引不可用时回退到 LIKE 模糊匹配
- **服务端语音处理** — ASR/TTS 通过百度 AIP 在服务端完成，客户端无需安装音频库
- **连接池与缓存** — 服务端使用 LRU 缓存（tags 300s、stats 60s、recent 30s），减少重复查询

## 快速开始

### 环境要求

- Python 3.11+
- Windows / macOS / Linux

### 安装依赖

```bash
pip install -r requirements.txt
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

**启动 Web 服务：**

```bash
cd web
python main.py
```

浏览器打开 http://127.0.0.1:5000 ，管理后台在 http://127.0.0.1:5000/admin

### 打包为桌面程序

将 Flask 后端 + Web 前端打包为一个 Windows 桌面程序，双击即可运行，自动以原生窗口打开界面。

**1. 安装打包依赖：**

```bash
pip install pywebview pythonnet pywin32 pyinstaller
```

**2. 执行打包：**

```bash
python build_exe.py
```

**3. 运行：**

打包完成后，在 `dist/暖暖记忆助手/` 目录下双击 `暖暖记忆助手.exe` 即可启动。
程序会在 `%APPDATA%/记忆助手/` 下创建数据目录（数据库、保险库、`.env` 配置），首次启动需编辑该 `.env` 填入 API 密钥。

> 说明：桌面程序通过 pywebview 创建原生窗口加载本地 Flask 服务，关闭窗口即退出。系统需已安装 WebView2 运行时（Windows 10/11 默认自带）。

## API 接口

详见 [使用文档](USAGE.md#api-接口)。

## 许可证

私人项目，仅供个人使用。
