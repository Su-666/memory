# 暖暖记忆助手

一个基于大语言模型的个人记忆管理助手。用户可以通过文字、语音或图片与"暖暖"对话，保存想法、联系人、备忘等内容，并随时搜索回忆。

支持两种使用方式：
- **桌面程序**（推荐）— Windows 安装包，双击即用，原生窗口体验
- **Web 服务** — 浏览器访问，适合服务器部署

## 核心能力

- **智能意图识别** — 自动判断保存、搜索、闲聊三种意图
- **全文检索** — SQLite FTS5 中文全文搜索 + BM25 相关性排序，自动回退 LIKE 模糊匹配
- **图片理解** — 上传图片自动生成描述、标签和 OCR 文字提取（GLM-4V）
- **语音交互** — 百度 ASR/TTS，支持连续语音对话模式
- **文件保险库** — 所有记忆以 Markdown 文件归档到 `memory_vault/` 目录
- **管理员后台** — 独立管理页面，支持数据查看、搜索、删除、导出和备份
- **可视化设置** — 程序内直接配置 API 密钥，支持测试连接，无需手动编辑文件
- **暗色模式** — 支持亮色/暗色主题切换

## 技术栈

| 组件 | 技术 |
|------|------|
| 桌面窗口 | pywebview（WebView2 内核）|
| 后端 | Flask + Flask-CORS + Flask-Compress |
| 前端 | 原生 HTML/CSS/JS 单页应用 |
| 数据库 | SQLite + FTS5 全文索引 |
| 大模型 | 智谱 AI（GLM-4-Flash 对话，GLM-4V-Flash 图片理解） |
| 语音 | 百度 AIP（ASR + TTS） |
| 打包 | PyInstaller + Inno Setup |

## 项目结构

```
记忆助手/
├── launcher.py               # 桌面程序入口（Flask + pywebview）
├── build_exe.py              # 一键打包脚本（PyInstaller → Inno Setup）
├── 安装包.iss                 # Inno Setup 安装脚本
├── appicon.ico / .png        # 应用图标
├── requirements.txt          # Python 依赖
├── .env.example              # 环境变量模板
├── .env                      # API 密钥（不提交到 Git）
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
│   └── utils.py              #   工具函数（含 .env 读写）
│
├── web/                      # Web 端
│   ├── main.py               #   Flask 后端（全部 API 路由 + 设置接口）
│   ├── index.html            #   Web 前端 SPA
│   ├── admin.html            #   管理员后台 SPA
│   └── import_setup.py       #   路径修复
│
└── .github/workflows/
    └── build-release.yml     # GitHub Actions 自动构建发布
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
  Flask API (web/main.py)
        │
   ┌────┴────┐
   ▼         ▼
 pywebview   浏览器
(桌面窗口)  (Web 端)
```

**关键设计决策：**

- **桌面 + Web 双模式** — 桌面程序用 pywebview 封装 Flask 为原生窗口，也可作为 Web 服务部署
- **FTS5 + LIKE 双重搜索** — 优先使用 FTS5 全文索引，索引不可用时回退到 LIKE 模糊匹配
- **服务端语音处理** — ASR/TTS 通过百度 AIP 在服务端完成，客户端无需安装音频库
- **可视化设置** — API 密钥通过程序内设置页面管理，保存到用户数据目录的 `.env`，立即生效
- **数据隔离** — 桌面程序数据存储在 `%APPDATA%/记忆助手/`，卸载不删除用户数据

## 快速开始

### 方式一：桌面程序（推荐）

1. 下载 `暖暖记忆助手-Setup.exe` 安装包
2. 双击运行，按向导完成安装
3. 桌面快捷方式启动程序
4. 首次启动自动弹出设置页面，填入智谱 API Key → 点"测试连接"验证 → 保存
5. 即可使用

> 系统需已安装 WebView2 运行时（Windows 10/11 默认自带）。

### 方式二：Web 服务（开发/服务器部署）

#### 环境要求

- Python 3.11+
- Windows / macOS / Linux

#### 安装依赖

```bash
pip install -r requirements.txt
```

#### 配置 API 密钥

复制 `.env.example` 为 `.env` 并填入实际值，或启动后在设置页面中填写：

```bash
cp .env.example .env
```

获取方式：
- 智谱 AI：https://open.bigmodel.cn/ — 注册后在 API Keys 页面获取
- 百度语音：https://ai.baidu.com/ — 创建应用后获取

#### 启动

```bash
cd web
python main.py
```

浏览器打开 http://127.0.0.1:5000 ，管理后台在 http://127.0.0.1:5000/admin

### 方式三：桌面开发模式

```bash
pip install -r requirements.txt
python launcher.py
```

直接以桌面窗口方式运行，无需打包。

## 设置页面

点击界面右上角 **⚙️ 设置** 按钮打开设置页面：

| 配置项 | 说明 | 必填 |
|--------|------|------|
| 智谱 AI API Key | 大模型 API 密钥 | ✅ |
| 对话模型 | 默认 glm-4-flash-250414（免费） | 否 |
| 图片理解模型 | 默认 glm-4v-flash（免费） | 否 |
| 百度 App ID | 语音功能需要 | 否 |
| 百度 API Key | 语音功能需要 | 否 |
| 百度 Secret Key | 语音功能需要 | 否 |
| 管理员密码 | 管理后台登录密码 | 否 |

- **测试连接** — 保存前可验证 API Key 是否有效
- **访问管理后台** — 点击按钮在浏览器中打开管理后台（自动带上密码登录）
- 配置保存在 `%APPDATA%/记忆助手/.env`，保存后立即生效，无需重启

## 打包与发布

### 本地打包

一键生成 Windows 安装包（PyInstaller → Inno Setup → 自动清理中间产物）：

```bash
pip install pyinstaller pywebview pythonnet pywin32
python build_exe.py
```

输出：`installer_output/暖暖记忆助手-Setup.exe`

## 数据存储

| 文件 | 位置 | 作用 |
|------|------|------|
| `.env` | `%APPDATA%/记忆助手/` | API 密钥等配置 |
| `agent.db` | `%APPDATA%/记忆助手/data/` | SQLite 数据库（记忆内容 + 全文索引） |
| `memory_vault/` | `%APPDATA%/记忆助手/data/` | Markdown 归档 + 上传附件 |

- 卸载程序不删除用户数据
- 备份只需复制整个 `%APPDATA%/记忆助手/` 文件夹

## API 接口

详见 [使用文档](USAGE.md#api-接口)。

## 捐款支持

如果这个项目对你有帮助，欢迎请作者喝杯咖啡 ☕

| 支持作者 |
|:---:|
| ![支持作者](data/assets/暖暖记忆助手-author-support-poster.png) |

> 扫码即可捐赠，金额随意，感谢支持！

## 许可证

私人项目，仅供个人使用。
