# 暖暖记忆助手 - Web版 v6.0

## 功能特性

### 核心优化
- **版本管理**: `/api/version` 端点支持客户端更新检查
- **增强统计**: 包含文件数量、热门标签、存储大小等详细信息
- **输入验证**: 防止恶意输入和注入攻击
- **异常处理**: 完善的错误恢复机制

### 用户体验
- **暗色模式**: 一键切换亮色/暗色主题
- **消息气泡**: 带头像和时间戳的现代聊天界面
- **统计面板**: 查看记忆数量、热门标签等统计信息
- **响应式设计**: 完美适配手机、平板和桌面设备
- **图片放大**: 搜索结果中的图片可点击全屏查看
- **复制记忆**: 每条搜索结果支持一键复制
- **附件上传**: 点击📎按钮选择文件上传

### 搜索增强
- **高级搜索**: 支持标签、日期范围等多条件搜索
- **标签系统**: 为记忆添加和管理标签
- **实时过滤**: 快速找到相关记忆内容

## API 端点

### 基础功能
- `GET /api/health` - 健康检查
- `GET /api/version` - 版本信息（用于更新检查）
- `POST /api/chat` - 发送聊天消息
- `POST /api/chat/confirm_save` - 确认保存记忆
- `POST /api/search` - 搜索记忆
- `POST /api/save` - 保存记忆

### 语音功能
- `POST /api/speech_recognize` - 语音识别
- `POST /api/speech_synthesize` - 语音合成
- `POST /api/voice_dialogue` - 语音对话

### 记忆管理
- `GET /api/memories` - 获取记忆列表
- `GET /api/memories/recent` - 获取最近记忆
- `POST /api/memories/search` - 高级搜索
- `GET /api/memory/<id>` - 获取记忆详情
- `PUT /api/memory/<id>` - 更新记忆
- `DELETE /api/memory/<id>` - 删除记忆
- `GET /api/tags` - 获取所有标签
- `GET /api/stats` - 获取统计信息

### 文件处理
- `POST /api/upload` - 文件上传
- `GET /api/file/image` - 图片预览
- `POST /api/file/open` - 打开文件

### 其他
- `GET /api/vault/path` - 获取保险库路径
- `POST /api/clear` - 清空对话历史

## 配置要求

### 环境变量
```bash
# 智谱AI API（必填）
ZHIPU_API_KEY=your_api_key

# 百度语音API（可选 - 语音功能需要）
BAIDU_APP_ID=your_app_id
BAIDU_API_KEY=your_api_key
BAIDU_SECRET_KEY=your_secret_key

# 服务器配置
PORT=5000
DATA_DIR=./data
```

### 依赖包
```
flask>=2.3.0
baidu-aip>=2.2.18
Pillow>=10.0.0
```

## 部署指南

### Railway 部署
1. 推送代码到 Git 仓库
2. 在 Railway 中连接仓库
3. 配置环境变量
4. 自动部署完成

### 本地开发
```bash
cd web
pip install -r requirements.txt
python main.py
```

## 使用说明

### 基本操作
1. **聊天对话**: 在输入框输入文字或点击麦克风录音
2. **保存记忆**: 说"帮我记住..."或"记一下..."
3. **搜索记忆**: 说"帮我找..."或"查一下..."
4. **查看统计**: 点击右上角"📊 统计"按钮

### 高级功能
- **标签管理**: 为记忆添加标签便于分类
- **主题切换**: 点击月亮图标切换暗色模式
- **文件上传**: 点击📎按钮或拖拽文件到聊天区域
- **图片预览**: 点击记忆中的图片可全屏查看

---

**版本**: 6.0
