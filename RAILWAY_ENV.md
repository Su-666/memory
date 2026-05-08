# Railway 环境变量配置

在 Railway 部署时，需要在 **Variables** 页面添加以下环境变量。

## 必需变量

| 变量名 | 说明 | 示例值 |
|--------|------|--------|
| `ZHIPU_API_KEY` | 智谱AI API Key（对话+图片理解共用） | `your-zhipu-api-key` |
| `BAIDU_APP_ID` | 百度语音APP ID | `12345678` |
| `BAIDU_API_KEY` | 百度语音API Key | `your-baidu-api-key` |
| `BAIDU_SECRET_KEY` | 百度语音Secret Key | `your-baidu-secret` |

## 可选变量（模型配置）

| 变量名 | 说明 | 默认值 | 可选值 |
|--------|------|--------|--------|
| `LOCAL_AGENT_MODEL` | 对话大模型 | `GLM-4-Flash-250414` | `GLM-4-Flash-250414`（免费）<br>`GLM-4-Plus`<br>`GLM-4-Air` |
| `LOCAL_AGENT_VISION_MODEL` | 图片理解模型 | `glm-4v-flash` | `glm-4v-flash`（免费）<br>`glm-4v-plus`<br>`glm-4v` |

## 可选变量（其他）

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `LOCAL_AGENT_VISION_RETRIES` | 图片理解失败重试次数 | `3` |
| `PORT` | 服务端口（Railway自动设置） | `5000` |

## 添加步骤

1. 打开 Railway 项目 → 点击 **Variables** 标签
2. 点击 **+ New Variable** 或 **原始编辑器**
3. 添加必需变量（ZHIPU_API_KEY、BAIDU 三个Key）
4. 如需切换模型，添加 `LOCAL_AGENT_MODEL` 或 `LOCAL_AGENT_VISION_MODEL`
5. 重新部署（Redeploy）生效

## 模型选择建议

| 场景 | 对话模型 | 图片理解模型 |
|------|----------|-------------|
| 免费使用 | `GLM-4-Flash-250414` | `glm-4v-flash` |
| 更好理解能力 | `GLM-4-Plus` | `glm-4v-plus` |
| 平衡选择 | `GLM-4-Air` | `glm-4v` |

> 注意：非 Flash 模型可能产生费用，请确认智谱账户余额充足。
