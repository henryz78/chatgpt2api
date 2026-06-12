# chatgpt2api — Development Progress

## 已完成 / Completed

### 1. 环境搭建
- 克隆 `hy` 分支到 Replit 工作区
- 通过 `uv` 安装 Python 3.13 依赖
- 配置服务运行在 8000 端口，通过反向代理对外暴露
- 构建 Next.js 前端（`web/` → `web_dist/`），登录页可正常访问

### 2. 账号导入
- 导入 ChatGPT Session Token 到 `data/accounts.json`
- 服务可正常通过该账号与 ChatGPT 后端通信

### 3. 单轮工具调用（Single-Turn Tool Calling）
- 新增 `services/protocol/tool_calling.py`
  - 工具定义转换为系统提示注入（无需修改上游 API 调用）
  - 解析模型输出中的工具调用格式（支持 `` ```json `` 代码块、裸 JSON、旧版 XML 三种格式）
  - 将解析结果格式化为 OpenAI 标准的 `tool_calls` 结构
  - 支持流式（stream）和非流式两种响应
- 修改 `services/protocol/openai_v1_chat_complete.py` 集成工具调用逻辑

### 4. 多轮工具调用（Multi-Turn Tool Calling）
- `tool_calling.py` 新增 `normalize_tool_history()`
  - 将 `role=assistant` 消息中的 `tool_calls` 字段转换为可读文本
  - 将 `role=tool` 的工具返回消息转换为 `role=user` 的文本消息
  - 使 ChatGPT 后端无感知地理解多轮工具对话历史

### 5. 提示词优化
- 明确禁止模型使用内置 web search / browsing 等自带工具
- 使用严格 JSON 输出格式示例，提升工具调用触发可靠性
- 健壮的多格式解析器，兼容模型不同输出风格

## 接口信息 / API Access

| 字段 | 值 |
|---|---|
| Base URL | 通过 Replit 反向代理暴露 |
| API Key | `chatgpt2api`（见 `config.json`） |
| 主要端点 | `POST /v1/chat/completions` |
| 支持 stream | ✅ |
| 工具调用 | ✅ 单轮 + 多轮 |

## 文件改动清单 / Changed Files

```
services/protocol/tool_calling.py          [新增] 工具调用核心逻辑
services/protocol/openai_v1_chat_complete.py [修改] 集成工具调用
services/protocol/conversation.py          [修改] 历史消息规范化
```

## 下一步 / Next Steps

### 优先级高
- [ ] **stream 模式工具调用测试** — 验证 `stream: true` + `tools` 组合在主流客户端（OpenCat、Cursor 等）下的兼容性
- [ ] **tool_choice 精细控制** — 支持 `tool_choice: {"type":"function","function":{"name":"xxx"}}` 指定特定工具
- [ ] **parallel tool calls** — 验证并行调用多个工具后，多个 `role=tool` 消息的多轮处理

### 优先级中
- [ ] **错误处理** — 工具调用解析失败时的优雅降级（返回原始文本而非 500）
- [ ] **单元测试** — 为 `tool_calling.py` 的 parser 添加测试覆盖
- [ ] **账号池负载均衡** — 多账号轮换策略完善

### 优先级低
- [ ] **OpenAI function calling 格式兼容** — 旧版 `functions` 参数支持（已被 `tools` 取代但部分客户端仍使用）
- [ ] **response format** — `response_format: {"type":"json_object"}` 支持

## 注意事项 / Notes

- `data/accounts.json` 包含用户 Session Token，**不提交到 git**（已加入 .gitignore 逻辑）
- 模型通过 ChatGPT Web 后端调用，不直接使用 OpenAI API Key
- 工具调用依赖系统提示注入，模型需足够智能才能可靠遵循（GPT-4 级别）
