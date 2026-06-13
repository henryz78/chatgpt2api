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
  - 解析模型输出中的工具调用格式（支持 ` ```json ` 代码块、裸 JSON、旧版 XML 三种格式）
  - 将解析结果格式化为 OpenAI 标准的 `tool_calls` 结构
  - 支持流式（stream）和非流式两种响应
- 修改 `services/protocol/openai_v1_chat_complete.py` 集成工具调用逻辑

### 4. 多轮工具调用（Multi-Turn Tool Calling）
- `tool_calling.py` 新增 `normalize_tool_history()`
  - 将 `role=assistant` 消息中的 `tool_calls` 字段转换为可读文本
  - 将 `role=tool` 的工具返回消息转换为 `role=user` 的文本消息
  - 使 ChatGPT 后端无感知地理解多轮工具对话历史

### 5. 提示词优化（第二轮）
- 明确禁止模型使用内置 web search / browsing 等自带工具
- 使用严格 JSON 输出格式示例，提升工具调用触发可靠性
- 健壮的多格式解析器，兼容模型不同输出风格
- **新增规则**：整个响应只允许出现一个工具调用块
- **新增规则**：不要把工具调用放进思考内容/推理过程
- **新增示例**：嵌套参数示例（示例D）
- **新增示例**：3 个工具并行调用示例（示例C，≥3 个工具时触发）

### 6. Parser 增强（对标 ds-free-api）
- 新增 ds-free-api DeepSeek native token 格式支持：
  - `<|tool▁calls▁begin|>...<|tool▁calls▁end|>`（模糊匹配 `▁`/`_` 和 `｜`/`|`）
  - `<|tool▁call▁begin|>...<|tool▁call▁end|>`（单次调用块）
  - `<|tool▁sep|>` 分隔符处理
- 新增 `<tool_call>...</tool_call>` 简化标签支持
- 新增 `<invoke name="..."><parameter name="...">...</parameter></invoke>` 格式（Anthropic/Claude 风格）
- 新增裸单对象（`{"name":..., "arguments":...}`，非数组）自动包装为数组
- `arguments` 是 JSON 字符串时自动解析规范化
- 代码块（`` ``` ``）内的示例不再被误识别为真实工具调用
- JSON repair 增强：
  - 非法反斜杠修复
  - 未加引号的 key 修复
  - 裸对象自动包装为数组

### 7. tool_choice 行为强化
- `tool_choice: "none"` → 不注入工具提示（已有）
- `tool_choice: "required"` → 模型未返回工具调用时返回 422 可诊断错误
- `tool_choice: {"type":"function","function":{"name":"xxx"}}` → 过滤只保留指定工具调用；未调用时返回错误
- `parallel_tool_calls: false` → 多工具时只保留第一个
- 未知工具名称过滤：不在 `request.tools` 中的工具自动过滤；全部未知则返回错误
- 新增 `enforce_tool_choice()` 公开函数

### 8. Stream 模式改善
- `stream_tool_calls_chunks()` 结构对齐 OpenAI SDK 期望的 `delta.tool_calls` 格式
- 最后一个 chunk 的 `finish_reason` 保证为 `"tool_calls"`
- 支持可选 `usage` 字段注入到最后一个 chunk
- 无工具调用时正常返回文本流（分块 32 字符）
- `tool_choice` 约束未满足时在流模式下也返回可诊断错误 chunk（`finish_reason: "error"`）
- stream + tools 测试覆盖已添加

### 9. Legacy 兼容（旧版 OpenAI function calling）
- 新增 `normalize_legacy_body()` 函数：
  - `functions` 数组 → `tools` 数组（`{"type":"function","function":...}`）
  - `function_call: "none"` → `tool_choice: "none"`
  - `function_call: "auto"` → `tool_choice: "auto"`
  - `function_call: {"name":"xxx"}` → `tool_choice: {"type":"function","function":{"name":"xxx"}}`
- `normalize_tool_history()` 增强：
  - assistant 消息中的 `function_call`（旧版）转换为 `tool_calls` 格式再处理
  - `role=function`（旧版工具结果）转换为 `<tool_results>` user 消息

### 10. Usage 修复
- 非流式工具调用响应现在正确计算 `prompt_tokens` / `completion_tokens` / `total_tokens`
  - `prompt_tokens` = `count_message_text_tokens(messages)`
  - `completion_tokens` = `count_text_tokens(full_text)`（模型原始输出）
- 流式工具调用的最后一个 chunk 携带 `usage` 字段
- `tool_calls_response()` 新增 `messages` / `full_text` 可选参数用于 token 计算

### 11. Parser 单元测试（新增 `test/test_tool_calling.py`）
- 覆盖 8 大测试类，~60 个测试用例：
  - 所有 tag 格式（canonical XML、DeepSeek、invoke、bare JSON、single object）
  - JSON repair（非法反斜杠、未加引号 key、裸对象）
  - 代码块误识别防护
  - 并行工具调用
  - `enforce_tool_choice` 全路径
  - `normalize_legacy_body` 全路径
  - `normalize_tool_history` 全路径（包括 legacy function_call）
  - `tools_system_message` 提示词内容验证
  - `stream_tool_calls_chunks` 结构验证 + arguments 可重建
  - ds-free-api 10 类异常格式（能支持的 9/10 已支持，R10 缺失右括号 TODO）

---

## 接口信息 / API Access

| 字段 | 值 |
|---|---|
| Base URL | 通过 Replit 反向代理暴露 |
| API Key | `chatgpt2api`（见 `config.json`） |
| 主要端点 | `POST /v1/chat/completions` |
| 支持 stream | ✅ |
| 工具调用 | ✅ 单轮 + 多轮 |
| Legacy functions | ✅ 自动转换 |

---

## 文件改动清单 / Changed Files

```
services/protocol/tool_calling.py           [大幅更新] 解析器增强 + legacy compat + enforce_tool_choice + usage
services/protocol/openai_v1_chat_complete.py [更新] 集成 enforce_tool_choice + legacy compat + usage
test/test_tool_calling.py                   [新增] 单元测试 (~60 用例)
PROGRESS.md                                 [本文件]
```

---

## ds-free-api 对齐状态

| 能力 | 状态 | 说明 |
|---|---|---|
| 主标签 `<\|tool▁calls▁begin\|>` / `<\|tool▁calls▁end\|>` | ✅ 已对齐 | 含 ▁/_ 和 ｜/\| 模糊匹配 |
| Fallback `<\|tool_call_begin\|>` | ✅ 已对齐 | |
| `<tool_call>` / `<tool_calls>` | ✅ 已对齐 | |
| 全角 `｜` 和 `▁`/`_` 模糊匹配 | ✅ 已对齐 | |
| 单对象而非数组 | ✅ 已对齐 | |
| `<invoke name=...>` 格式 | ✅ 已对齐 | |
| `arguments` 是字符串时规范化 | ✅ 已对齐 | |
| JSON repair：非法反斜杠 | ✅ 已对齐 | |
| JSON repair：未加引号 key | ✅ 已对齐 | |
| JSON repair：缺失数组括号 | ⚠️ 部分支持 | 尾部缺失 `]` 的简单情况，深度嵌套 TODO |
| JSON repair：只有对象 | ✅ 已对齐 | 自动包装为数组 |
| 代码块示例不误识别 | ✅ 已对齐 | strip_code_fences |
| tool_choice: required 校验 | ✅ 已对齐 | 422 诊断错误 |
| tool_choice: named 过滤 | ✅ 已对齐 | |
| parallel_tool_calls: false | ✅ 已对齐 | |
| 未知工具过滤 | ✅ 已对齐 | |
| 流式滑动窗口 ToolCallStream | ❌ 未对齐 | 仍为 collect_text 后伪造，TODO 长期优化 |

---

## 未对齐项及原因

1. **滑动窗口实时流式工具调用**（ds-free-api ToolCallStream）：当前依然是先 `collect_text` 收全量响应再伪造 stream chunks。真正的实时流需要在模型输出过程中检测 tool call 边界，复杂度较高，短期接受现状，长期 TODO。

2. **缺失右括号的深度修复**（R10）：对于 `[{"name":"foo","arguments":{"a":"b"}` 这类尾部 `]` 和 `}` 都缺失的情况，当前 repair 无法处理。需要引入 tokenizer-level 修复（参考 ds-free-api 的 bracket balance 算法），TODO。

---

## 下一步 / Next Steps

- [ ] **实时流式工具调用**：参考 ds-free-api 滑动窗口，实现真正边解析边输出
- [ ] **深度 JSON bracket repair**：处理多层缺失括号的情况
- [ ] **端到端集成测试**：用真实账号验证 stream + tools 在 OpenCat / Cursor 下的兼容性
- [ ] **账号池负载均衡**：多账号轮换策略完善
- [ ] **response_format: json_object**：JSON 模式支持

---

## 注意事项 / Notes

- `data/accounts.json` 包含用户 Session Token，**不提交到 git**（已加入 .gitignore 逻辑）
- 模型通过 ChatGPT Web 后端调用，不直接使用 OpenAI API Key
- 工具调用依赖系统提示注入，模型需足够智能才能可靠遵循（GPT-4 级别）
- 不要改动图片生成、Responses API、Anthropic Messages 等现有逻辑（本次未触及）
