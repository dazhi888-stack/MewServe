# MewHelp · 第 01 章：纯对话电商客服

Python 3.11+、FastAPI、langchain-core / langchain-openai。应用以 `ChatOpenAI` 直连 OpenAI Chat Completions 兼容上游；不使用数据库、会话存储、工具调用、Agent、检索或真实售后操作。

## 本地启动（Windows PowerShell）

先安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)，在项目根目录执行：

```powershell
uv sync --dev
if (-not (Test-Path -LiteralPath .env)) { Copy-Item -LiteralPath .env.example -Destination .env }
notepad .env
```

填写 `OPENAI_BASE_URL`、`OPENAI_MODEL`、`OPENAI_API_KEY`，使用供应商给你的兼容入口、可用模型名与密钥；本地上游也须填写其要求的非空 key / 占位值。上面只在文件不存在时复制模板，保留已有 `.env`。不要把密钥填进请求文件或浏览器。环境变量优先于 `.env`；修改后重启服务。

```powershell
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

浏览器打开 <http://127.0.0.1:8000>；接口说明在 <http://127.0.0.1:8000/docs>。另开 PowerShell：

```powershell
curl.exe --silent --show-error http://127.0.0.1:8000/health
```

`model_configured: true` 只代表三项非空，不证明鉴权、模型或上游网络可用。配置为空也能启动并打开页面；模型接口返回 503 / `model_not_configured`。不要覆盖真实 `.env` 来跑模拟演示。

## 配置与供应商边界

- `STRUCTURED_OUTPUT_METHOD=json_mode`（默认）：明确发送 `response_format.type=json_object`，Prompt 描述字段；本地严格检查原始 JSON 和三字段 schema。
- `STRUCTURED_OUTPUT_METHOD=json_schema`：明确发送 JSON Schema，要求上游支持该模式；设置 `strict=True`。本地仍检查原始 JSON 和字段。模式不支持或返回非法内容时明确失败，没有降级到工具调用或另一模式。
- `OPENAI_TOKEN_LIMIT_FIELD=max_tokens`（默认），若目标模型文档要求则改为 `max_completion_tokens`。最终只发送所选的一个字段，值取 `MAX_OUTPUT_TOKENS`；上游须实际遵守该参数。
- `CONTEXT_WINDOW_TOKENS=8192`、`MAX_OUTPUT_TOKENS=512`、`TOKEN_SAFETY_MARGIN=512`；输出预留与安全余量之和必须小于上下文窗口。
- `HISTORY_MAX_MESSAGES=20`：最多保留的历史消息数，奇数向下取完整问答对，0 禁用历史；超预算继续从最旧完整问答对删除，保留系统 Prompt 与当前输入。
- `REQUEST_TIMEOUT_SECONDS=60`、`SHOP_NAME=MewHelp`。密钥只在后端配置中；`.env` 与 `artifacts/` 已忽略。

当前输入 token 估计是 **UTF-8 字节数 + 每条消息 16**，不是供应商 tokenizer 的真实计数；服务模板也计入。该策略偏保守，可能提前裁剪；不能保证覆盖任意供应商模板或隐含开销。根据所选模型的真实窗口设置预算并保留余量；受保护的系统/当前消息已超预算时返回 413，绝不静默截断当前输入。

统一协议只统一应用调用形状，不承诺 GPT、Claude、DeepSeek、Ollama 的能力相同，也不支持它们的全部原生扩展。请逐一核对兼容入口、流式文本、输出上限字段及 `response_format` 支持。Claude 的 OpenAI 兼容文档说明 `response_format` 会被忽略，因此该路径不能视为有原生 JSON 约束；Prompt 请求 + 本地校验可能成功也可能明确失败。不能将一家的配置或测试推广为四家均通过。当前仅做过受控兼容上游验证，四家真实模型均未完成本章验收。

官方依据：[LangChain OpenAI 集成](https://docs.langchain.com/oss/python/integrations/chat/openai)、[Claude OpenAI SDK 兼容](https://platform.claude.com/docs/en/cli-sdks-libraries/libraries/openai-sdk)。配置使用当前账号实际可用模型，不在示例中固定未经验证的最新型号。

## curl 首轮与第二轮

使用 `curl.exe` 避免 Windows PowerShell 的 `curl` 别名；JSON 从 UTF-8 文件读取，避免命令行引号转义。所有命令在项目根目录运行：

```powershell
New-Item -ItemType Directory -Force artifacts | Out-Null
curl.exe --silent --show-error --no-buffer --max-time 90 -H "Content-Type: application/json" --data-binary "@examples/chat-1.json" http://127.0.0.1:8000/api/chat/stream
```

首轮请求：

```json
{"message":"订单 001234 的杯子碎了，我想换一个新的。","history":[]}
```

事件为 `meta`（估计输入 tokens / 裁剪条数）、`token`（`text` 增量）、`done`（`finish_reason`）或 `error`（安全 code/message）。`token` 名称表示文本增量，不保证是一枚模型 token。`done.finish_reason=length` 表示已到输出上限，应提示用户回复可能不完整。响应开始后的错误通过 SSE `error` 发送，不能只看 HTTP 200 判断成功。

要验证连续对话，先保存**本次真实首轮**输出，再生成第二轮 JSON；不是把预设助手文字当成实际回复。以下脚本只接受合法 `done` 且没有 `error` 的首轮：

```powershell
curl.exe --silent --show-error --no-buffer --max-time 90 -H "Content-Type: application/json" --data-binary "@examples/chat-1.json" --output artifacts/chat-1.sse http://127.0.0.1:8000/api/chat/stream
if ($LASTEXITCODE -ne 0) { throw '首轮网络请求失败' }
./examples/make-followup.ps1
curl.exe --silent --show-error --no-buffer --max-time 90 -H "Content-Type: application/json" --data-binary "@artifacts/chat-2.json" http://127.0.0.1:8000/api/chat/stream
```

`examples/chat-2.json` 是可读的**静态格式示例**，其中助手内容是示意文本；实际连续两轮应使用上述生成的 `artifacts/chat-2.json`。接口不接受 `conversation_id`，客户端每次携带完整交替的 user/assistant 问答对；后端只在当前请求内裁剪，不保存历史。每条内容最多 8000 字符，history 最多 100 条，不允许客户端 system 角色。

要查看多段网络接收时间，可给首轮命令加入：

```powershell
curl.exe --silent --show-error --no-buffer --max-time 90 --trace-time --trace-ascii artifacts/chat-timing.txt -H "Content-Type: application/json" --data-binary "@examples/chat-1.json" --output artifacts/chat-timing.sse http://127.0.0.1:8000/api/chat/stream
```

`--no-buffer` 关闭 curl 输出缓冲，无法强迫上游增量生成。应观察多个接收时间，以及首段正文早于最后正文/终止事件；不要把 TestClient 聚合输出当成时序证据。trace 会保存请求与正文，只用合成输入，勿公开含私人订单信息的日志。

## 售后提取

```powershell
curl.exe --silent --show-error --max-time 90 -H "Content-Type: application/json" --data-binary "@examples/after-sales.json" http://127.0.0.1:8000/api/after-sales/extract
```

合成输入为“订单 001234 的杯子碎了，我想换一个新的。”；期望提取形状（不是承诺模型已实测）：

```json
{"order_id":"001234","request_type":"exchange","expected_solution":"换一个新的"}
```

仅返回这三个必需键，禁止额外字段；`order_id` 与 `expected_solution` 缺失时为 `null`。订单号为字符串，保留前导零；类型枚举为 `refund` / `return_refund` / `exchange` / `repair` / `logistics` / `other` / `unknown`，未知类型使用 `unknown`。这是描述提取，不会查询订单、退款、换货或发货。非法 JSON、字段错误、拒绝/工具响应与不支持的模式均返回明确错误。接口错误格式为 `code` / `message`（输入校验为 FastAPI 422）。

## 验证与当前结果

```powershell
uv run pytest -q
uv run python scripts/evaluate_prompts.py --output artifacts/ch01-eval.json
```

评估仅包含 9 个合成售后样本与 2 组客服 Prompt；无配置时退出码 2，产物 `not_executed`，不是模型失败率或通过率。配置完整后发生首个上游错误会停止并保留已有结果。字段 exact match 只是诊断，`expected_solution` 同义改写与客服 rubric 须人工阅读；结果成功执行也保留 `pending_manual_review`，不自动宣称语义验收通过。

2026-11-05 本地全套后端测试：115 项通过。受控真实 HTTP 链路已观察多段 SSE、用实际首轮回复构造的第二轮历史、三字段 JSON；Chrome 已观察逐字显示、追问、忙碌禁用、失败/断流提示、390px 窄屏及刷新/新对话清空。受控回复是预设合成内容，只证明协议、页面与历史传递。

真实 `.env` 三项为空，真实模型两轮/流式效果、9+2 评估及人工语义 rubric **未执行**；GPT / Claude / DeepSeek / Ollama 未逐家验证。后端整体独立评审与本地整合由后续交付阶段处理，不能由本节测试结果代替。页面按用户例外免 TDD / 代码评审；自然 EOF 无终止事件和 100 条页面历史上限仅实现/自查，未逐项浏览器实测。

聊天历史只保存在页面内存，刷新/新对话清空；模型缺乏真实订单工具，输出不能作为真实售后办理结果。取消请求会取消上游流/等待并释放连接；浏览器只将合法 done 后的完整问答加入后续历史。页面无外部 CDN。开发过程和纠偏见 [dev-notes/ch01.md](dev-notes/ch01.md)，已批准设计见 `docs/superpowers/specs/`。
