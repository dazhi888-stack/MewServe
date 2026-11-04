# 第 01 章：电商智能客服纯对话设计

日期：2026-11-05（北京时间）
状态：独立 builder check 已通过（nothing important left）；用户于 2026-11-05 回复“批准”，设计已定稿。此文档不代表功能已实现。

## 1. 意图与验收

从空仓库做一个可运行、可演示的电商客服第一章：先理解直接模型调用、Prompt、流式响应、结构化提取与有限上下文，再为以后章节保留清晰的扩展位置。

用户要求的验收：

1. curl 调用对话接口能逐步收到 SSE 回复。
2. 第二轮携带第一轮问答后，模型能理解追问。
3. 一段售后描述得到订单号、诉求类型、期望方案三个固定字段的 JSON。
4. 浏览器打开聊天页，显示消息气泡，回复逐字出现，能继续多轮对话。

离线测试或模拟上游通过不能冒充真实模型验收通过。真实模型的逐步输出、第二轮理解和抽取效果必须另外记录。

## 2. 固定约束与不做项

- Python + FastAPI + LangChain。
- 应用通过 `langchain-openai` 的 `ChatOpenAI` 直连上游，统一 OpenAI Chat Completions 协议。
- 上游地址、模型名、密钥从 `.env` / 进程环境读取；密钥不进入浏览器、日志、版本控制或错误详情。
- 不引入数据库、Redis、LangGraph、工具调用、Agent 循环、检索、订单系统或真实售后执行。
- 本章不做登录、云部署、多租户和持久化；服务默认供本地开发演示。
- 使用现行官方文档中的接口，Context7 查证后才实现；版本由项目依赖文件与锁文件固定。
- 客服不能假装查询过订单、提交过退款或有本章未接入的业务能力。

## 3. 已讨论的方案与决定

### 方案 A：后端内存保存会话（未采用）

服务端保存问答，首轮返回 conversation_id。curl 追问方便，但有会话生命周期、并发和进程重启语义，后端要维护状态。

### 方案 B：客户端携带历史（用户选择）

每次请求包含 `message` 和可选 `history`，后端裁剪后直接调用模型，不保存会话。浏览器自动发送当前问答；curl 第二轮显式带上前一轮问答。代价是请求体随历史增长，且历史来自客户端，不是可信业务记录。

用户选择原话：“B：客户端每次请求携带历史消息”。因此不实现 conversation_id，也不增加内存会话存储。

### 供应商结构化输出边界（用户已接受）

**事实校勘（2026-11-05）**：最初设计依据 Context7 对 Claude 官方同 URL 的旧缓存，记录为支持 streaming、忽略 `response_format`。交付评审后直接核对[当前官方兼容文档](https://platform.claude.com/docs/en/cli-sdks-libraries/libraries/openai-sdk)的 API behavior / Simple fields：`response_format` 仅接受 `json_schema` 且 `strict:true`；`json_object` / `text` 返回 400。Context7 本轮仍返回旧 Ignored 索引，不能覆盖当前官网。此处更正来源事实，早期检索/用户确认过程保留在 dev-notes；不重开已批准设计、不替换 SDK 或协议。统一协议并不意味着各上游提供相同的原生结构化约束。

用户已接受：保持统一 OpenAI 协议和 `with_structured_output`，按能力显式配置输出方式，结构化校验失败明确报错。实现只允许 `json_mode` / `json_schema`，不使用隐含的 function_calling 默认值，不自动切换 SDK 或协议，不暗中用工具调用兜底。

项目默认仍为 `json_mode`，只用于支持 `json_object` 的上游；对 Claude 应显式设置 `STRUCTURED_OUTPUT_METHOD=json_schema`，现有实现为该模式发送 `strict:true`，并遵守其 JSON Schema 限制。成功时仍必须符合相同的响应字段定义；不支持所选模式或校验失败时返回稳定错误，由开发者调整配置，禁止静默降级。当前官方能力说明不等于本项目已经完成 Claude 真实模型验收，四家真实模型仍待验。

## 4. 后端接口契约

### POST `/api/chat/stream`

请求：

```json
{
  "message": "那退货运费谁承担？",
  "history": [
    {"role": "user", "content": "我收到的电饭煲有破损，想退货。"},
    {"role": "assistant", "content": "请保留商品和包装，并提供订单号以便进一步说明售后步骤。"}
  ]
}
```

- `message` 必须是非空文本；禁止空白消息、未知字段、过长输入。
- `history` 默认为空，仅允许 user/assistant 文本消息，按完整的 user→assistant 轮次排列；禁止客户端发送 system/tool 消息。
- 历史条数、单条文本长度有请求验证上限；历史裁剪不能代替请求大小限制。
- 在建立 SSE 前完成请求验证、配置检查和 token 预算检查。验证错误为 HTTP 422；未配置模型为 HTTP 503。
- 成功响应为 `text/event-stream`，不缓冲到整段生成结束。
- 每个非空上游文本 delta 立即转发 `event: token`，`data` 为 `{"text":"..."}`；一个 delta 可能包含多个模型 token，不谎称协议保证一事件一 token，也不在服务端人为拆字拖慢上游。
- 完成时发送 `event: done`，包含 `finish_reason`；流中失败时发送 `event: error`，包含稳定 `code` 和可公开的 `message`，随后结束，不发送成功 done。
- 可发送起始 `meta` 事件，供调用者确认预算裁剪情况。数据使用 JSON 编码，正确保留换行和中文。
- 采用 SSE 禁缓存与禁止代理缓冲的响应头。客户端断连时取消上游请求，释放流和连接。
- 不把推理/思考块、工具调用块或上游完整错误体作为客服正文输出。

### POST `/api/after-sales/extract`

请求为 `{"description":"订单 001234 的杯子到了就碎了，我想换一个新的。"}`。

成功响应固定三个字段：

```json
{
  "order_id": "001234",
  "request_type": "exchange",
  "expected_solution": "换一个新的"
}
```

- `order_id`: `str | null`，保留前导零；缺失、含糊或无法确定唯一订单时为 null，不猜测。
- `request_type`: refund / return_refund / exchange / repair / logistics / other / unknown。对应仅退款 / 退货退款 / 换货 / 维修 / 物流 / 其他明确诉求 / 无法判断。
- `expected_solution`: `str | null`，只提取用户明确表达的期望；缺失时为 null。
- 使用 Pydantic schema + `with_structured_output`，在 Prompt 中提供字段规范，输入描述作为数据处理，拒绝把描述中的指令作为系统规则。
- 这是独立的一次模型提取，不调用售后工具、不改变聊天历史、不替用户执行售后。
- 输入验证失败为 HTTP 422；未配置模型为 HTTP 503；上游不支持模式、超时、无效 JSON、schema 校验失败为可区分的稳定非 2xx 错误，原始响应和密钥不外泄。

### GET `/health` 与 GET `/`

- `/health` 为本地进程存活检查，不调用付费模型；返回能否发起模型调用的布尔配置状态，不暴露密钥或完整配置。
- `/` 返回同源聊天页；静态资源由 FastAPI 提供，无需另建前端服务。
- 缺少 `.env` 时可启动进程、健康检查和页面；真实模型接口返回明确的未配置错误，不提供伪装成真实模型的演示回复。

## 5. Prompt 与上下文预算

- 客服 system prompt 独立文本文件，用 `PromptTemplate` 注入店铺名称等受控参数，再以 SystemMessage 放入聊天 Prompt；历史用 MessagesPlaceholder 或等价的消息对象拼接，避免把用户的花括号再次当成模板变量。
- 角色约束：用清楚、礼貌、简洁的中文处理电商咨询；信息不足时询问；不编造订单状态、店铺政策、赔偿金额、查询结果或已经执行的动作；不索取密码、验证码等无关敏感信息；描述无法核实的业务事实时说明需要实际客服核实。
- 抽取使用独立 PromptTemplate，字段说明和 JSON schema 作为模板变量传入，防止 JSON 花括号误解析。
- 总预算包含 system prompt、保留历史、当前用户消息、回复上限以及安全余量。
- 优先保留 system 和本轮用户文本，从最老的完整 user/assistant 对开始删除历史，同时限制保留消息数量；不得留下孤立 assistant，不得把本轮文本悄悄截断。
- 如果 system + 当前消息 + 回复预留已经超过预算，在调模型前明确拒绝。
- 跨 GPT/Claude/DeepSeek/Ollama 无法共用一个保证精确的 tokenizer。本章用统一、保守的计数估算并预留安全余量，明确它不是供应商精确计费 token 数。计数算法和边界测试交给实现者决定并记录，不能宣传适配所有模型的严格 tokenizer 等价性。

## 6. 配置、错误与项目边界

配置包括上游 base URL、model、API key、结构化输出 method、模型上下文预算、回复上限、安全余量、历史上限与请求超时。提供带注释的 `.env.example`，真实 `.env` 加入忽略规则。

不强制发送所有模型都不支持的 temperature、stream_usage 等可选参数；以查到的官方兼容接口为准。Ollama 仍走其 OpenAI 兼容地址，不引入专用客户端；它是否能正常返回取决于本机服务与模型实际可用性。

错误处理覆盖缺少配置、输入超限、鉴权失败、上游连接失败、限流、超时、结构化解析失败、流式中断。对外错误不直接拼接异常字符串；开发日志也不记录 API key 和完整用户售后文本。

## 7. 聊天页面（按用户要求走 Vibe Coding）

这里只记录已要求的功能和与后端的交接契约，不把页面纳入 brainstorm、TDD 或 code review 门禁。

- 同源静态 HTML/CSS/JavaScript，消息气泡区分用户与客服；提供输入、发送、新对话与忙碌/失败状态，适应桌面和窄屏。
- 用 fetch POST + ReadableStream 消费 SSE；支持跨数据块事件、中文 UTF-8 和换行，不能用只支持 GET 的 EventSource 发此 POST。
- 按到达的文本逐字渲染，采用 textContent 等安全方式显示模型与用户文本，避免当 HTML 执行。
- 页面保存当前问答于页面内存，每次自动携带完整已成功问答作为 history；刷新或“新对话”清空。无持久化需求。
- 生成时禁止重复发送；失败或断流显示可理解的提示，不把不完整回复当作成功问答加入后续历史。
- 页面不接触上游密钥。用户未要求在页面做售后提取表单，本章通过独立接口和 curl 验收该小功能。
- 用实际浏览器验收显示、逐字追加、追问及失败提示；不把静态页面检查当作浏览器已运行的证据。

## 8. 验证与过程记录

- 后端可测试代码走 RED→GREEN→必要重构。测试聚焦接口输入、SSE 增量/终止/异常、历史裁剪预算、消息安全和结构化输出真实解析路径。
- 纯 Prompt / 数据类变更用标注样例评估代替为文案写实现镜像单测。提供至少 8 个售后标注样例，覆盖缺字段、前导零、不同诉求、歧义订单和描述里的指令；记录真实模型评估结果，若无可用上游则明确标为未执行。
- 离线集成测试可以使用受控 OpenAI 协议上游验证 HTTP/SSE 和 Pydantic 解析；它不代表真实模型语义效果。
- 每任务完成后立即追记 `dev-notes/ch01.md`；记录用户关键原话、产出路径与结论、拒绝/纠偏、翻车返工。
- 后端实现经独立 spec/代码评审，重要问题修复后复核；页面免代码评审但实际浏览器验收。
- 交付包含启动步骤、PowerShell 友好的 curl.exe 演示命令、测试命令及实测结果、真实上游验收范围、局限和 dev-notes 路径。
- 未获真实 API 配置前，不编造真实模型验收结果；不把测试用 mock 设为产品默认后端。

## 9. 实现者可自行决定的事项

文件模块划分、Python 依赖锁定方式、SSE 使用 FastAPI 内置类型还是官方 StreamingResponse、具体保守 token 估算器、稳定错误码名称、合理输入上限、测试夹具组织、页面视觉细节，均由实现者在不改变上述行为的前提下决定并写入计划/说明。

## 10. 依据（均已先经 Context7）

- FastAPI SSE：<https://fastapi.tiangolo.com/tutorial/server-sent-events/>（文档注明内置 SSE 自 0.135.0 提供）。
- FastAPI SSE 类型：<https://fastapi.tiangolo.com/reference/sse/>。
- FastAPI 自定义响应与取消：<https://fastapi.tiangolo.com/advanced/custom-response/>。
- LangChain 模型与 structured output：<https://docs.langchain.com/oss/python/langchain/models>。
- LangChain OpenAI：<https://docs.langchain.com/oss/python/integrations/chat/openai>。
- Prompt 花括号规则：<https://github.com/langchain-ai/docs/blob/main/src/langsmith/prompt-template-format.mdx>。
- Claude OpenAI 兼容边界：<https://platform.claude.com/docs/en/cli-sdks-libraries/libraries/openai-sdk>。
