# Ch01 Customer Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从空仓库交付能通过 curl 和浏览器验证的电商客服纯对话、SSE、有限上下文与售后结构化提取。

**Architecture:** 无状态 FastAPI 后端接收 message + history，通过 LangChain PromptTemplate 和 ChatOpenAI 直连 OpenAI Chat Completions 兼容上游。服务端在调用前裁剪完整历史轮次，再流式转发正文 delta；同源静态聊天页在页面内存保存已完成问答。结构化提取独立调用 with_structured_output，不接工具与 Agent。

**Tech Stack:** Python >=3.11；FastAPI、Uvicorn、LangChain Core / langchain-openai、Pydantic Settings；pytest + HTTPX；原生 HTML/CSS/JavaScript；uv 管理依赖和锁文件。

**Spec:** `docs/superpowers/specs/2026-11-05-ch01-customer-service-design.md`（用户已批准，必须与本计划一起阅读）。

## Global Constraints

- “Python + FastAPI + LangChain。”实际只安装所需的 langchain-core 与 langchain-openai，不安装 Agent/LangGraph。
- “应用通过 langchain-openai 的 ChatOpenAI 直连上游，统一 OpenAI Chat Completions 协议。”
- “上游地址、模型名、密钥从 .env / 进程环境读取；密钥不进入浏览器、日志、版本控制或错误详情。”
- “不引入数据库、Redis、LangGraph、工具调用、Agent 循环、检索、订单系统或真实售后执行。”
- “每次请求包含 message 和可选 history，后端裁剪后直接调用模型，不保存会话。”
- “实现只允许 json_mode / json_schema，不使用隐含的 function_calling 默认值”。
- “客户端断连时取消上游请求，释放流和连接。”
- “离线测试或模拟上游通过不能冒充真实模型验收通过。”
- 聊天页面按用户的 Vibe Coding 例外处理：免 brainstorm、TDD、code review；仍须浏览器实际验收。
- 先通过 Context7 核对具体库/API，再使用；固定技术选型遇阻停下询问，不能自行替换。
- 每个阶段和任务完成后及时向 `dev-notes/ch01.md` 追加四项记录；每次 Git 操作前逐文件检查并向用户解释。

## Review Focus

1. 中文、emoji、换行、用户花括号：不得损坏文本或变成模板变量；任务 2/3 用实际字符串断言。
2. 伪造 system、错序历史或孤立 assistant：请求在上游调用前拒绝；任务 2/3 测试。
3. 流已开始后超时/中断/取消：不能产生成功 done、泄露错误体或继续消耗上游；任务 3 测试并做真实 HTTP 冒烟。
4. 切换上游时结构化 API 的隐含 tool-calling fallback：任务 4 检查真实 ChatOpenAI 序列化请求中不存在 tools/tool_choice，并覆盖错误模式。
5. 缺密钥、上游 401/429、无效 JSON：页面可打开，模型接口返回稳定公开错误且不含原始异常中的密钥；任务 1/3/4 覆盖。

## 文件与职责

- `pyproject.toml`、`uv.lock`、`.gitignore`、`.env.example`：依赖、运行配置与忽略规则。
- `app/config.py`：Settings、环境校验、get_settings；`app/schemas.py`：请求/响应字段与轮次验证。
- `app/prompts.py` + `prompts/customer_service.txt`、`prompts/after_sales.txt`：模板加载与渲染；Prompt 的语义效果由标注样例验证。
- `app/context.py`：预算估算与完整轮次裁剪；`app/errors.py`：稳定公开错误码与上游异常映射。
- `app/model.py`：ChatOpenAI 工厂与依赖；`app/service.py`：模型流式调用与结构化提取。
- `app/main.py`：FastAPI 路由、SSE 编码、静态资源、health；`app/__init__.py`：包标记。
- `static/index.html`、`static/styles.css`、`static/chat.js`：同源聊天页。
- `tests/test_config.py`、`test_context.py`、`test_chat.py`、`test_extract.py` 及 `tests/conftest.py`：后端验证。
- `evals/after_sales.jsonl`、`evals/chat_cases.jsonl`：带标注的 Prompt 评估集；`scripts/evaluate_prompts.py`：真实模型评估。
- `examples/*.json`、`README.md`：可直接运行的请求体和 PowerShell/curl 验收命令。
- `artifacts/`：忽略的本地验证产物；不得把密钥写入任何产物。

## Task 1: 可启动的配置边界与 health

**Files:** 创建 pyproject/lock、ignore、env example、app 包、config/errors/main；测试 `tests/test_config.py`。

**Interfaces:**
- 产出 `Settings(BaseSettings)`、`get_settings() -> Settings`；默认缺失的 OpenAI 地址/模型/密钥不阻止 health 启动。
- 环境变量：`OPENAI_BASE_URL`、`OPENAI_MODEL`、`OPENAI_API_KEY`、`STRUCTURED_OUTPUT_METHOD=json_mode`、`CONTEXT_WINDOW_TOKENS=8192`、`MAX_OUTPUT_TOKENS=512`、`TOKEN_SAFETY_MARGIN=512`、`HISTORY_MAX_MESSAGES=20`、`REQUEST_TIMEOUT_SECONDS=60`、`SHOP_NAME=MewHelp`。
- 产出 `app.main:app`、GET `/health` → `{"status":"ok","model_configured":false|true}`；`AppError(code, message, status_code)` 供后续任务使用。
- Settings 中密钥用 SecretStr；配置为空可启动，配置值非法则明确报错；输出预留与安全余量之和必须小于上下文总预算。

- [x] 按 using-git-worktrees 检查隔离方式。定稿 spec/计划/日志逐文件审查后形成初始文档快照，再建立功能分支/隔离工作区；不拉取、覆盖或推送未知远端历史。
- [x] 查证剩余配置/测试 API，安装项目测试依赖，生成可复现锁文件。配置和安装属于本任务，不单独制造脚手架任务。
- [x] RED：先写 `test_health_without_env`（200 + model_configured=false）、`test_settings_load_dotenv`（临时 env 文件确实改变模型配置）、`test_settings_budget_validation`（非法预算拒绝）、`test_secret_not_in_repr`；运行 `uv run pytest tests/test_config.py -q`，确认失败源于尚未实现行为。
- [x] GREEN：实现上述接口及 `.env.example`，完成同名测试；缺配置的模型请求在任务 3/4 覆盖。
- [x] 跑 `uv run pytest tests/test_config.py -q`，断言全部通过；逐文件检查，日志记录 RED/GREEN 结果并提交任务快照。

## Task 2: 有界请求、模板与上下文

**Files:** 创建 schemas/context/prompts、两个文本模板及 `tests/test_context.py`、evals 两个标注集。

**Interfaces:**
- `HistoryMessage(role: Literal['user','assistant'], content: str)`；`ChatRequest(message: str, history: list[HistoryMessage]=[])`，历史必须以 user 开始、以 assistant 结束并严格交替。
- 接受最多 100 条 history，每条和当前 message 最多 8000 字符；禁止空白内容、未知字段、system/tool。最多保留 20 条（或 HISTORY_MAX_MESSAGES 指定的偶数）历史。
- `PreparedChat(messages: list[BaseMessage], estimated_input_tokens: int, dropped_messages: int)`。
- `build_chat_messages(request: ChatRequest, settings: Settings) -> PreparedChat`。
- `estimate_tokens(messages: Sequence[BaseMessage]) -> int` 使用每条 UTF-8 字节数 + 固定消息开销的保守估算；预算条件为 input_estimate + MAX_OUTPUT_TOKENS + TOKEN_SAFETY_MARGIN <= CONTEXT_WINDOW_TOKENS。该算法不称为任意上游的精确 tokenizer。
- `build_extraction_messages(description: str, settings: Settings) -> list[BaseMessage]`；超预算与聊天一致地在模型调用前拒绝。

- [ ] RED：测试 `test_trims_oldest_complete_turns`、`test_message_limit_preserves_pairs`、`test_system_and_latest_user_never_trimmed`、`test_oversized_current_message_rejected`；分别断言最老轮次被移除、保留顺序、总估算预算成立、超限抛稳定错误。
- [ ] RED：验证错序历史和 system 输入拒绝；字符串 `订单{001}，碎了\n😢` 拼装后原样保留且 system 只有服务端模板；运行 `uv run pytest tests/test_context.py -q` 确认 RED。
- [ ] GREEN：实现 schema、PromptTemplate 和消息裁剪接口；按完整问答对移除，不做摘要、不静默截断当前输入；运行 config/context 测试至通过。
- [ ] Prompt/数据验证替代文案 TDD：建立至少 8 个售后标注样例 + 2 组客服对话样例，覆盖前导零、缺字段、主要售后诉求、歧义订单、指令注入、不能假称退款已完成。此时只检查标注/输入可读，真实模型效果留到任务 4/6 执行，不能先写“评估通过”。
- [ ] 逐文件检查，追加本任务日志并提交快照。

## Task 3: 真实 ChatOpenAI 调用路径与 SSE 对话

**Files:** 创建 model/service、扩展 main/errors，测试 `tests/test_chat.py`、共享测试夹具 `tests/conftest.py`。

**Interfaces:**
- `create_model(settings: Settings) -> ChatOpenAI`：只用 Chat Completions，读取 base_url/model/key，设置输出上限、timeout、显式禁止不必要重试；不强发 temperature，不启用额外 usage 请求。
- 产出可覆盖的 `get_model` 依赖；缺配置抛 503 `model_not_configured`。
- `stream_chat(prepared: PreparedChat, model: ChatOpenAI) -> AsyncIterator[tuple[str, dict]]`：meta、非空正文 token、done 或 error；上游以 model.astream 迭代。
- POST `/api/chat/stream` 返回官方 StreamingResponse；SSE data=json.dumps(..., ensure_ascii=False)，事件以空行结束，带 Cache-Control: no-cache 和 X-Accel-Buffering: no。

- [ ] RED：先写 `/api/chat/stream` 首轮 SSE 契约、第二轮上游请求含第一轮 user/assistant、空历史互不串话、请求错误在 headers 开始前返回 422/503 的测试。上游用受控 OpenAI 协议响应，不绕过真实 ChatOpenAI 序列化/反序列化。
- [ ] RED：写增量门控测试：在第二个上游 chunk 尚未放行时已能消费第一个 token；中文换行/花括号不变；上游结束前不能产生 done。
- [ ] RED：写流中超时、401/429、空正文或工具/思考块、客户端取消的测试；断言 error 无秘密、取消使上游迭代器结束、失败后无成功 done。运行 `uv run pytest tests/test_chat.py -q`，确认失败原因。
- [ ] GREEN：实现工厂、依赖、流式生成器和 SSE 路由；错误按稳定公开码映射，始终关闭/取消上游流，不泄露原始异常文本。
- [ ] 跑 config/context/chat 测试；真实网络冒烟的增量时间和 curl 证据在任务 6 记录，不以 TestClient 拼接结果证明网络实时性。
- [ ] 逐文件检查，追加本任务日志并提交快照。

## Task 4: 售后结构化提取与 Prompt 评估入口

**Files:** 扩展 schemas/prompts/service/main/errors；测试 `tests/test_extract.py`；创建 `scripts/evaluate_prompts.py`。

**Interfaces:**
- `AfterSalesRequest(description: str)`，非空且 <=8000 字符。
- `AfterSalesResult(order_id: str|None, request_type: Literal['refund','return_refund','exchange','repair','logistics','other','unknown'], expected_solution: str|None)`；三个字段必填、未知字段禁止。
- `extract_after_sales(description: str, settings: Settings, model: ChatOpenAI) -> AfterSalesResult`（async）。
- POST `/api/after-sales/extract` 成功直接返回上述三个字段；不包装多余业务字段。
- 显式调用 `with_structured_output(AfterSalesResult, method=settings.structured_output_method, include_raw=True)`；strict 只在 json_schema 且文档允许时设置，禁止意外 tool-calling fallback。
- 评估入口：`uv run python scripts/evaluate_prompts.py --output artifacts/ch01-eval.json`，保留合成样例、实际结果、字段评分/客服行为判据、耗时和模型名，不保存 key。配置缺失为未执行并非通过；失败退出非 0。

- [ ] RED：用真实 ChatOpenAI + 受控 OpenAI HTTP 响应测试三个字段、缺字段 null、订单前导零；无效 JSON/非法枚举/缺必填/多余字段/不支持 response_format 返回稳定错误。
- [ ] RED：检查 json_mode 和 json_schema 发出的 HTTP JSON，必须明确 response_format，且不含 tools/tool_choice。对库可能发生的隐含降级要拒绝而非照发工具调用；输入超预算时不得访问上游。运行 `uv run pytest tests/test_extract.py -q` 确认 RED。
- [ ] GREEN：实现 with_structured_output 调用、schema 解析和公开错误映射，跑所有后端测试。
- [ ] 建立并验证评估命令。若用户已在本地 .env 填好可用上游，用标注集跑真实提取和客服行为评估，记录逐样例结果；没有可用上游则保持待验，不伪造数据或跨供应商通过结论。
- [ ] 逐文件检查，追加本任务日志并提交快照。

## Task 5: 聊天页面（Vibe Coding，免 TDD/评审）

**Files:** 创建 static 三文件，main 增加 GET `/` 和资源挂载。

**Interfaces:** 只消费任务 3 的 message/history POST 与 meta/token/done/error 事件；不改后端契约，不做售后表单扩展。

- [ ] 直接制作中文客服页面：消息气泡、输入与发送、忙碌状态、新对话、手机宽度布局；同源资源无额外前端依赖。
- [ ] fetch 流式读取 + 持续 TextDecoder/SSE 缓冲；逐字渲染收到的正文，仅成功 done 后追加完整问答到 history；EOF 无终止事件为失败。
- [ ] 用 textContent 输出文本；生成期间禁止重复发送；失败/断流明确提示并避免污染 history；新对话/刷新清空页面内存。
- [ ] 使用实际浏览器检查视觉、输入、逐字更新和第二轮请求携带 history；真实模型可用时检查追问效果，否则区分受控上游交互验证和未完成的真实模型验收。
- [ ] 逐文件检查，追加本任务日志并提交快照；不要将聊天页加入后端 TDD 或代码审阅门禁。

## Task 6: 验收、后端整体评审与交付

**Files:** README、examples 请求文件、必要的后端修复及回归测试、dev-notes；本地 artifacts 放忽略目录。

- [ ] 写 README 与可复用请求文件，含 `uv sync --dev`、复制/填写 .env、`uv run uvicorn app.main:app --host 127.0.0.1 --port 8000`，以及 curl.exe 使用 `--data-binary @examples/xxx.json` 避免 PowerShell JSON 引号问题的首轮、第二轮、售后演示。
- [ ] 全量运行 `uv run pytest -q`；按需运行静态检查，不堆叠与本章无关的工具。
- [ ] 实际启动服务，用 curl.exe 观察多个事件到达时序；验证两轮请求和提取接口；浏览器完成本章交互验收。明确记录使用真实模型还是受控测试上游。
- [ ] 若真实 .env 未配置，先完成全部独立工作，交付可运行命令与明确的待验条件；不得把该状态写成三个模型验收都已通过。
- [ ] 使用 requesting-code-review 分派独立审阅者做后端整体 spec/代码评审，范围不含聊天页面；重要问题先写回归测试、修复后重跑并复核。评审结论立即追记日志。
- [ ] 按 verification-before-completion 取得新鲜验证证据；按 finishing-a-development-branch 完成获授权的本地整合，外部推送/PR 按用户选择处理。完成时追加 finish 四项记录。
- [ ] 最终交付实际运行状态、功能演示命令、测试/评估结果、已知局限和 `dev-notes/ch01.md` 路径；未完成项不能计为完成。

## 计划自审（2026-11-05）

- Spec 覆盖：SSE/错误/取消→任务 3；Prompt/预算→任务 2；with_structured_output→任务 4；页面→任务 5；配置/保密→任务 1；演示/过程/评审→各任务 + 任务 6。
- 类型与契约：message/history、SSE token 的 text 字段、三字段提取接口在各任务一致；无 conversation_id 或后端会话存储。
- Review Focus 五项已落入对应任务的具体验证；页面保留用户流程例外。
- 计划状态：自审完成，用户于 2026-11-05 选择“subagent-driven 模式”，确认按此计划执行；尚未安装依赖或编写产品代码。
