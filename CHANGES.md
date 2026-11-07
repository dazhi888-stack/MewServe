# MewHelp 二次开发改动清单

基线：`feat/ch01-customer-service` 分支（PR #1）—— LangChain + FastAPI 纯对话电商客服。
本清单区分「原有代码（基线自带，未改动 / 仅小改）」与「本次新增模块」。

## 一、原有代码（基线保留）

| 文件 | 状态 | 说明 |
|---|---|---|
| `app/main.py` | 保留框架，接入新模块 | FastAPI 路由结构、SSE 流式响应、断连清理（anyio CancelScope）、售后提取接口全部保留；仅在对话管线前插入意图分支与 RAG 上下文注入 |
| `app/service.py` | 仅小改 | `stream_chat` 流式转发、事件契约（meta/token/error/done）、`extract_after_sales` 结构化提取逻辑保留；类型注解改 `ChatDeepSeek`，结构化输出收敛为 json_mode，删除 gpt 系列模型名拦截（DeepSeek 场景不需要） |
| `app/context.py` | 仅小改 | 历史裁剪、token 预算（413）逻辑保留；`build_chat_messages` 新增可选 `faq_context` 参数注入 system prompt |
| `app/schemas.py` | 仅小改 | 请求校验规则保留；`ChatRequest` 新增 `confirm_intent: bool = False` 字段 |
| `app/errors.py` | 仅小改 | 错误码体系保留；`httpx2` → `httpx` |
| `app/diagnostics.py`、`app/prompts.py` | 仅小改 | 安全诊断保留；prompts.py 注册意图模板 |
| `prompts/customer_service.txt`、`prompts/extract_aftersales.txt` | 仅小改 | 客服 Prompt 增加知识库参考段 `{faq_context}`；提取 Prompt 未动 |
| `static/`（内置聊天页） | 未改动 | 原页面继续可用；新增的 intent/rag/confirm 事件会被忽略，拒答文本以 token 流展示不受影响 |
| `scripts/evaluate_prompts.py`、`evals/`、`examples/` | 仅小改 | 评估脚本配置字段名随配置改名 |
| `tests/`（基线 100+ 用例） | 全部通过 | 适配性改名（openai→deepseek、httpx2→httpx），语义未变；`conftest.py` 测试隔离新增 RAG/意图环境变量，测试默认关闭 RAG 与意图分支 |
| `dev-notes/`、`docs/` | 未改动 | 基线设计文档保留 |

## 二、模型替换：DeepSeek（去掉 OpenAI 依赖）

| 文件 | 改动 |
|---|---|
| `app/model.py` | `langchain_openai.ChatOpenAI` → `langchain_deepseek.ChatDeepSeek`；请求级客户端生命周期（创建/关闭、`model_response_scope` 响应句柄回收）契约不变；移除 `http_socket_options`（本地 langchain-openai 1.1.11 不识别该字段） |
| `app/config.py` | `OPENAI_*` 环境变量/字段全部替换为 `DEEPSEEK_*`（DEEPSEEK_BASE_URL 默认官方地址、DEEPSEEK_MODEL 默认 deepseek-chat） |
| `pyproject.toml` | 移除 `langchain-openai` 直接依赖，改为 `langchain-deepseek>=1,<2`；`httpx2` → `httpx`。注：`langchain-deepseek` 底层复用 openai SDK 的兼容协议，这是库的传递依赖，应用代码已无任何 OpenAI 专属调用 |
| `.env.example` | DeepSeek + RAG + 意图配置完整示例 |
| `tests/test_extract.py` | 结构化输出收敛为 json_mode：DeepSeek API 只支持 `json_object`，不支持 OpenAI 的 `json_schema` 线协议，故在配置层（`Literal["json_mode"]`）直接拒绝 json_schema 配置 |

## 三、新增：RAG 模块（Chroma + 电商 FAQ 知识库）

| 文件 | 说明 |
|---|---|
| `app/rag.py`（新增） | `FaqRetriever`：Chroma PersistentClient 持久化（`./chroma_db`）；FAQ 按 `## ` 标题分片；embedding 默认走硅基流动 `BAAI/bge-m3`（OpenAI 兼容接口，中文效果好），未配 key 回退 Chroma 本地默认 embedding；`retrieve(query)` 返回 top_k 条 FAQ 原文，异常时静默降级为空上下文；`ensure_collection()` 集合为空自动导入 |
| `knowledge/faqs.md`（新增） | 17 条 FAQ：商品规格 5 条、运费 2 条、发货时效 3 条、退换货/退款 5 条、物流/售后 2 条 |
| `scripts/ingest_faq.py`（新增） | 手动导入 CLI：`python scripts/ingest_faq.py [faq文件]`（幂等，先清后写） |
| `app/main.py` 接入 | 启动 lifespan 中 `ensure_collection()`；answer 分支检索 top_k FAQ 注入 system prompt；命中时发送 `rag` 事件（`{"faq_used": true}`） |
| 配置项（新增） | `RAG_ENABLED` / `CHROMA_DIR` / `FAQ_FILE` / `RAG_TOP_K` / `EMBEDDING_BASE_URL` / `EMBEDDING_MODEL` / `EMBEDDING_API_KEY` |

## 四、新增：意图识别分支

| 文件 | 说明 |
|---|---|
| `app/intent.py`（新增） | `IntentResult`（category + confidence）；`classify_intent()` 复用基线结构化输出链路（json_mode + Pydantic 校验 + 上游响应校验 + 线协议守卫）；9 类意图：商品规格/运费/退换货/退款/发货时效/订单售后/问候/其他/无关 |
| `route_action()`（新增） | 三档路由：`irrelevant` 类别或置信度 ≤0.60 → 拒绝；(0.60, 0.85) → 弹确认；≥0.85 → 正常回答。注：无关问题即使高置信度也直接拒答（与"拒绝无关问题"的业务目标一致） |
| `prompts/intent.txt`（新增） | 意图分类 Prompt：只分类不回答、防注入、字段规范 |
| `app/main.py` 接入 | SSE 新事件：`intent`（类别+置信度+action）、`confirm`（中档确认卡片文案）、拒答分支直接下发固定文案 token 流并以 `finish_reason: "refused"` 结束；确认分支以 `finish_reason: "need_confirm"` 结束。意图分类与 413 预算校验都在响应开始前完成，保持基线"先校验后开流"契约 |
| 配置项（新增） | `INTENT_ENABLED` / `INTENT_HIGH_THRESHOLD=0.85` / `INTENT_LOW_THRESHOLD=0.60` |
| `tests/test_intent.py`（新增） | 路由三档边界、自定义阈值、非法阈值、FAQ 分片解析的纯逻辑测试 |

## 五、新增：Streamlit 演示页面

| 文件 | 说明 |
|---|---|
| `streamlit_app.py`（新增，仓库根目录） | 聊天气泡界面、SSE 流式逐字渲染、意图置信度标签、中档置信度"✅确认回答 / ❌取消"交互卡片、拒答展示、新对话按钮；失败回复不进入后续历史（沿用基线约定） |
| 运行方式 | 先启动后端 `uvicorn app.main:app --port 8000`，再 `streamlit run streamlit_app.py` |

## 六、测试与验证

- `pytest`：**109 项全部通过**（基线 100+ 项语义保留 + 新增 9 项意图/RAG 测试）。
- 真实 DeepSeek API 冒烟验证（本机 8765 端口实测）：
  1. 「退货运费谁承担？」→ 意图 return_exchange 0.85 → RAG 命中 → 流式回答 ✓
  2. 「帮我写一个 Python 冒泡排序」→ irrelevant 0.98 → 固定文案拒答 ✓
  3. 「那个大概多久啊」→ delivery_time 0.72 → confirm 事件 + need_confirm ✓
  4. 确认后 `confirm_intent=true` 重发 → 跳过意图直接 RAG 回答 ✓
  5. 多轮追问携带 history → 正常理解上下文 ✓

## 七、已知边界

- `langchain-deepseek` 底层仍依赖 openai SDK 协议（库传递依赖），应用层已无 OpenAI 调用；如需彻底移除需自研 HTTP 客户端，不在本次范围。
- 结构化输出仅支持 json_mode：DeepSeek API 不支持 OpenAI 的 json_schema 参数。
- 意图识别只看当前消息（不携带 history），多轮指代模糊时依赖确认机制兜底。
- Streamlit 演示页未纳入 pytest 门禁（与基线对静态页的处理一致），需实际运行验收。

---

# 二次开发第三批（2026-11-07）：兜底式路由 + RAG 深化 + 评估体系

## 八、意图路由改兜底式（调整）

| 位置 | 改动 |
|---|---|
| `app/intent.py` | `route_action()` 由三档改为**两档兜底式**：≤0.60 或 `irrelevant` → 拒绝，其余一律回答；新增 `needs_hedge()`：中间档 (0.60, 0.85) 回答但话术带对冲，不再把确认卡片抛给用户 |
| `app/context.py` | 新增 `HEDGE_INSTRUCTION` 对冲指令；`build_chat_messages()` 加 `hedge` 参数，中间档回答前在 system prompt 追加"先确认理解、不给绝对承诺"话术 |
| `app/main.py` | 移除 confirm 事件分支（保留 `confirm_intent` 兼容路径与前端处理逻辑）；`intent` 事件新增 `hedged` 字段；拒答文案补充人工客服/留言引导 |
| `tests/test_intent.py` | 路由边界测试同步改为两档语义；新增 `TestNeedsHedge` 用例 |

## 九、RAG 混合检索深化（新增）

| 位置 | 改动 |
|---|---|
| `app/rag.py` | 检索管线升级为 **BM25 + 向量双路召回 → RRF 融合 → bge-reranker 精排**。BM25 为纯 Python 实现（中文按字 + bigram 切分，零新增依赖）；RRF k=60；重排走硅基流动 `BAAI/bge-reranker-v2-m3`（失败自动降级为 RRF 序） |
| 查询改写 | 新增 `rewrite_query()`：多轮场景先用模型把追问改写成独立检索问题（如"那质量问题呢"→"质量问题退货运费谁承担"），失败回退原消息 |
| `app/main.py` | 检索移入线程池（`asyncio.to_thread`）避免阻塞事件循环 |
| 配置项（新增） | `RAG_CANDIDATE_K=12` / `RERANK_ENABLED` / `RERANK_MODEL` / `QUERY_REWRITE_ENABLED` |
| `tests/test_rag.py`（新增） | BM25 打分/空索引、中文分词、RRF 融合、FAQ 分片的纯逻辑测试 |

## 十、离线评估体系（新增）

| 位置 | 说明 |
|---|---|
| `evals/golden_qa.jsonl`（新增） | 45 条标注样本：意图标签（9 类）+ FAQ 相关题目的期望命中条目关键词 |
| `evals/run_eval.py`（新增） | 离线评估管线：意图准确率 + 每类 Precision/Recall/F1 + 混淆矩阵 + 误分类明细；检索 Recall@1/3/5 + MRR + 未命中明细；报告落盘 `evals/report.md` |
| 评估结论（首次全量） | 意图准确率 **86.7%**（45 条）；检索 Recall@1 **94.1%** / Recall@3 97.1% / **MRR 0.956**；A/B 对比：混合+重排比纯向量 Recall@1 **91.2% → 94.1%**、MRR **0.936 → 0.956** |

## 十一、测试与验证（第三批后）

- `pytest`：**122 项全部通过**。
- 运行评估：`uv run python evals/run_eval.py`（支持 `--part intent|retrieval`、`--limit N`）。
