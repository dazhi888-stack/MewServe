# MewServe · 基于 RAG 的电商智能客服系统

LangChain + FastAPI + DeepSeek 构建的电商客服服务：混合检索 RAG、兜底式意图路由、订单查询工具与语义缓存，全链路 SSE 流式输出，157 项自动化测试，附离线评估体系。

> 本项目基于开源项目 [huashenxiushi/MewHelp](https://github.com/huashenxiushi/MewHelp)（LangChain + FastAPI 纯对话流式客服基线）二次开发。基线提供多轮对话、SSE 流式转发与上下文预算管理；RAG 混合检索、意图路由、订单工具、语义缓存与离线评估体系均为本人独立设计与实现（详见 [CHANGES.md](CHANGES.md) 分批改动清单）。

## 核心特性

### 1. 混合检索 RAG
- **双路召回**：BM25（中文单字 + bigram 分词，无外部分词器依赖）+ 向量检索（BAAI/bge-m3）并行召回
- **RRF 融合**：倒数排名融合（k=60），避免 BM25 分数与余弦相似度量纲不可比的问题
- **精排**：融合后的 top-12 候选经 `BAAI/bge-reranker-v2-m3`（cross-encoder）重排取 top-k 注入 prompt
- **多轮查询改写**：追问（"它多少钱"）先由 LLM 改写为语义完整的独立问题再检索，失败自动降级
- 离线评估：混合检索较纯向量召回 **Recall@1 91.2% → 94.1%**（MRR 0.956）

### 2. 兜底式意图路由
- LLM json_mode 结构化输出意图类别与置信度，Pydantic 二次校验
- 两档阈值（0.60 / 0.85）路由：高置信度回答 / 中间档回答并注入对冲话术 / 低置信度或无关问题拒答转人工
- `irrelevant` 类别无论置信度数值一律拒答，防高置信误答；阈值经 45 条标注样本评估集校准（意图准确率 86.7%）

### 3. 订单查询工具（Mock 数据）
- **两级触发**：消息含订单号（`MH` + 9~12 位，lookaround 正则支持紧贴中文）→ 正则直查，零模型开销；订单类意图但无订单号 → 一次 json_mode 结构化抽取兜底
- 工具结果与 FAQ 同通道注入 system prompt，**不引入 function-calling 多轮循环**，保持 SSE"先校验后开流"契约
- Mock 数据收件人姓名与手机号已脱敏；数据源抽象化，替换为真实订单系统只改一层

### 4. 语义缓存
- 基于 Chroma 向量相似度（阈值 0.92）复用首轮回答，命中时**跳过订单工具、RAG 与模型调用**
- 只缓存上下文无关的首轮问答（多轮回答复用会答错）
- **实体比对双重防线**：含订单号的问题不入缓存（答案含个人信息）；缓存条目带 `order_ids` 实体标注，查询含订单号时必须完全匹配才可命中，杜绝"换订单号命中别人答案"的串单事故
- SHA-1 键 upsert 去重 + FIFO 淘汰 + `POST /api/cache/clear` 运维接口

### 5. 工程化
- SSE 全链路流式输出；意图识别、RAG、预算校验等所有可失败操作在开流前完成（413 等错误保持标准 HTTP 语义）
- 阻塞调用（BM25 / rerank / Chroma）经 `asyncio.to_thread` 下放线程池，旁路失败静默降级不传染主链路
- pytest **157 项**：语义缓存测试注入确定性伪 embedding（实现 chromadb EmbeddingFunction 协议），全量无网络秒级回归
- uv 依赖管理；前端原生 JS 与 Streamlit 双演示页，SSE 事件（intent / rag / tool / cache）均有可视化标注

## 请求处理链路

```
请求 ──> 多轮历史校验 ──> 意图识别(json_mode) ──> 路由决策
                                              │
              ┌─────────────── answer 分支 ───┤── refuse 分支
              ▼                                              ▼
      订单工具(正则直查/抽取兜底)                    拒答话术 + 人工引导
              │
      多轮? ──> LLM 查询改写 ──> 混合检索(BM25+向量→RRF→rerank)
              │
      语义缓存查询 ──命中──> 回放缓存答案（跳过模型）
              │ 未命中
      消息构建(预算校验) ──> DeepSeek 流式生成 ──> SSE 转发 ──> 写回缓存
```

## 快速开始

要求 Python 3.11+ 与 [uv](https://docs.astral.sh/uv/getting-started/installation/)。

```bash
uv sync --dev
cp .env.example .env    # 填入 DEEPSEEK_API_KEY 与 EMBEDDING_API_KEY
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

浏览器打开 <http://127.0.0.1:8000>，接口文档在 `/docs`；Streamlit 演示页另开终端：

```bash
uv run streamlit run streamlit_app.py
```

| 环境变量 | 说明 |
|---|---|
| `DEEPSEEK_API_KEY` | 必填，对话/意图/改写模型（deepseek-chat） |
| `EMBEDDING_API_KEY` | 推荐，硅基流动 bge-m3；留空回退 Chroma 本地 embedding |
| `RAG_ENABLED` / `INTENT_ENABLED` / `SEMANTIC_CACHE_ENABLED` / `ORDER_TOOL_ENABLED` | 各模块独立开关，false 时回退纯对话基线行为 |
| `INTENT_HIGH_THRESHOLD` / `INTENT_LOW_THRESHOLD` | 意图路由两档阈值（默认 0.85 / 0.60） |
| `CACHE_SIMILARITY_THRESHOLD` / `CACHE_MAX_ENTRIES` | 语义缓存命中阈值（0.92）与容量上限（1000） |

## 主要接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/chat/stream` | SSE 流式对话（事件：meta / intent / rag / tool / cache / token / done / error） |
| GET | `/api/orders/{order_id}` | Mock 订单直接查询（演示/联调） |
| POST | `/api/cache/clear` | 清空语义缓存 |
| GET | `/health` | 健康检查（不依赖模型连通性） |

## 测试与评估

```bash
uv run pytest -q                                # 157 项，无网络依赖
uv run python evals/run_eval.py --part retrieval # 检索评估：Recall@k / MRR
uv run python evals/run_eval.py --part intent    # 意图评估：准确率 / 混淆矩阵
```

评估集 `evals/golden_qa.jsonl`：45 条标注样本（意图标签 + 检索关键词）。当前报告见 [evals/report.md](evals/report.md)。

## 项目结构

```
app/
  main.py      # FastAPI 入口、对话管线编排、SSE 事件
  rag.py       # BM25+向量双路召回、RRF、rerank、查询改写
  intent.py    # 意图分类与兜底式路由
  tools.py     # 订单查询工具（正则直查 + LLM 抽取兜底）
  cache.py     # 语义缓存（实体比对双重防线）
  service.py   # 流式转发与结构化输出
  context.py   # prompt 模板与消息构建
  config.py    # pydantic-settings 配置校验
evals/         # 评估集与评估脚本
knowledge/     # 电商 FAQ 知识库源文件
prompts/       # 客服 / 意图 / 售后提取 prompt 模板
tests/         # 157 项测试
static/        # 原生 JS 前端演示页
```

## 已知局限与 Roadmap

- [ ] 会话/用户级缓存隔离（当前全局共享，仅建议缓存与用户无关的问答）
- [ ] 意图置信度改用 logprobs 或独立分类头（LLM 自报置信度校准性差，当前靠兜底架构兜住）
- [ ] 评估集扩容 + RAGAS 生成质量评估（faithfulness / answer relevance）
- [ ] 认证鉴权与限流
- [ ] 订单工具接入真实订单系统（工具层已抽象）

## 致谢

- 基线项目：[huashenxiushi/MewHelp](https://github.com/huashenxiushi/MewHelp) —— 多轮对话、SSE 流式转发、上下文预算管理与前端页面骨架
- 模型与模型服务：DeepSeek、硅基流动（BAAI/bge-m3 / bge-reranker-v2-m3）
