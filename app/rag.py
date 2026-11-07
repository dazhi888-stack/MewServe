"""RAG 模块：Chroma 向量库 + BM25 混合检索 + 重排（二次开发新增并深化）。

检索管线（retrieve）：
1. 双路召回：Chroma 向量检索（bge-m3）+ 本地 BM25（中文 bigram 分词，零额外依赖）
2. RRF 倒数排名融合（k=60），合并两路候选
3. bge-reranker-v2-m3 精排（硅基流动 rerank 接口；失败自动降级为 RRF 排序）
4. 返回 top_k 条 FAQ 原文

其他要点：
- 向量库持久化在 settings.chroma_dir，FAQ 源文件为 Markdown，按 `## ` 分片。
- Embedding 优先走 OpenAI 兼容接口（默认硅基流动 BAAI/bge-m3）；
  未配置 key 时回退到 Chroma 自带的本地默认 embedding（ONNX MiniLM）。
- 服务启动时 ensure_collection()：集合不存在或为空则自动导入 FAQ；
  重启后从集合回读文档重建 BM25 索引。
"""

import logging
import math
import re
from collections import defaultdict
from pathlib import Path

import chromadb
import httpx
from chromadb.utils import embedding_functions

from app.config import Settings

logger = logging.getLogger("mewhelp.rag")

COLLECTION_NAME = "ecommerce_faq"

_RRF_K = 60


# ---------- BM25（纯 Python 实现，中文按字 + bigram 切分） ----------

_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
_ASCII_WORD = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> list[str]:
    """中文按单字与二元组切分，英文/数字按词切分。无需分词器依赖。"""
    tokens: list[str] = []
    tokens.extend(word.lower() for word in _ASCII_WORD.findall(text))
    for run in _CJK_RUN.findall(text):
        tokens.extend(run)
        tokens.extend(f"{run[i]}{run[i + 1]}" for i in range(len(run) - 1))
    return tokens


class Bm25Index:
    """轻量 BM25 (k1=1.5, b=0.75)。文档量在几百条量级，直接线性扫描足够快。"""

    def __init__(self, documents: list[str]) -> None:
        self._doc_tokens = [_tokenize(doc) for doc in documents]
        self._doc_count = len(self._doc_tokens)
        self._avg_len = (
            sum(len(t) for t in self._doc_tokens) / self._doc_count
            if self._doc_count else 0.0
        )
        self._tf: list[dict[str, int]] = [defaultdict(int) for _ in self._doc_tokens]
        for i, tokens in enumerate(self._doc_tokens):
            for token in tokens:
                self._tf[i][token] += 1
        self._idf: dict[str, float] = {}
        df: dict[str, int] = defaultdict(int)
        for tf in self._tf:
            for token in tf:
                df[token] += 1
        for token, freq in df.items():
            # BM25 标准 IDF（Lucene 变体，避免负值）。
            self._idf[token] = math.log(1 + (self._doc_count - freq + 0.5) / (freq + 0.5))

    def scores(self, query: str) -> list[float]:
        if not self._doc_count:
            return []
        k1, b = 1.5, 0.75
        query_tokens = _tokenize(query)
        result = [0.0] * self._doc_count
        for token in query_tokens:
            idf = self._idf.get(token)
            if not idf:
                continue
            for i, tf in enumerate(self._tf):
                freq = tf.get(token)
                if not freq:
                    continue
                norm = 1 - b + b * len(self._doc_tokens[i]) / (self._avg_len or 1)
                result[i] += idf * freq * (k1 + 1) / (freq + k1 * norm)
        return result


def _rrf_fuse(rankings: list[list[str]], top_n: int) -> list[str]:
    """倒数排名融合：score(d) = Σ 1/(k + rank)，rank 从 1 开始。"""
    fused: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for position, doc_id in enumerate(ranking, start=1):
            fused[doc_id] += 1.0 / (_RRF_K + position)
    ordered = sorted(fused.items(), key=lambda item: -item[1])
    return [doc_id for doc_id, _ in ordered[:top_n]]


class FaqRetriever:
    """进程内共享一个 PersistentClient；retrieve 返回命中的 FAQ 原文列表。"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = chromadb.PersistentClient(path=settings.chroma_dir)
        self._collection = None
        self._docs: dict[str, str] = {}  # id -> document 原文（BM25 索引用）
        self._bm25: Bm25Index | None = None

    # ---------- embedding ----------

    def _build_embedding(self):
        settings = self._settings
        if settings.embedding_configured:
            # OpenAI 兼容 embeddings 接口（硅基流动 / DeepSeek 兼容网关均可）。
            return embedding_functions.OpenAIEmbeddingFunction(
                api_key=settings.embedding_api_key.get_secret_value(),
                model_name=settings.embedding_model,
                api_base=settings.embedding_base_url,
            )
        logger.warning("未配置 EMBEDDING_API_KEY，回退到 Chroma 默认本地 embedding。")
        return embedding_functions.DefaultEmbeddingFunction()

    # ---------- collection ----------

    def ensure_collection(self) -> None:
        """幂等初始化：集合为空时自动导入 FAQ；随后重建 BM25 索引。"""
        try:
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=self._build_embedding(),
                metadata={"hnsw:space": "cosine"},
            )
            if self._collection.count() == 0:
                self.ingest()
            else:
                self._rebuild_bm25()
        except Exception:
            self._collection = None
            logger.exception("Chroma 知识库初始化失败，RAG 检索将不可用。")

    @property
    def available(self) -> bool:
        return self._collection is not None

    def _rebuild_bm25(self) -> None:
        """从集合回读全部文档，重建 BM25 索引（重启/重导后调用）。"""
        data = self._collection.get(include=["documents"])
        ids = data.get("ids") or []
        documents = data.get("documents") or []
        self._docs = dict(zip(ids, documents))
        self._bm25 = Bm25Index(list(self._docs.values())) if self._docs else None

    # ---------- ingest ----------

    @staticmethod
    def parse_faq(text: str) -> list[dict]:
        """按 `## Q: 标题` 分片，返回 [{id, document}]。"""
        chunks: list[dict] = []
        blocks = re.split(r"\n(?=## )", text)
        for index, block in enumerate(blocks):
            block = block.strip()
            # 有效 FAQ 片段 = `## ` 开头且带正文（存在换行）。
            if not block.startswith("## ") or "\n" not in block:
                continue
            chunks.append({"id": f"faq-{index:04d}", "document": block})
        return chunks

    def ingest(self, faq_path: str | None = None) -> int:
        """（重新）导入 FAQ 知识库，返回导入条数。"""
        if self._collection is None:
            raise RuntimeError("知识库集合尚未初始化，请先调用 ensure_collection()。")
        path = Path(faq_path or self._settings.faq_file)
        chunks = self.parse_faq(path.read_text(encoding="utf-8"))
        if not chunks:
            logger.warning("FAQ 文件 %s 未解析出任何条目。", path)
            return 0
        # 幂等：先清空旧条目再写入，便于更新知识库后重跑。
        existing = self._collection.get()
        if existing["ids"]:
            self._collection.delete(ids=existing["ids"])
        self._collection.add(
            ids=[item["id"] for item in chunks],
            documents=[item["document"] for item in chunks],
        )
        self._rebuild_bm25()
        logger.info("已导入 %d 条 FAQ 到 Chroma。", len(chunks))
        return len(chunks)

    # ---------- retrieve ----------

    def _vector_recall(self, query: str) -> list[str]:
        result = self._collection.query(
            query_texts=[query], n_results=min(self._settings.rag_candidate_k,
                                               self._collection.count()),
        )
        ids = result.get("ids") or []
        return list(ids[0]) if ids else []

    def _bm25_recall(self, query: str) -> list[str]:
        if self._bm25 is None:
            return []
        doc_ids = list(self._docs)
        ranked = sorted(
            zip(doc_ids, self._bm25.scores(query)), key=lambda pair: -pair[1],
        )
        return [
            doc_id for doc_id, score in ranked[:self._settings.rag_candidate_k]
            if score > 0
        ]

    def _rerank(self, query: str, candidates: list[str]) -> list[str] | None:
        """调用硅基流动 rerank 接口精排；失败返回 None（调用方降级为 RRF 序）。"""
        settings = self._settings
        if not settings.rerank_enabled or not settings.embedding_configured:
            return None
        documents = [self._docs[doc_id] for doc_id in candidates]
        try:
            response = httpx.post(
                f"{settings.embedding_base_url.rstrip('/')}/rerank",
                headers={"Authorization": f"Bearer {settings.embedding_api_key.get_secret_value()}"},
                json={
                    "model": settings.rerank_model,
                    "query": query,
                    "documents": documents,
                    "top_n": settings.rag_top_k,
                    "return_documents": False,
                },
                timeout=15,
            )
            response.raise_for_status()
            results = response.json().get("results") or []
            return [
                candidates[item["index"]]
                for item in sorted(results, key=lambda r: -r["relevance_score"])
            ]
        except Exception:
            logger.warning("rerank 调用失败，降级为 RRF 融合排序。", exc_info=True)
            return None

    def retrieve(self, query: str) -> list[str]:
        """混合检索：向量 + BM25 召回 → RRF 融合 → reranker 精排 → top_k 原文。

        任何环节不可用或异常时降级（无 BM25 → 纯向量；rerank 失败 → RRF 序）。
        """
        if self._collection is None:
            return []
        try:
            vector_ids = self._vector_recall(query)
            bm25_ids = self._bm25_recall(query)
            candidates = _rrf_fuse([vector_ids, bm25_ids], top_n=self._settings.rag_candidate_k)
            if not candidates:
                return []
            reranked = self._rerank(query, candidates)
            final_ids = (reranked or candidates)[:self._settings.rag_top_k]
            return [self._docs[doc_id] for doc_id in final_ids if doc_id in self._docs]
        except Exception:
            logger.exception("RAG 检索失败，本次回答将不带知识库上下文。")
            return []


# ---------- 查询改写（多轮追问 → 独立检索问题） ----------

_REWRITE_SYSTEM = (
    "你是电商客服的检索查询改写器。结合对话历史，把用户最新的追问/指代消解为一句"
    "完整、独立、适合关键词与语义检索的中文问题。只输出改写后的问题本身，"
    "不要解释、不要加引号。若最新消息本身就是完整问题，原样输出。"
)


async def rewrite_query(request, settings: Settings, model) -> str:
    """把多轮对话中的追问改写成独立检索问题；失败时回退为原始消息。"""
    from langchain_core.messages import HumanMessage, SystemMessage

    lines = [f"{'用户' if item.role == 'user' else '客服'}: {item.content}"
             for item in request.history[-6:]]
    context_text = "\n".join(lines) if lines else "（无历史）"
    messages = [
        SystemMessage(_REWRITE_SYSTEM),
        HumanMessage(f"对话历史：\n{context_text}\n\n最新用户消息：{request.message}"),
    ]
    try:
        response = await model.ainvoke(messages)
        rewritten = (response.content or "").strip() if isinstance(response.content, str) else ""
        return rewritten or request.message
    except Exception:
        logger.warning("查询改写失败，回退为原始消息检索。", exc_info=True)
        return request.message


_retriever: FaqRetriever | None = None


def get_retriever(settings: Settings) -> FaqRetriever:
    """进程级单例；Settings 变化需重启进程（与 get_settings 的 lru_cache 一致）。"""
    global _retriever
    if _retriever is None:
        _retriever = FaqRetriever(settings)
    return _retriever
