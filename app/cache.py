"""语义缓存模块（二次开发新增）。

用 Chroma 向量库缓存"问题 → 回答"：新问题先做向量相似度检索，与缓存中的
历史问题相似度达到阈值（默认 0.92）时直接复用历史回答，跳过 RAG 检索与
模型调用，显著降低重复问题的首字延迟与 token 成本。

设计要点：
- 与 FAQ 知识库共用 chroma_dir（独立的 `semantic_cache` 集合），共用同一
  embedding 函数（build_embedding_function），向量空间一致。
- 缓存键 = 归一化问题文本的 SHA-1，重复问题写入即覆盖（天然去重）。
- 命中返回缓存回答与相似度；未命中由调用方在流式回答完成后写入。
- 只缓存首轮提问（无 history）的回答——多轮回答依赖上下文，语义相近的
  追问复用首轮答案反而是错的。
- embedding 函数可注入（测试用确定性伪 embedding，避免真实模型/网络）。
- 任何异常都不影响主链路：初始化失败 → 缓存不可用；写入失败 → 仅记录日志。
"""

import hashlib
import logging
import time

import chromadb

from app.config import Settings
from app.rag import build_embedding_function

logger = logging.getLogger("mewhelp.cache")

COLLECTION_NAME = "semantic_cache"


def _normalize_query(query: str) -> str:
    """缓存键归一化：去掉全部空白、去首尾、小写（英文），保留中文原样。

    缓存键只要求确定性：同一问题任何写法必须得到同一键，因此这里直接
    去掉所有空白字符（含中文之间被输入法/复制带来的空格）。
    """
    return "".join(query.split()).strip().lower()


class SemanticCache:
    """进程级单例（见 get_semantic_cache）；lookup/store 均为同步阻塞调用，
    事件循环中应通过 asyncio.to_thread 调用。"""

    def __init__(self, settings: Settings, embedding_fn=None) -> None:
        self._settings = settings
        self._embedding_fn = embedding_fn
        self._collection = None

    # ---------- collection ----------

    def ensure(self) -> None:
        """幂等初始化集合；失败时置空并降级为"缓存不可用"。"""
        if self._collection is not None:
            return
        try:
            client = chromadb.PersistentClient(path=self._settings.chroma_dir)
            embedding = self._embedding_fn or build_embedding_function(self._settings)
            self._collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=embedding,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception:
            self._collection = None
            logger.exception("语义缓存初始化失败，本次运行将直接透传模型。")

    @property
    def available(self) -> bool:
        self.ensure()
        return self._collection is not None

    # ---------- lookup / store ----------

    def lookup(self, query: str) -> dict | None:
        """按语义相似度查缓存；命中返回 {"answer", "score"}，否则 None。"""
        if not self.available:
            return None
        try:
            normalized = _normalize_query(query)
            if not normalized:
                return None
            count = self._collection.count()
            if count == 0:
                return None
            result = self._collection.query(query_texts=[normalized], n_results=1)
            distances = (result.get("distances") or [[]])[0]
            documents = (result.get("documents") or [[]])[0]
            metadatas = (result.get("metadatas") or [[]])[0]
            if not distances:
                return None
            # cosine 空间下 distance = 1 - cosine_similarity。
            score = 1.0 - float(distances[0])
            if score < self._settings.cache_similarity_threshold:
                return None
            answer = metadatas[0].get("answer") if metadatas else None
            # 旧格式把回答存为 document；为空时回退读 document 字段。
            answer = answer or documents[0]
            if not answer:
                return None
            return {"answer": answer, "score": round(score, 4)}
        except Exception:
            logger.exception("语义缓存查询失败，本次按未命中处理。")
            return None

    def store(self, query: str, answer: str) -> bool:
        """写入/覆盖缓存（同问题重复写入即更新）；失败仅记录日志返回 False。"""
        if not answer.strip():
            return False
        if not self.available:
            return False
        try:
            normalized = _normalize_query(query)
            key = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
            self._collection.upsert(
                ids=[key],
                documents=[normalized],
                # 浮点时间戳保证同秒内多次写入也可按写入顺序 FIFO 淘汰。
                metadatas=[{"answer": answer, "created_at": time.time()}],
            )
            self._evict()
            return True
        except Exception:
            logger.exception("语义缓存写入失败（不影响本次回答）。")
            return False

    def clear(self) -> int:
        """清空缓存，返回清除的条数；集合未就绪时返回 -1。"""
        if not self.available:
            return -1
        count = self._collection.count()
        if count:
            self._collection.delete(ids=self._collection.get()["ids"])
        return count

    def _evict(self) -> None:
        """超过上限时按 created_at 淘汰最旧条目（FIFO），防止无限增长。"""
        max_entries = self._settings.cache_max_entries
        count = self._collection.count()
        if count <= max_entries:
            return
        data = self._collection.get(include=["metadatas"])
        items = sorted(
            zip(data["ids"], data.get("metadatas") or []),
            key=lambda pair: (pair[1] or {}).get("created_at", 0),
        )
        stale = [item_id for item_id, _ in items[: count - max_entries]]
        if stale:
            self._collection.delete(ids=stale)


_cache: SemanticCache | None = None


def get_semantic_cache(settings: Settings) -> SemanticCache:
    """进程级单例；Settings 变化需重启进程（与 get_retriever 一致）。"""
    global _cache
    if _cache is None:
        _cache = SemanticCache(settings)
    return _cache
