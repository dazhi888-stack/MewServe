"""RAG 模块：Chroma 持久化向量库 + 电商 FAQ 知识库检索（二次开发新增）。

设计要点：
- 向量库持久化在 settings.chroma_dir，FAQ 源文件为 Markdown，按 `## ` 分片。
- Embedding 优先走 OpenAI 兼容接口（默认硅基流动 BAAI/bge-m3，中文效果好）；
  未配置 key 时回退到 Chroma 自带的本地默认 embedding（ONNX MiniLM）。
- 服务启动时 ensure_collection()：集合不存在或为空则自动导入 FAQ。
"""

import logging
import re
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from app.config import Settings

logger = logging.getLogger("mewhelp.rag")

COLLECTION_NAME = "ecommerce_faq"


class FaqRetriever:
    """进程内共享一个 PersistentClient；retrieve 返回命中的 FAQ 原文列表。"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = chromadb.PersistentClient(path=settings.chroma_dir)
        self._collection = None

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
        """幂等初始化：集合为空时自动导入 FAQ。失败只记录，不阻断服务启动。"""
        try:
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=self._build_embedding(),
                metadata={"hnsw:space": "cosine"},
            )
            if self._collection.count() == 0:
                self.ingest()
        except Exception:
            self._collection = None
            logger.exception("Chroma 知识库初始化失败，RAG 检索将不可用。")

    @property
    def available(self) -> bool:
        return self._collection is not None

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
        logger.info("已导入 %d 条 FAQ 到 Chroma。", len(chunks))
        return len(chunks)

    # ---------- retrieve ----------

    def retrieve(self, query: str) -> list[str]:
        """检索与问题最相关的 top_k 条 FAQ 原文；不可用或异常时返回空列表。"""
        if self._collection is None:
            return []
        try:
            result = self._collection.query(
                query_texts=[query], n_results=self._settings.rag_top_k,
            )
            documents = result.get("documents") or []
            return list(documents[0]) if documents else []
        except Exception:
            logger.exception("RAG 检索失败，本次回答将不带知识库上下文。")
            return []


_retriever: FaqRetriever | None = None


def get_retriever(settings: Settings) -> FaqRetriever:
    """进程级单例；Settings 变化需重启进程（与 get_settings 的 lru_cache 一致）。"""
    global _retriever
    if _retriever is None:
        _retriever = FaqRetriever(settings)
    return _retriever
