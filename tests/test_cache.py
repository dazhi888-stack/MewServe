"""语义缓存测试：注入确定性伪 embedding，不依赖真实模型与网络。"""

import pytest

from chromadb.api.types import EmbeddingFunction

from app.cache import SemanticCache, _normalize_query
from app.config import Settings


class HashEmbedding(EmbeddingFunction):
    """确定性伪 embedding：按字符与 bigram hash 到 64 维稀疏向量。

    相同文本相似度恒为 1；无公共字符的文本相似度为 0，可预测、可复现。
    继承 chromadb EmbeddingFunction 以满足查询路径的 embed_query 协议。
    """

    DIM = 64

    def __call__(self, input):  # noqa: A002 - 与 chromadb 协议签名一致
        vectors = []
        for text in input:
            vec = [0.0] * self.DIM
            tokens = list(text) + [text[i:i + 2] for i in range(len(text) - 1)]
            for token in tokens:
                vec[hash(token) % self.DIM] += 1.0
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            vectors.append([v / norm for v in vec])
        return vectors

    def name(self) -> str:
        return "hash_test_embedding"

    def get_config(self) -> dict:
        return {}

    def build_from_config(self, config: dict) -> "HashEmbedding":
        return self


def make_cache(tmp_path, **overrides) -> SemanticCache:
    settings = Settings(
        semantic_cache_enabled=True,
        chroma_dir=str(tmp_path / "cache_db"),
        **overrides,
    )
    return SemanticCache(settings, embedding_fn=HashEmbedding())


class TestNormalizeQuery:
    def test_removes_whitespace_and_lowercases(self):
        assert _normalize_query("  退货运费  谁承担  ") == _normalize_query("退货运费谁承担")
        assert _normalize_query("Is It FREE?") == "isitfree?"


class TestLookupAndStore:
    def test_store_then_exact_hit(self, tmp_path):
        cache = make_cache(tmp_path)
        assert cache.store("退货运费谁承担", "满 59 元包邮，退货运费买家承担。") is True
        hit = cache.lookup("退货运费谁承担")
        assert hit is not None
        assert hit["answer"].startswith("满 59 元包邮")
        assert hit["score"] >= 0.99

    def test_hit_after_whitespace_normalization(self, tmp_path):
        cache = make_cache(tmp_path)
        cache.store("退货运费谁承担", "答案 A")
        assert cache.lookup("  退货运费   谁承担  ")["answer"] == "答案 A"

    def test_miss_for_disjoint_question(self, tmp_path):
        cache = make_cache(tmp_path)
        cache.store("退货运费谁承担", "答案 A")
        assert cache.lookup("今天天气怎么样") is None

    def test_lookup_before_any_store(self, tmp_path):
        cache = make_cache(tmp_path)
        assert cache.lookup("任何问题") is None

    def test_empty_answer_not_stored(self, tmp_path):
        cache = make_cache(tmp_path)
        assert cache.store("问题", "   ") is False
        assert cache.lookup("问题") is None

    def test_repeated_store_overwrites(self, tmp_path):
        cache = make_cache(tmp_path)
        cache.store("退货运费谁承担", "旧答案")
        cache.store("退货运费谁承担", "新答案")
        assert cache.lookup("退货运费谁承担")["answer"] == "新答案"

    def test_threshold_configurable(self, tmp_path):
        cache = make_cache(tmp_path, cache_similarity_threshold=0.99)
        cache.store("退货运费谁承担", "答案 A")
        # 相似度 1.0 的完全相同问题仍命中；阈值设满时其他问题不可能命中。
        assert cache.lookup("退货运费谁承担") is not None


class TestOrderEntityGuard:
    """订单类问题不入缓存 + 实体比对二道防线（防串单/防泄漏）。"""

    def test_order_question_not_stored(self, tmp_path):
        cache = make_cache(tmp_path)
        # 含订单号的问题不入缓存：答案绑定具体订单且含个人信息。
        assert cache.store("我的订单MH20261101002到哪了", "您的订单运输中") is False
        assert cache.lookup("我的订单MH20261101002到哪了") is None

    def test_cross_order_lookup_misses(self, tmp_path):
        cache = make_cache(tmp_path)
        # 即使有相似的非订单条目，含订单号的查询也必须走实体比对。
        cache.store("我的订单到哪了", "请提供订单号")
        # hash embedding 下两句高度相似，若无实体防线会串单命中。
        assert cache.lookup("我的订单MH20261105003到哪了") is None

    def test_mismatched_order_id_misses(self, tmp_path):
        # 防御历史脏数据：手工构造订单号标注不一致的条目。
        cache = make_cache(tmp_path)
        cache.ensure()  # 惰性初始化集合
        collection = cache._collection  # noqa: SLF001
        collection.upsert(
            ids=["manual-key"],
            documents=[_normalize_query("我的订单MH20261101002到哪了")],
            metadatas=[{"answer": "别的订单的答案", "created_at": 1.0,
                        "order_ids": "MH20261101002"}],
        )
        assert cache.lookup("我的订单MH20261105003到哪了") is None

    def test_matching_order_id_hits(self, tmp_path):
        # 订单号一致时实体比对放行（同一订单重复提问是合法复用场景）。
        cache = make_cache(tmp_path)
        cache.ensure()  # 惰性初始化集合
        collection = cache._collection  # noqa: SLF001
        collection.upsert(
            ids=["manual-key"],
            documents=[_normalize_query("我的订单MH20261101002到哪了")],
            metadatas=[{"answer": "您的订单运输中", "created_at": 1.0,
                        "order_ids": "MH20261101002"}],
        )
        assert cache.lookup("我的订单MH20261101002到哪了")["answer"] == "您的订单运输中"

    def test_legacy_entry_without_order_ids_never_hits(self, tmp_path):
        cache = make_cache(tmp_path)
        cache.ensure()  # 惰性初始化集合
        collection = cache._collection  # noqa: SLF001
        # 旧版本条目没有 order_ids 标注（历史脏数据），一律不命中。
        collection.upsert(
            ids=["legacy-key"],
            documents=[_normalize_query("我的订单到哪了")],
            metadatas=[{"answer": "旧版缓存的订单答案", "created_at": 1.0}],
        )
        assert cache.lookup("我的订单到哪了") is None


class TestEvictAndClear:
    def test_fifo_eviction(self, tmp_path):
        cache = make_cache(tmp_path, cache_max_entries=2)
        cache.store("问题一", "答案一")
        cache.store("问题二", "答案二")
        cache.store("问题三", "答案三")
        assert cache.lookup("问题三") is not None
        assert cache.lookup("问题一") is None  # 最先写入的被淘汰

    def test_clear(self, tmp_path):
        cache = make_cache(tmp_path)
        cache.store("问题一", "答案一")
        assert cache.clear() == 1
        assert cache.lookup("问题一") is None
        assert cache.clear() == 0
