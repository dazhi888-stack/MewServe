"""混合检索纯逻辑测试：BM25 / RRF / FAQ 分片（无模型、无网络、无 Chroma）。"""

from app.rag import Bm25Index, FaqRetriever, _rrf_fuse, _tokenize


class TestTokenize:
    def test_chinese_unigram_and_bigram(self):
        tokens = _tokenize("退货运费")
        assert "退" in tokens and "运费" in tokens and "退货" in tokens

    def test_ascii_words_lowercased(self):
        tokens = _tokenize("BAAI/bge-m3 500ml")
        assert "baai" in tokens and "500ml" in tokens

    def test_mixed(self):
        tokens = _tokenize("包邮门槛 59 元")
        assert "59" in tokens and "包邮" in tokens


class TestBm25:
    DOCS = [
        "满 59 元包邮，偏远地区另计。",
        "退货时运费由买家承担，质量问题除外。",
        "付款后 24 小时内发货。",
    ]

    def test_relevant_doc_scores_highest(self):
        index = Bm25Index(self.DOCS)
        scores = index.scores("退货运费谁承担")
        assert scores[1] == max(scores)
        assert scores[1] > 0

    def test_no_match_scores_zero(self):
        index = Bm25Index(self.DOCS)
        scores = index.scores("xyz")
        assert all(score == 0 for score in scores)

    def test_empty_index(self):
        index = Bm25Index([])
        assert index.scores("任何问题") == []


class TestRrfFuse:
    def test_fuses_two_rankings(self):
        fused = _rrf_fuse([["a", "b", "c"], ["b", "d"]], top_n=3)
        # b 在两路都靠前，应排第一。
        assert fused[0] == "b"
        assert set(fused) == {"b", "a", "d"}

    def test_top_n_limits(self):
        fused = _rrf_fuse([["a", "b", "c", "d"]], top_n=2)
        assert fused == ["a", "b"]


class TestFaqParsing:
    SAMPLE = """# 标题

## Q: 运费怎么算？
满 59 元包邮。

## Q: 发货时效？
付款后 24 小时内发货。
"""

    def test_parse_faq_splits_sections(self):
        chunks = FaqRetriever.parse_faq(self.SAMPLE)
        assert len(chunks) == 2
        assert chunks[0]["id"] == "faq-0001"  # 标题块不满足 ## 前缀被跳过
        assert "满 59 元包邮" in chunks[0]["document"]
        assert "24 小时内发货" in chunks[1]["document"]

    def test_parse_faq_ignores_noise(self):
        assert FaqRetriever.parse_faq("") == []
        assert FaqRetriever.parse_faq("## too short") == []
