"""意图识别路由规则与 RAG FAQ 解析的纯逻辑测试（无模型、无网络）。"""

import pytest

from app.config import Settings
from app.intent import IntentResult, route_action
from app.rag import FaqRetriever


def _settings(**values):
    return Settings(
        _env_file=None,
        deepseek_base_url="https://upstream.invalid/v1",
        deepseek_model="deepseek-chat",
        deepseek_api_key="test-only-key",
        **values,
    )


class TestRouteAction:
    def test_high_confidence_answers(self):
        intent = IntentResult(category="product_spec", confidence=0.85)
        assert route_action(intent, _settings()) == "answer"
        intent = IntentResult(category="return_exchange", confidence=0.99)
        assert route_action(intent, _settings()) == "answer"

    def test_irrelevant_always_refused(self):
        # 二次开发修正：模型高度确信无关的问题直接拒答，不再进入回答分支。
        intent = IntentResult(category="irrelevant", confidence=0.98)
        assert route_action(intent, _settings()) == "refuse"
        intent = IntentResult(category="irrelevant", confidence=0.70)
        assert route_action(intent, _settings()) == "refuse"

    def test_mid_confidence_requires_confirm(self):
        intent = IntentResult(category="shipping_fee", confidence=0.61)
        assert route_action(intent, _settings()) == "confirm"
        intent = IntentResult(category="other", confidence=0.84)
        assert route_action(intent, _settings()) == "confirm"

    def test_low_confidence_refuses(self):
        # 0.60 属于拒绝档（confirm 档为开区间 (0.60, 0.85)）。
        intent = IntentResult(category="irrelevant", confidence=0.60)
        assert route_action(intent, _settings()) == "refuse"
        intent = IntentResult(category="irrelevant", confidence=0.01)
        assert route_action(intent, _settings()) == "refuse"

    def test_custom_thresholds(self):
        intent = IntentResult(category="greeting", confidence=0.70)
        assert route_action(intent, _settings(
            intent_high_threshold=0.65, intent_low_threshold=0.5,
        )) == "answer"
        assert route_action(intent, _settings(
            intent_high_threshold=0.9, intent_low_threshold=0.75,
        )) == "refuse"

    def test_invalid_threshold_order_rejected(self):
        with pytest.raises(Exception):
            _settings(intent_high_threshold=0.5, intent_low_threshold=0.9)


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
