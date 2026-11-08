"""Mock 订单工具测试：订单号提取/查询/格式化/路由（无模型、无网络）。"""

import asyncio

from app.intent import IntentResult
from app.tools import (
    ASK_ORDER_ID_GUIDANCE, MISSING_ORDER_TEMPLATE, extract_order_ids,
    format_order, normalize_order_id, query_order, resolve_order_tool,
)


class TestNormalizeOrderId:
    def test_keps_valid_id(self):
        assert normalize_order_id("MH20261101002") == "MH20261101002"

    def test_strips_separators_and_uppercases(self):
        assert normalize_order_id(" mh 2026 1101 002 ") == "MH20261101002"
        assert normalize_order_id("MH20261101002。") == "MH20261101002"

    def test_rejects_wrong_format(self):
        assert normalize_order_id("12345") is None
        assert normalize_order_id("MH123") is None  # 位数不足
        assert normalize_order_id("XX20261101002") is None
        assert normalize_order_id("") is None


class TestExtractOrderIds:
    def test_extracts_from_message(self):
        message = "帮我查一下订单 MH20261101002 到哪了，还有 MH20261028001"
        assert extract_order_ids(message) == ["MH20261101002", "MH20261028001"]

    def test_deduplicates_case_insensitively(self):
        assert extract_order_ids("mh20261101002 和 MH20261101002") == ["MH20261101002"]

    def test_no_order_id(self):
        assert extract_order_ids("我的快递怎么还没到") == []

    def test_adjacent_to_chinese(self):
        # 回归：订单号紧贴中文时 \b 词边界不成立（中文也算 \w），曾漏匹配。
        assert extract_order_ids("我的订单MH20261101002到哪了？") == ["MH20261101002"]
        assert extract_order_ids("订单号：MH20261101002。") == ["MH20261101002"]

    def test_rejects_embedded_in_longer_token(self):
        # 前面贴字母（RMH...）或后面跟更长数字串时不得误截取。
        assert extract_order_ids("编号RMH20261101002") == []
        assert extract_order_ids("MH2026110100212345") == []


class TestQueryOrder:
    def test_known_order(self):
        order = query_order("MH20261101002")
        assert order is not None
        assert order["status"] == "运输中"
        assert order["logistics"]["carrier"] == "中通快递"

    def test_sensitive_fields_masked(self):
        known_ids = [
            "MH20261028001", "MH20261101002", "MH20261105003",
            "MH20261020004", "MH20261106005",
        ]
        for order in map(query_order, known_ids):
            assert order is not None
            assert "*" in order["recipient"]
            assert "*" in order["phone"]

    def test_unknown_order(self):
        assert query_order("MH00000000001") is None
        assert query_order("乱写的") is None


class TestFormatOrder:
    def test_contains_key_fields(self):
        order = query_order("MH20261101002")
        text = format_order(order)
        assert "MH20261101002" in text
        assert "运输中" in text
        assert "中通快递" in text
        assert "79" in text or "1299" in text

    def test_refund_order_contains_refund_info(self):
        text = format_order(query_order("MH20261020004"))
        assert "已退款" in text
        assert "原支付路径退回" in text

    def test_pending_order_has_no_logistics_line(self):
        text = format_order(query_order("MH20261105003"))
        assert "待发货" in text
        assert "暂无物流信息" in text


class TestResolveOrderTool:
    def run(self, message, intent=None):
        from app.config import get_settings

        return asyncio.run(
            resolve_order_tool(message, intent, get_settings(), model=None)
        )

    def test_regex_hit_returns_order(self):
        result = self.run("帮我查下订单 MH20261101002 到哪了")
        assert result.found is True
        assert result.order_id == "MH20261101002"
        assert "运输中" in result.context

    def test_regex_hit_unknown_id(self):
        result = self.run("查一下 MH99999999999")
        assert result.found is False
        assert result.order_id == "MH99999999999"
        assert MISSING_ORDER_TEMPLATE.format(order_id="MH99999999999") in result.context

    def test_order_intent_without_id_gives_guidance(self):
        intent = IntentResult(category="order_issue", confidence=0.9)
        result = self.run("我的订单怎么还没到", intent)
        assert result.found is False
        assert result.order_id is None
        assert result.context == ASK_ORDER_ID_GUIDANCE

    def test_unrelated_message_returns_empty(self):
        result = self.run("退货运费谁承担")
        assert result.context == ""
        assert result.found is False
