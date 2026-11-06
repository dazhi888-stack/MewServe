"""Server-owned templates; user content is always a separate HumanMessage."""

from pathlib import Path

from langchain_core.prompts import PromptTemplate


_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "prompts"
CHAT_PROMPT = PromptTemplate.from_template(
    (_TEMPLATE_DIR / "customer_service.txt").read_text(encoding="utf-8")
)
EXTRACTION_PROMPT = PromptTemplate.from_template(
    (_TEMPLATE_DIR / "extract_aftersales.txt").read_text(encoding="utf-8")
)
# 二次开发新增：意图识别 Prompt。
INTENT_PROMPT = PromptTemplate.from_template(
    (_TEMPLATE_DIR / "intent.txt").read_text(encoding="utf-8")
)

# Prompt-only shared contract. Task 4 owns the validated extraction result type.
EXTRACTION_SCHEMA_DESCRIPTION = """仅有以下三个必填 JSON key，禁止额外字段：
order_id: string 或 null。保留前导零；未提供、含糊或多订单无法唯一确定时为 null，不得猜测。
request_type: refund（仅退款）、return_refund（退货退款）、exchange（换货）、repair（维修）、logistics（物流）、other（其他明确诉求）、unknown（无法判断）之一。
expected_solution: string 或 null。仅提取用户明确表达的期望；未表达时为 null。
存在多个诉求时，以用户最终明确选择的主要售后诉求为准；无法确定时 request_type 为 unknown。"""

# 二次开发新增：意图识别字段规范。
INTENT_SCHEMA_DESCRIPTION = """仅有以下两个必填 JSON key，禁止额外字段：
category: product_spec（商品规格咨询）、shipping_fee（运费咨询）、return_exchange（退换货）、refund（退款）、delivery_time（发货/物流时效）、order_issue（订单/售后问题）、greeting（问候寒暄）、other（其他电商相关咨询）、irrelevant（与电商客服无关）之一。
confidence: 0 到 1 之间的小数，表示消息属于该类别的置信度，两位小数。"""
