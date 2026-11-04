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

# Prompt-only shared contract. Task 4 owns the validated extraction result type.
EXTRACTION_SCHEMA_DESCRIPTION = """仅有以下三个必填 JSON key，禁止额外字段：
order_id: string 或 null。保留前导零；未提供、含糊或多订单无法唯一确定时为 null，不得猜测。
request_type: refund（仅退款）、return_refund（退货退款）、exchange（换货）、repair（维修）、logistics（物流）、other（其他明确诉求）、unknown（无法判断）之一。
expected_solution: string 或 null。仅提取用户明确表达的期望；未表达时为 null。
存在多个诉求时，以用户最终明确选择的主要售后诉求为准；无法确定时 request_type 为 unknown。"""
