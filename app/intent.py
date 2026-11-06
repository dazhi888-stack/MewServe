"""意图识别模块（二次开发新增）。

复用 app/service.py 的结构化输出链路（json_mode + Pydantic 校验 + 上游响应校验），
对用户消息做一次独立分类，产出类别与置信度，由路由规则决定后续分支：

- confidence >= intent_high_threshold（默认 0.85）→ 正常 RAG 问答
- intent_low ~ high 之间（默认 0.60~0.85）  → 弹出确认，用户确认后再回答
- confidence <= intent_low_threshold（默认 0.60） → 拒绝无关问题
"""

import json
from typing import Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, ValidationError

from langchain_deepseek import ChatDeepSeek

from app.config import Settings
from app.errors import AppError, public_upstream_error, structured_error
from app.model import model_response_scope
from app.prompts import INTENT_PROMPT, INTENT_SCHEMA_DESCRIPTION

IntentCategory = Literal[
    "product_spec",      # 商品规格咨询
    "shipping_fee",      # 运费咨询
    "return_exchange",   # 退换货
    "refund",            # 退款
    "delivery_time",     # 发货/物流时效
    "order_issue",       # 订单/售后问题
    "greeting",          # 问候寒暄
    "other",             # 其他电商相关咨询
    "irrelevant",        # 与电商客服无关
]

CATEGORY_LABELS = {
    "product_spec": "商品规格咨询",
    "shipping_fee": "运费咨询",
    "return_exchange": "退换货",
    "refund": "退款",
    "delivery_time": "发货/物流时效",
    "order_issue": "订单/售后问题",
    "greeting": "问候寒暄",
    "other": "其他电商相关咨询",
    "irrelevant": "与电商客服无关",
}


class IntentResult(BaseModel):
    """意图分类结果；confidence 由模型给出，仅在 0~1 内有效。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    category: IntentCategory
    confidence: float


def route_action(
    intent: IntentResult, settings: Settings,
) -> Literal["answer", "confirm", "refuse"]:
    """三档路由规则；阈值可在 .env 调整（INTENT_HIGH_THRESHOLD / INTENT_LOW_THRESHOLD）。

    - 分类为 irrelevant（模型高度确信与电商无关）→ 直接拒绝；
    - 其余电商类意图按置信度三档路由：>=high 正常回答、(low, high) 弹确认、<=low 拒绝。
    """
    if intent.category == "irrelevant" or intent.confidence <= settings.intent_low_threshold:
        return "refuse"
    if intent.confidence >= settings.intent_high_threshold:
        return "answer"
    return "confirm"


def build_intent_messages(message: str) -> list[BaseMessage]:
    return [
        SystemMessage(INTENT_PROMPT.format(schema=INTENT_SCHEMA_DESCRIPTION)),
        HumanMessage(message),
    ]


async def classify_intent(
    message: str, settings: Settings, model: ChatDeepSeek,
) -> IntentResult:
    """独立的一次意图提取，不改变聊天历史，不执行业务动作。"""
    messages = build_intent_messages(message)
    # 二次开发：与售后提取一致，结构化输出收敛为 json_mode（DeepSeek 兼容）。
    try:
        structured = model.with_structured_output(
            IntentResult, method="json_mode", include_raw=True,
        )
        async with model_response_scope(model):
            envelope = await structured.ainvoke(messages)
        raw = envelope["raw"]
        if (raw.tool_calls or raw.invalid_tool_calls
                or raw.additional_kwargs.get("function_call")
                or raw.response_metadata.get("finish_reason") != "stop"):
            raise structured_error("structured_unsupported_response")
        if not isinstance(raw.content, str):
            raise structured_error("structured_unsupported_response")
        decoded = json.loads(raw.content)
        intent = IntentResult.model_validate(decoded)
        if envelope["parsing_error"] is not None or envelope["parsed"] is None:
            raise structured_error("structured_invalid_schema")
        return intent
    except AppError:
        raise
    except json.JSONDecodeError:
        raise structured_error("structured_invalid_json") from None
    except ValidationError:
        raise structured_error("structured_invalid_schema") from None
    except Exception as exc:
        raise public_upstream_error(exc) from None
