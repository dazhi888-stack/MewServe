"""Mock 订单查询工具（二次开发新增）。

模拟电商客服中的"订单查询"工具，供对话管线在订单/物流类问题上调用：

- 内置一份 Mock 订单数据（演示用，重启即恢复默认；收件人姓名/手机号已脱敏）。
- 触发分两级，尽量避免不必要的模型调用：
  1) 消息中出现订单号（MH + 9~12 位数字，正则命中）→ 直接查询，零模型开销；
  2) 意图为订单/物流相关但消息中无订单号 → 一次 json_mode 结构化抽取
     尝试从消息中提取订单号。
- 查询结果格式化为"订单上下文"注入 system prompt（与 FAQ 上下文同通道），
  不引入 function-calling 多轮循环，保持基线流式转发契约（工具调用即报错）
  与"先校验后开流"的 413 契约不变。
- 找不到订单号时返回引导话术，让模型向用户索要订单号。
"""

import json
import re
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, ValidationError

from langchain_deepseek import ChatDeepSeek

from app.config import Settings
from app.errors import AppError
from app.model import model_response_scope

# 订单号规则：MH 前缀 + 9~12 位数字（与 Mock 数据保持一致）。
# 注意不能用 \b 词边界：中文与 M 之间不产生 \b（中文也算 \w），
# "订单MH20261101002" 这种紧贴写法会漏匹配。改用显式 lookaround：
# 前面不能是字母/数字（防 RMH... 误命中），后面不能紧跟数字（防长数字串截断）。
ORDER_ID_RE = re.compile(r"(?<![A-Za-z0-9])MH\d{9,12}(?!\d)", re.IGNORECASE)

# 需要查询订单的意图类别。
ORDER_INTENT_CATEGORIES = {"order_issue", "delivery_time"}

# 消息中无订单号时的引导话术（注入 system prompt）。
ASK_ORDER_ID_GUIDANCE = (
    "【订单查询】用户在咨询订单/物流问题，但消息中没有可识别的订单号。"
    "请先友好地请用户提供订单号（格式如 MH20261101002），"
    "并说明可在「我的-全部订单」页面找到。"
)


def _masked(name: str) -> str:
    return name[0] + "*" * max(len(name) - 1, 1)


# ---------- Mock 订单数据（演示用） ----------

_MOCK_ORDERS: dict[str, dict] = {
    "MH20261028001": {
        "order_id": "MH20261028001",
        "status": "已签收",
        "recipient": _masked("王晓芳"),
        "phone": "138****6721",
        "items": [{"name": "萌爪保温杯 500ml 樱花粉", "qty": 2}],
        "amount": "¥158.00",
        "created_at": "2026-10-28 14:32",
        "logistics": {
            "carrier": "顺丰速运",
            "tracking_no": "SF3162884401",
            "expected_delivery": "2026-10-30（已送达）",
        },
        "refund": None,
    },
    "MH20261101002": {
        "order_id": "MH20261101002",
        "status": "运输中",
        "recipient": _masked("李强"),
        "phone": "159****3345",
        "items": [{"name": "轻便婴儿推车 Pro 曜石黑", "qty": 1}],
        "amount": "¥1299.00",
        "created_at": "2026-11-01 09:15",
        "logistics": {
            "carrier": "中通快递",
            "tracking_no": "78532109984412",
            "expected_delivery": "预计 2026-11-09 送达",
        },
        "refund": None,
    },
    "MH20261105003": {
        "order_id": "MH20261105003",
        "status": "待发货",
        "recipient": _masked("张敏"),
        "phone": "186****0290",
        "items": [
            {"name": "全棉婴儿纱布浴巾 6 层", "qty": 1},
            {"name": "竹纤维柔抚巾 3 条装", "qty": 2},
        ],
        "amount": "¥96.50",
        "created_at": "2026-11-05 21:47",
        "logistics": None,
        "refund": None,
    },
    "MH20261020004": {
        "order_id": "MH20261020004",
        "status": "已退款",
        "recipient": _masked("陈浩"),
        "phone": "137****8811",
        "items": [{"name": "智能宠物喂食器 Mini", "qty": 1}],
        "amount": "¥329.00",
        "created_at": "2026-10-20 11:03",
        "logistics": None,
        "refund": {
            "amount": "¥329.00",
            "reason": "商品与预期不符",
            "refunded_at": "2026-10-22 16:20",
            "channel": "原支付路径退回（1~3 个工作日内到账）",
        },
    },
    "MH20261106005": {
        "order_id": "MH20261106005",
        "status": "已签收",
        "recipient": _masked("赵磊"),
        "phone": "150****5577",
        "items": [{"name": "儿童防摔水杯 吸管款 350ml", "qty": 3}],
        "amount": "¥117.00",
        "created_at": "2026-11-06 08:26",
        "logistics": {
            "carrier": "圆通速递",
            "tracking_no": "YT9251773302",
            "expected_delivery": "2026-11-08（已送达）",
        },
        "refund": None,
    },
}


def normalize_order_id(raw: str) -> str | None:
    """清洗用户输入的订单号；不符合格式返回 None。"""
    candidate = re.sub(r"[\s:：,，。]+", "", raw).upper()
    return candidate if re.fullmatch(r"MH\d{9,12}", candidate) else None


def extract_order_ids(text: str) -> list[str]:
    """从消息中提取全部订单号（去重、保序、统一大写）。"""
    seen: list[str] = []
    for match in ORDER_ID_RE.findall(text):
        normalized = normalize_order_id(match)
        if normalized and normalized not in seen:
            seen.append(normalized)
    return seen


def query_order(order_id: str) -> dict | None:
    """按订单号查询 Mock 订单；未收录返回 None。"""
    normalized = normalize_order_id(order_id)
    return _MOCK_ORDERS.get(normalized) if normalized else None


def format_order(order: dict) -> str:
    """把订单数据格式化为注入 system prompt 的上下文文本。"""
    items_text = "、".join(
        f"{item['name']}×{item['qty']}" for item in order["items"]
    )
    lines = [
        "【订单查询】已从订单系统查到该订单，信息如下（请在回答中自然使用，不要罗列原始 JSON）：",
        f"- 订单号：{order['order_id']}",
        f"- 当前状态：{order['status']}",
        f"- 商品：{items_text}",
        f"- 实付金额：{order['amount']}",
        f"- 下单时间：{order['created_at']}",
        f"- 收件人：{order['recipient']}（{order['phone']}，已脱敏）",
    ]
    if order.get("logistics"):
        logistics = order["logistics"]
        lines.append(
            f"- 物流：{logistics['carrier']} {logistics['tracking_no']}，{logistics['expected_delivery']}"
        )
    else:
        lines.append("- 物流：暂无物流信息（如为待发货状态，可参考 FAQ 中的发货时效）")
    if order.get("refund"):
        refund = order["refund"]
        lines.append(
            f"- 退款：{refund['amount']}，原因“{refund['reason']}”，"
            f"{refund['refunded_at']} 处理完成，{refund['channel']}"
        )
    return "\n".join(lines)


MISSING_ORDER_TEMPLATE = (
    "【订单查询】用户提供了订单号 {order_id}，但订单系统中未找到该订单。"
    "请礼貌告知未查询到，请用户核对订单号是否有误（可在「我的-全部订单」查看）。"
)


@dataclass(frozen=True)
class OrderToolResult:
    """订单工具执行结果；context 为空串表示本次无需注入订单上下文。"""

    context: str
    order_id: str | None
    found: bool


def _lookup(ids: list[str]) -> OrderToolResult:
    order = query_order(ids[0])
    if order:
        return OrderToolResult(format_order(order), order["order_id"], True)
    return OrderToolResult(MISSING_ORDER_TEMPLATE.format(order_id=ids[0]), ids[0], False)


# ---------- 模型抽取订单号（消息无显式订单号时的兜底） ----------

_EXTRACT_SYSTEM = (
    "你是订单号抽取器。从用户消息中提取订单号，只输出一个 JSON 对象："
    '{"order_id": "MH开头的订单号"}；消息中没有明确订单号时输出 {"order_id": null}。'
    "不要猜测、不要拼接不完整的订单号。"
)


class OrderIdExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    order_id: str | None


async def extract_order_id(
    message: str, settings: Settings, model: ChatDeepSeek,
) -> str | None:
    """一次 json_mode 结构化抽取；任何失败都返回 None（不影响主回答链路）。"""
    try:
        structured = model.with_structured_output(
            OrderIdExtraction, method="json_mode", include_raw=True,
        )
        async with model_response_scope(model):
            envelope = await structured.ainvoke(
                [SystemMessage(_EXTRACT_SYSTEM), HumanMessage(message)]
            )
        raw = envelope["raw"]
        if (raw.tool_calls or raw.invalid_tool_calls
                or raw.additional_kwargs.get("function_call")
                or raw.response_metadata.get("finish_reason") != "stop"
                or not isinstance(raw.content, str)):
            return None
        decoded = json.loads(raw.content)
        parsed = OrderIdExtraction.model_validate(decoded)
        if envelope["parsing_error"] is not None or envelope["parsed"] is None:
            return None
        return normalize_order_id(parsed.order_id) if parsed.order_id else None
    except (AppError, json.JSONDecodeError, ValidationError):
        return None
    except Exception:
        return None


async def resolve_order_tool(
    message: str,
    intent,  # IntentResult | None（避免循环导入，运行时按鸭子类型判断）
    settings: Settings,
    model: ChatDeepSeek,
) -> OrderToolResult:
    """订单工具入口：正则直查优先，模型抽取兜底，无订单号时给引导话术。"""
    ids = extract_order_ids(message)
    if ids:
        return _lookup(ids)

    order_related = intent is not None and intent.category in ORDER_INTENT_CATEGORIES
    if order_related and settings.model_configured:
        extracted = await extract_order_id(message, settings, model)
        if extracted:
            return _lookup([extracted])
    if order_related:
        return OrderToolResult(ASK_ORDER_ID_GUIDANCE, None, False)
    return OrderToolResult("", None, False)
