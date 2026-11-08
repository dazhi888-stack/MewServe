"""MewHelp 对话演示页（Streamlit，二次开发新增）。

运行前提：
1. 启动后端：uvicorn app.main:app --port 8000（仓库根目录下，.env 已配置 DeepSeek）。
2. 启动演示：streamlit run streamlit_app.py

功能：聊天窗口、流式逐字输出、意图置信度标签、中档置信度确认交互、低置信度拒绝展示。
"""

import json

import httpx
import streamlit as st

st.set_page_config(page_title="MewHelp 电商客服演示", page_icon="🐱", layout="centered")

DEFAULT_BACKEND = "http://127.0.0.1:8000"


def get_state() -> None:
    st.session_state.setdefault("messages", [])          # [{"role", "content", "meta"}]
    st.session_state.setdefault("pending_confirm", None) # {"message", "confirm_event"}
    st.session_state.setdefault("backend", DEFAULT_BACKEND)


get_state()

with st.sidebar:
    st.header("⚙️ 连接设置")
    st.session_state.backend = st.text_input("后端地址", st.session_state.backend).strip() or DEFAULT_BACKEND
    st.caption("后端启动：`uvicorn app.main:app --port 8000`")
    st.divider()
    st.subheader("🧭 意图识别规则")
    st.markdown(
        "- 置信度 **≥ 0.85**：直接检索 FAQ 回答\n"
        "- **0.60 ~ 0.85**：弹出确认，确认后再答\n"
        "- **≤ 0.60**：拒绝无关问题"
    )
    if st.button("🗑️ 新对话", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_confirm = None
        st.rerun()


def render_history() -> None:
    for item in st.session_state.messages:
        with st.chat_message(item["role"], avatar="🧑" if item["role"] == "user" else "🐱"):
            st.markdown(item["content"])
            meta = item.get("meta")
            if meta:
                st.caption(meta)


def call_backend(message: str, confirm: bool = False) -> dict:
    """调用 /api/chat/stream，解析 SSE，返回聚合结果。

    返回 {"text", "intent", "confirm", "error", "finish_reason"}。
    """
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
        if not m.get("meta") or m["meta"].get("kind") != "refused"
    ]
    payload = {
        "message": message,
        "history": history[-20:],
        "confirm_intent": confirm,
    }
    result = {"text": "", "intent": None, "confirm": None, "error": None,
              "finish_reason": None, "tool_notes": [], "cache_note": None}
    try:
        with httpx.Client(timeout=120) as client:
            with client.stream(
                "POST", f"{st.session_state.backend}/api/chat/stream", json=payload,
            ) as response:
                if response.status_code != 200:
                    body = response.read().decode("utf-8", "replace")
                    try:
                        detail = json.loads(body)
                        result["error"] = detail.get("message", body)
                    except ValueError:
                        result["error"] = body or f"HTTP {response.status_code}"
                    return result
                event, data = "", ""
                for line in response.iter_lines():
                    if line.startswith("event: "):
                        event = line[7:].strip()
                    elif line.startswith("data: "):
                        data = line[6:]
                    elif not line and event:
                        _consume(result, event, data)
                        event, data = "", ""
                if event:  # 流结尾未跟空行的最后一个事件
                    _consume(result, event, data)
    except httpx.ConnectError:
        result["error"] = f"无法连接后端 {st.session_state.backend}，请先启动 uvicorn 服务。"
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"请求失败：{exc}"
    return result


def _consume(result: dict, event: str, data: str) -> None:
    try:
        payload = json.loads(data)
    except ValueError:
        return
    if event == "token":
        result["text"] += payload.get("text", "")
    elif event == "intent":
        result["intent"] = payload
    elif event == "tool":
        # 二次开发第四批：订单工具执行结果标注。
        if payload.get("tool") == "order_query":
            icon = "🔧 已查询订单" if payload.get("found") else "🔍 未查到订单"
            result["tool_notes"].append(f"{icon} {payload.get('order_id') or ''}".strip())
    elif event == "cache":
        # 二次开发第四批：语义缓存命中标注。
        score = payload.get("score")
        score_text = f"{score:.2f}" if isinstance(score, (int, float)) else "?"
        result["cache_note"] = f"⚡ 语义缓存命中（相似度 {score_text}），本次未调用模型"
    elif event == "confirm":
        result["confirm"] = payload
    elif event == "error":
        result["error"] = payload.get("message", "未知错误")
    elif event == "done":
        result["finish_reason"] = payload.get("finish_reason")


def intent_meta(intent: dict | None, finish_reason: str | None) -> str | None:
    if not intent:
        return None
    label = intent.get("category_label", intent.get("category", "?"))
    icon = {"answer": "✅", "confirm": "❓", "refuse": "🚫"}.get(intent.get("action"), "")
    text = f"{icon} 意图：{label}｜置信度 {intent['confidence']:.2f}"
    if finish_reason == "refused":
        text += "｜超出服务范围，已拒答"
    return text


def send_message(message: str, confirm: bool = False) -> None:
    """发送消息并把结果写入会话历史。"""
    if not confirm:
        st.session_state.messages.append({"role": "user", "content": message})
    with st.chat_message("assistant", avatar="🐱"):
        placeholder = st.empty()
        result = call_backend(message, confirm=confirm)

        if result["confirm"]:
            # 中档置信度：展示确认卡片，等待用户操作。
            placeholder.warning(result["confirm"]["message"])
            st.session_state.pending_confirm = {
                "message": message,
                "intent": result["intent"],
            }
            return

        if result["error"] and not result["text"]:
            placeholder.error(f"⚠️ {result['error']}")
            # 失败的回答不进入后续历史（与原基线约定一致）。
            st.session_state.pending_confirm = None
            return

        content = result["text"] or "（模型没有返回内容）"
        placeholder.markdown(content)
        meta = intent_meta(result["intent"], result["finish_reason"])
        # 二次开发第四批：工具/缓存标注拼进 meta 小字。
        extras = list(result.get("tool_notes") or [])
        if result.get("cache_note"):
            extras.append(result["cache_note"])
        if extras:
            meta = f"{meta} ｜ {' ｜ '.join(extras)}" if meta else " ｜ ".join(extras)
        if meta:
            st.caption(meta)
        entry = {"role": "assistant", "content": content, "meta": None}
        if meta:
            entry["meta"] = {"text": meta, "kind": "refused" if result["finish_reason"] == "refused" else "intent"}
        st.session_state.messages.append(entry)
        st.session_state.pending_confirm = None


render_history()

# 待确认卡片（中档置信度）
pending = st.session_state.pending_confirm
if pending:
    with st.chat_message("assistant", avatar="🐱"):
        st.info("请确认上面的问题是否与您的意图相符？")
        col1, col2, _ = st.columns([1, 1, 3])
        if col1.button("✅ 是的，回答它", key="confirm_yes"):
            send_message(pending["message"], confirm=True)
            st.rerun()
        if col2.button("❌ 不是，重新输入", key="confirm_no"):
            st.session_state.messages.append({
                "role": "assistant",
                "content": "好的，已取消本次回答。您可以换个说法重新提问。",
            })
            st.session_state.pending_confirm = None
            st.rerun()

if prompt := st.chat_input("请输入购物相关问题，例如：退货运费谁承担？"):
    send_message(prompt)
    st.rerun()
