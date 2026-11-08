"use strict";

const form = document.querySelector("#chat-form");
const input = document.querySelector("#message");
const sendButton = document.querySelector("#send");
const sendLabel = document.querySelector("#send-label");
const newButton = document.querySelector("#new-chat");
const messages = document.querySelector("#messages");
const welcome = document.querySelector("#welcome");
const status = document.querySelector("#status");
let history = [];
let busy = false;

class ChatError extends Error {}

function setStatus(text, error = false) {
  status.textContent = text;
  status.classList.toggle("error", error);
}

function updateControls() {
  input.disabled = busy;
  sendButton.disabled = busy || !input.value.trim();
  newButton.disabled = busy;
  sendLabel.textContent = busy ? "回复中" : "发送";
  form.setAttribute("aria-busy", String(busy));
}

function scrollToBottom() {
  messages.scrollTop = messages.scrollHeight;
}

function addMessage(role, text) {
  const row = document.createElement("article");
  row.className = `message ${role}`;
  if (role === "assistant") {
    const avatar = document.createElement("span");
    avatar.className = "message-avatar";
    avatar.textContent = "m.";
    avatar.setAttribute("aria-hidden", "true");
    row.append(avatar);
  }
  const content = document.createElement("div");
  content.className = "message-content";
  const name = document.createElement("span");
  name.className = "message-name";
  name.textContent = role === "user" ? "你" : "小喵客服";
  const bubble = document.createElement("p");
  bubble.className = "bubble";
  bubble.textContent = text;
  content.append(name, bubble);
  row.append(content);
  messages.append(row);
  scrollToBottom();
  return { row, content, bubble };
}

// Keep decoding and SSE framing independent: neither UTF-8 characters nor event
// boundaries are guaranteed to line up with a network read.
async function readEvents(response, onEvent) {
  if (!response.body || !response.headers.get("content-type")?.includes("text/event-stream")) {
    throw new ChatError("回复格式异常，请稍后重试。");
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8", { fatal: true });
  let buffer = "";
  let eventName = "message";
  let data = [];
  let terminal = false;
  function consumeLines(atEnd = false) {
    while (!terminal) {
      const match = /\r\n|\r|\n/.exec(buffer);
      if (!match || (!atEnd && match[0] === "\r" && match.index === buffer.length - 1)) break;
      const line = buffer.slice(0, match.index);
      buffer = buffer.slice(match.index + match[0].length);
      if (!line) {
        if (data.length) terminal = onEvent(eventName, JSON.parse(data.join("\n")));
        eventName = "message";
        data = [];
      } else if (!line.startsWith(":")) {
        const colon = line.indexOf(":");
        const field = colon < 0 ? line : line.slice(0, colon);
        let value = colon < 0 ? "" : line.slice(colon + 1);
        if (value.startsWith(" ")) value = value.slice(1);
        if (field === "event") eventName = value;
        if (field === "data") data.push(value);
      }
    }
  }
  try {
    while (!terminal) {
      const { value, done } = await reader.read();
      if (done) {
        buffer += decoder.decode();
        consumeLines(true);
        if (!terminal) throw new ChatError("回复连接已中断，请重新发送。这次问答未加入后续对话。");
        break;
      }
      buffer += decoder.decode(value, { stream: true });
      consumeLines();
    }
  } finally {
    try { await reader.cancel(); } catch { /* The connection may already be closed. */ }
    reader.releaseLock();
  }
}

// Paint one Unicode character at a time, even when an upstream delta is a sentence.
function createTyping(bubble) {
  const queue = [];
  let timer = null;
  let resolveIdle = null;
  let shown = "";
  function tick() {
    if (!queue.length) {
      timer = null;
      resolveIdle?.();
      resolveIdle = null;
      return;
    }
    const pinned = messages.scrollHeight - messages.scrollTop - messages.clientHeight < 90;
    shown += queue.shift();
    bubble.textContent = shown;
    if (pinned) scrollToBottom();
    timer = window.setTimeout(tick, queue.length > 100 ? 5 : 18);
  }
  return {
    append(text) {
      queue.push(...Array.from(text));
      if (timer === null) tick();
    },
    idle() {
      return timer === null ? Promise.resolve() : new Promise(resolve => { resolveIdle = resolve; });
    },
    stop() {
      window.clearTimeout(timer);
      timer = null;
      queue.length = 0;
      resolveIdle?.();
      resolveIdle = null;
    },
  };
}

input.addEventListener("input", updateControls);
input.addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    if (!busy && input.value.trim()) form.requestSubmit();
  }
});

// 二次开发新增：意图识别结果的小字标注（置信度 + 路由动作）。
function intentNote(intent, action) {
  if (!intent) return null;
  const label = intent.category_label || intent.category || "?";
  const confidence = typeof intent.confidence === "number" ? intent.confidence.toFixed(2) : "?";
  const icon = { answer: "✅", confirm: "❓", refuse: "🚫" }[action] || "";
  return `${icon} 意图：${label}｜置信度 ${confidence}${action === "refuse" ? "｜超出服务范围，已拒答" : ""}`;
}

// 二次开发新增：中档置信度确认卡片（确认后携带 confirm_intent 重发）。
function showConfirmCard(reply, message, confirmData) {
  reply.bubble.textContent = confirmData.message;
  const actions = document.createElement("div");
  actions.className = "confirm-actions";
  const yes = document.createElement("button");
  yes.type = "button";
  yes.className = "confirm-btn";
  yes.textContent = "✅ 是的，回答它";
  const no = document.createElement("button");
  no.type = "button";
  no.className = "confirm-btn secondary";
  no.textContent = "❌ 不是，重新输入";
  actions.append(yes, no);
  reply.content.append(actions);
  scrollToBottom();

  function dismiss() {
    yes.disabled = true;
    no.disabled = true;
    actions.remove();
  }

  yes.addEventListener("click", () => {
    if (busy) return;
    dismiss();
    converse(message, { confirm: true, addUser: false });
  });
  no.addEventListener("click", () => {
    if (busy) return;
    dismiss();
    reply.bubble.textContent = "好的，已取消本次回答。您可以换个说法重新提问。";
    const note = document.createElement("p");
    note.className = "message-note";
    note.textContent = "已取消 · 这次问答未加入后续对话";
    reply.content.append(note);
    setStatus("已取消本次回答，您可以换个说法重新提问。");
    scrollToBottom();
  });
}

// 二次开发第四批新增：订单工具 / 语义缓存事件的小字标注。
// tool 事件 status 区分三种结果：hit（查到）/ not_found（给了订单号查无此单）/
// need_order_id（没给订单号，服务端已注入引导话术）。
function toolNote(data) {
  if (data.tool === "order_query") {
    const status = data.status || (data.found ? "hit" : "not_found");
    if (status === "need_order_id") return "💬 需要订单号才能查询，请把订单号发给我";
    if (status === "not_found") return `🔍 未查到订单 ${data.order_id || ""}`.trim();
    return `🔧 已查询订单 ${data.order_id || ""}`.trim();
  }
  return null;
}

function cacheNote(data) {
  const score = typeof data.score === "number" ? data.score.toFixed(2) : "?";
  return `⚡ 语义缓存命中（相似度 ${score}），本次未调用模型`;
}

// 统一的对话入口：普通提问与确认重发共用。
async function converse(message, { confirm = false, addUser = true } = {}) {
  busy = true;
  updateControls();
  welcome.hidden = true;
  if (addUser) addMessage("user", message);
  const reply = addMessage("assistant", "");
  reply.bubble.classList.add("pending");
  const typing = createTyping(reply.bubble);
  let answer = "";
  let succeeded = false;
  let finishReason = "";
  let intentData = null;
  let confirmData = null;
  let extraNotes = [];
  setStatus(confirm ? "已确认，小喵正在回答…" : "小喵正在想一想…");
  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json", "Accept": "text/event-stream" },
      body: JSON.stringify({ message, history: history.slice(-100), confirm_intent: confirm }),
    });
    if (!response.ok) {
      const error = await response.json().catch(() => null);
      throw new ChatError(typeof error?.message === "string" ? error.message
        : response.status === 422 ? "这条消息暂时无法发送，请检查内容后重试。" : "暂时无法回复，请稍后重试。");
    }
    await readEvents(response, (eventName, data) => {
      if (eventName === "token") {
        if (typeof data.text !== "string") throw new ChatError("回复格式异常，请重试。");
        answer += data.text;
        typing.append(data.text);
        setStatus("小喵正在回复…");
      } else if (eventName === "intent") {
        // 二次开发新增：记录意图信息，回答结束后展示。
        intentData = data;
      } else if (eventName === "tool") {
        // 二次开发第四批：订单工具执行结果标注。
        const note = toolNote(data);
        if (note) extraNotes.push(note);
      } else if (eventName === "cache") {
        // 二次开发第四批：语义缓存命中标注。
        extraNotes.push(cacheNote(data));
      } else if (eventName === "confirm") {
        // 二次开发新增：中档置信度确认分支——展示确认卡片并结束本次流。
        if (typeof data.message !== "string") throw new ChatError("回复格式异常，请重试。");
        confirmData = data;
        typing.stop();
        reply.bubble.classList.remove("pending");
        showConfirmCard(reply, message, data);
        succeeded = true;
        finishReason = "need_confirm";
        return true;
      } else if (eventName === "done") {
        if (["stop", "length", "refused"].includes(data.finish_reason) && answer.trim()) {
          succeeded = true;
          finishReason = data.finish_reason;
          return true;
        }
        if (data.finish_reason === "need_confirm" && confirmData) {
          succeeded = true;
          finishReason = "need_confirm";
          return true;
        }
        throw new ChatError("回复没有完整送达，请重新发送。");
      } else if (eventName === "error") {
        throw new ChatError(typeof data.message === "string" ? data.message : "回复失败，请重试。");
      }
      return false;
    });
    if (!succeeded) throw new ChatError("回复没有完整送达，请重新发送。");
    if (finishReason !== "need_confirm") {
      await typing.idle();
      if (finishReason !== "refused") {
        history.push({ role: "user", content: message }, { role: "assistant", content: answer });
        history = history.slice(-100);
      }
      const note = intentNote(intentData, intentData?.action || (finishReason === "refused" ? "refuse" : "answer"));
      if (note) extraNotes.unshift(note);
      if (extraNotes.length) {
        const noteEl = document.createElement("p");
        noteEl.className = "message-note";
        noteEl.textContent = extraNotes.join(" · ");
        reply.content.append(noteEl);
      }
      setStatus(finishReason === "length" ? "回复已达到本次长度上限，你可以继续追问。"
        : finishReason === "refused" ? "该问题超出了购物客服的服务范围。"
        : "你可以接着问，我会联系上文回答。");
    } else {
      setStatus("请确认问题是否与您的意图相符。");
    }
  } catch (error) {
    typing.stop();
    reply.row.classList.add("failed");
    if (!reply.bubble.textContent) reply.bubble.textContent = "这次没能完成回复。";
    const note = document.createElement("p");
    note.className = "message-note";
    note.textContent = "未完成 · 这次问答未加入后续对话";
    reply.content.append(note);
    setStatus(error instanceof ChatError ? error.message : "连接发生异常，请重新发送。这次问答未加入后续对话。", true);
    if (addUser) input.value = message;
  } finally {
    reply.bubble.classList.remove("pending");
    busy = false;
    updateControls();
    input.focus({ preventScroll: true });
  }
}

form.addEventListener("submit", event => {
  event.preventDefault();
  const message = input.value.trim();
  if (busy || !message) return;
  input.value = "";
  converse(message);
});

newButton.addEventListener("click", () => {
  if (busy) return;
  history = [];
  messages.querySelectorAll(".message").forEach(message => message.remove());
  welcome.hidden = false;
  input.value = "";
  updateControls();
  setStatus("新对话已开始，之前的内容已清空。");
  input.focus({ preventScroll: true });
  scrollToBottom();
});
