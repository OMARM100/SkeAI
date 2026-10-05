const messages = document.getElementById("messages");
const form = document.getElementById("chat-form");
const input = document.getElementById("message");
const send = document.getElementById("send");
const status = document.getElementById("status");

function addMessage(role, text) {
  const wrapper = document.createElement("div");
  wrapper.className = "message " + role;

  const label = document.createElement("div");
  label.className = "label";
  label.textContent = role === "user" ? "أنت" : "SkeAI";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  wrapper.append(label, bubble);
  messages.appendChild(wrapper);
  messages.scrollTop = messages.scrollHeight;
  return bubble;
}

async function loadStatus() {
  try {
    const response = await fetch("/api/status");
    const data = await response.json();
    if (!response.ok || !data.ok) throw new Error(data.error || "status error");
    const modelLabel = data.model_type === "level2_transformer"
      ? "Transformer"
      : "نموذج حرفي";
    status.textContent =
      modelLabel + " · مفردات " + data.vocabulary_size +
      " · سياق " + data.context_length +
      " · معاملات " + data.parameter_count;
  } catch (error) {
    status.textContent = "تعذر تحميل حالة النموذج";
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message || send.disabled) return;

  addMessage("user", message);
  input.value = "";
  send.disabled = true;
  send.textContent = "...";
  const pending = addMessage("assistant", "جاري التوليد...");

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        max_new_tokens: 64,
        temperature: 0.85,
        top_k: 0,
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "حدث خطأ");
    pending.textContent = data.response || "لم يُنتج النموذج نصًا.";
  } catch (error) {
    pending.textContent = "خطأ: " + error.message;
  } finally {
    send.disabled = false;
    send.textContent = "إرسال";
    input.focus();
  }
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

loadStatus();
input.focus();
