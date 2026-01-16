(() => {
  const chat = document.getElementById("chat");
  const form = document.getElementById("chatForm");
  const textarea = document.getElementById("message");
  const sendBtn = document.getElementById("sendBtn");
  const clearBtn = document.getElementById("clearBtn");
  const statusText = document.getElementById("statusText");
  const statusDot = document.getElementById("statusDot");
  const modeText = document.getElementById("modeText");
  const topKText = document.getElementById("topKText");
  const thrText = document.getElementById("thrText");
  const footnote = document.getElementById("footnote");

  const state = {
    busy: false,
    ready: false,
    history: [],
    top_k: 3,
    similarity_threshold: 0.75,
  };

  function setStatus(kind, text) {
    statusDot.classList.remove("ok", "busy", "err");
    if (kind) statusDot.classList.add(kind);
    statusText.textContent = text;
  }

  function setBusy(isBusy) {
    state.busy = isBusy;
    const disable = isBusy || !state.ready;
    sendBtn.disabled = disable;
    textarea.disabled = disable;
    clearBtn.disabled = disable;
    if (isBusy) {
      setStatus("busy", "Thinking…");
    } else {
      setStatus(
        state.ready ? "ok" : "busy",
        state.ready ? "Ready" : "Loading…"
      );
    }
  }

  function setReady(isReady) {
    state.ready = isReady;
    if (isReady) {
      setStatus("ok", "Ready");
      modeText.textContent = "RAG Online";
      footnote.textContent =
        "Your messages are processed locally by your backend API.";
    } else {
      setStatus("busy", "Loading…");
      modeText.textContent = "Loading";
    }
    setBusy(state.busy);
  }

  function scrollToBottom() {
    chat.scrollTop = chat.scrollHeight;
  }

  function el(tag, className) {
    const n = document.createElement(tag);
    if (className) n.className = className;
    return n;
  }

  function addMessage(role, text, extra = {}) {
    const msg = el("div", `msg ${role}`);

    const header = el("div", "role");
    const roleLabel = el("span");
    roleLabel.textContent = role === "user" ? "You" : "ScholarChat";
    const kpi = el("span", "kpi");
    if (extra.metrics) {
      const m = extra.metrics;
      kpi.textContent = `${m.total_ms}ms`;
    }
    header.appendChild(roleLabel);
    header.appendChild(kpi);

    const body = el("div", "text");
    body.textContent = text;

    msg.appendChild(header);
    msg.appendChild(body);

    if (role === "assistant" && extra.sources && extra.sources.length) {
      const details = el("details", "details");
      const summary = el("summary");
      summary.textContent = `Sources (${extra.sources.length})`;
      details.appendChild(summary);

      const list = el("ul", "sources");
      for (const s of extra.sources) {
        const item = el("li", "source");
        const left = el("div");
        const right = el("div");

        const code = el("code");
        code.textContent = `[Document ${s.document}: ${s.file_name}]`;
        left.appendChild(code);

        const score = el("div", "kpi");
        score.textContent = `score: ${Number(s.score).toFixed(3)}`;
        right.appendChild(score);

        item.appendChild(left);
        item.appendChild(right);
        list.appendChild(item);
      }
      details.appendChild(list);
      msg.appendChild(details);
    }

    chat.appendChild(msg);
    scrollToBottom();
  }

  function pushHistory(role, content) {
    state.history.push({ role, content });
    // Keep history bounded (last 16 items = 8 turns)
    if (state.history.length > 16) {
      state.history = state.history.slice(-16);
    }
  }

  function formatHealthEvent(ev) {
    if (typeof ev === "string") return ev;
    if (!ev || typeof ev !== "object") return String(ev);

    const ts = ev.ts ? String(ev.ts) : "";
    const level = ev.level ? String(ev.level) : "";
    const stageRaw = ev.stage ? String(ev.stage) : "";
    const stage = stageRaw ? stageRaw.replaceAll("_", " ") : "";
    const message = ev.message ? String(ev.message) : JSON.stringify(ev);

    const left = [ts, level].filter(Boolean).join(" ");
    const right = stage ? `${stage}: ${message}` : message;
    return left ? `${left} — ${right}` : right;
  }

  async function checkHealthOnce() {
    try {
      const res = await fetch("http://127.0.0.1:8000/health", {
        cache: "no-store",
      });
      const data = await res.json();

      if (data.ready) {
        setReady(true);
      } else {
        setReady(false);
        const stage = data.stage
          ? String(data.stage).replaceAll("_", " ")
          : "initializing";
        const secs =
          typeof data.elapsed_ms === "number"
            ? Math.floor(data.elapsed_ms / 1000)
            : 0;
        const msg = data.error
          ? `Backend error: ${data.error}`
          : `Loading (${stage}) — ${secs}s elapsed`;
        footnote.textContent = msg;
      }

      topKText.textContent = String(state.top_k);
      thrText.textContent = String(state.similarity_threshold);

      return Boolean(data.ready);
    } catch (e) {
      setReady(false);
      setStatus("err", "Cannot reach backend");
      modeText.textContent = "Offline";
      footnote.textContent = "Start the Python server, then refresh this page.";
      return false;
    }
  }

  async function waitForReady() {
    setReady(false);
    // Poll /health until ready. Long model loads are expected.
    for (;;) {
      const ready = await checkHealthOnce();
      if (ready) return;
      await new Promise((r) => setTimeout(r, 1500));
    }
  }

  async function sendMessage(message) {
    setBusy(true);

    const payload = {
      message,
      history: state.history,
      top_k: state.top_k,
      similarity_threshold: state.similarity_threshold,
    };

    try {
      const res = await fetch("http://127.0.0.1:8000/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(`HTTP ${res.status}: ${text.slice(0, 400)}`);
      }

      const data = await res.json();
      addMessage("assistant", data.answer, {
        sources: data.sources,
        metrics: data.metrics,
      });
      pushHistory("assistant", data.answer);
    } catch (err) {
      addMessage(
        "assistant",
        `Error: ${err.message}\n\nTry again, or check /health for details.`,
        {}
      );
      setStatus("err", "Error");
    } finally {
      setBusy(false);
    }
  }

  function autosize() {
    textarea.style.height = "auto";
    textarea.style.height = Math.min(textarea.scrollHeight, 140) + "px";
  }

  textarea.addEventListener("input", autosize);

  textarea.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      form.requestSubmit();
    }
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (state.busy) return;

    const message = textarea.value.trim();
    if (!message) return;

    textarea.value = "";
    autosize();

    addMessage("user", message);
    pushHistory("user", message);

    await sendMessage(message);
  });

  clearBtn.addEventListener("click", () => {
    chat.innerHTML = "";
    state.history = [];
    addMessage("assistant", "Cleared. Ask a new question when ready.");
  });

  // Boot
  addMessage(
    "assistant",
    "Hi — I can answer questions using your course materials.\n\nAsk a question, and I will cite the documents I used."
  );

  // Enable interface immediately - models are already loaded
  setReady(true);
  setBusy(false);

  // Check health in background for status display
  checkHealthOnce();
})();
