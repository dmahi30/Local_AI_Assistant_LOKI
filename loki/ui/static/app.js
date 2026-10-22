/**
 * LOKI Desktop UI Client Controller
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const sidebar = document.getElementById("sidebar");
  const toggleSidebarBtn = document.getElementById("toggleSidebarBtn");
  const newChatBtn = document.getElementById("newChatBtn");
  const clearChatBtn = document.getElementById("clearChatBtn");
  const navSettingsBtn = document.getElementById("navSettingsBtn");
  const toggleDrawerBtn = document.getElementById("toggleDrawerBtn");
  const closeDrawerBtn = document.getElementById("closeDrawerBtn");
  const statusDrawer = document.getElementById("statusDrawer");
  const refreshStatusBtn = document.getElementById("refreshStatusBtn");

  const ollamaStatusPill = document.getElementById("ollamaStatusPill");
  const statusDot = document.getElementById("statusDot");
  const statusPillText = document.getElementById("statusPillText");

  const messagesViewport = document.getElementById("messagesViewport");
  const emptyState = document.getElementById("emptyState");
  const chatStream = document.getElementById("chatStream");
  const composerForm = document.getElementById("composerForm");
  const messageInput = document.getElementById("messageInput");
  const sendBtn = document.getElementById("sendBtn");
  const voiceBtn = document.getElementById("voiceBtn");
  const attachBtn = document.getElementById("attachBtn");

  // Drawer Elements
  const drawerOllamaBadge = document.getElementById("drawerOllamaBadge");
  const drawerOllamaHost = document.getElementById("drawerOllamaHost");
  const drawerOllamaVersion = document.getElementById("drawerOllamaVersion");
  const drawerActiveModel = document.getElementById("drawerActiveModel");
  const drawerHardwareTier = document.getElementById("drawerHardwareTier");
  const drawerRamText = document.getElementById("drawerRamText");
  const drawerRamBar = document.getElementById("drawerRamBar");
  const drawerCpuText = document.getElementById("drawerCpuText");
  const drawerCpuBar = document.getElementById("drawerCpuBar");

  let isInferring = false;
  let lastPrompt = "";

  // =========================================================================
  // 1. Sidebar & Drawer Toggles
  // =========================================================================
  toggleSidebarBtn.addEventListener("click", () => {
    sidebar.classList.toggle("collapsed");
  });

  const openDrawer = () => {
    statusDrawer.classList.add("open");
    fetchStatus();
  };

  const closeDrawer = () => {
    statusDrawer.classList.remove("open");
  };

  toggleDrawerBtn.addEventListener("click", () => {
    if (statusDrawer.classList.contains("open")) {
      closeDrawer();
    } else {
      openDrawer();
    }
  });

  closeDrawerBtn.addEventListener("click", closeDrawer);
  navSettingsBtn.addEventListener("click", openDrawer);
  refreshStatusBtn.addEventListener("click", fetchStatus);

  // New Chat Action
  newChatBtn.addEventListener("click", resetConversation);
  clearChatBtn.addEventListener("click", resetConversation);

  // Prompt Starter Chips
  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const prompt = chip.getAttribute("data-prompt");
      if (prompt && !isInferring) {
        messageInput.value = prompt;
        submitMessage(prompt);
      }
    });
  });

  // Planned action buttons feedback
  if (voiceBtn) {
    voiceBtn.addEventListener("click", () => {
      if (emptyState) emptyState.style.display = "none";
      appendAssistantMessage({
        text: "🎤 **Voice Input (Phase 6)**: Offline speech recognition is planned for Phase 6 and is currently disabled. Please use keyboard input for now.",
        source: "system",
        success: true,
      });
      scrollToBottom();
    });
  }

  if (attachBtn) {
    attachBtn.addEventListener("click", () => {
      if (emptyState) emptyState.style.display = "none";
      appendAssistantMessage({
        text: "📎 **File Attachments (Phase 7)**: Document and image ingestion is planned for a future phase. Please use direct text commands or file paths (e.g., search, delete).",
        source: "system",
        success: true,
      });
      scrollToBottom();
    });
  }

  // Keyboard Shortcuts
  document.addEventListener("keydown", (e) => {
    if (e.ctrlKey && e.key.toLowerCase() === "b") {
      e.preventDefault();
      sidebar.classList.toggle("collapsed");
    }
    if (e.ctrlKey && e.key.toLowerCase() === "l") {
      e.preventDefault();
      resetConversation();
    }
    if (e.key === "Escape" && statusDrawer.classList.contains("open")) {
      closeDrawer();
    }
  });

  // =========================================================================
  // 2. Input Sizing & Submission
  // =========================================================================
  messageInput.addEventListener("input", () => {
    messageInput.style.height = "auto";
    messageInput.style.height = Math.min(messageInput.scrollHeight, 140) + "px";
  });

  messageInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!isInferring) {
        composerForm.dispatchEvent(new Event("submit"));
      }
    }
  });

  composerForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = messageInput.value.trim();
    if (text && !isInferring) {
      submitMessage(text);
    }
  });

  // =========================================================================
  // 3. Message Handling & Backend Communication
  // =========================================================================
  async function submitMessage(text) {
    lastPrompt = text;
    messageInput.value = "";
    messageInput.style.height = "auto";
    setInferringState(true);

    if (emptyState) {
      emptyState.style.display = "none";
    }

    // Append User Message
    appendUserMessage(text);

    // Append Loading Indicator
    const loadingElem = appendLoadingIndicator();
    scrollToBottom();

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      });

      const data = await res.json();
      loadingElem.remove();

      if (data.pending_confirmation) {
        appendConfirmationCard(data.pending_confirmation, text);
      } else {
        appendAssistantMessage(data);
      }
    } catch (err) {
      loadingElem.remove();
      appendAssistantMessage({
        text: `Network or runtime communication error: ${err.message}`,
        source: "system",
        success: false,
      });
    } finally {
      setInferringState(false);
      scrollToBottom();
      fetchStatus();
    }
  }

  function setInferringState(loading) {
    isInferring = loading;
    sendBtn.disabled = loading;
    messageInput.disabled = loading;
    if (!loading) {
      messageInput.focus();
    }
  }

  function scrollToBottom() {
    messagesViewport.scrollTop = messagesViewport.scrollHeight;
  }

  function formatTime(date = new Date()) {
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  // =========================================================================
  // 4. DOM Rendering for Messages
  // =========================================================================
  function appendUserMessage(text) {
    const row = document.createElement("div");
    row.className = "message-row user";

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;

    const meta = document.createElement("div");
    meta.className = "bubble-meta";
    meta.innerHTML = `<span></span><span>${formatTime()}</span>`;

    bubble.appendChild(meta);
    row.appendChild(bubble);
    chatStream.appendChild(row);
  }

  function appendLoadingIndicator() {
    const row = document.createElement("div");
    row.className = "message-row assistant loading-row";
    row.id = "activeLoadingIndicator";

    row.innerHTML = `
      <div class="avatar-circle">
        <img src="/static/emblem.svg" alt="Loki" class="avatar-img">
      </div>
      <div class="loading-bubble">
        <span class="dot"></span>
        <span class="dot"></span>
        <span class="dot"></span>
      </div>
    `;

    chatStream.appendChild(row);
    return row;
  }

  function formatMarkdown(rawText) {
    if (!rawText) return "";
    let safe = rawText
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");

    // Code blocks ```code```
    safe = safe.replace(/```([\s\S]*?)```/g, (match, code) => {
      return `<pre><code>${code.trim()}</code></pre>`;
    });

    // Inline code `code`
    safe = safe.replace(/`([^`]+)`/g, "<code>$1</code>");

    // Bold **text**
    safe = safe.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");

    // Line breaks
    safe = safe.replace(/\n\n/g, "</p><p>").replace(/\n/g, "<br>");

    return `<p>${safe}</p>`;
  }

  function appendAssistantMessage(data) {
    const row = document.createElement("div");
    row.className = "message-row assistant";

    const badgeClass =
      data.source === "deterministic" ? "badge-deterministic" : "badge-llm";
    const badgeLabel =
      data.source === "deterministic"
        ? `SYSTEM: ${data.matched_command || "COMMAND"}`
        : data.model_used || "LOKI AI";

    row.innerHTML = `
      <div class="avatar-circle">
        <img src="/static/emblem.svg" alt="Loki" class="avatar-img">
      </div>
      <div class="bubble">
        <div class="bubble-content">${formatMarkdown(data.text)}</div>
        <div class="bubble-meta">
          <span class="bubble-badge ${badgeClass}">${badgeLabel}</span>
          <div class="bubble-actions">
            <button class="copy-btn" title="Copy response">
              <svg viewBox="0 0 24 24" width="13" height="13" stroke="currentColor" stroke-width="2" fill="none"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
            </button>
            <span class="bubble-time">${formatTime()}</span>
          </div>
        </div>
      </div>
    `;

    // Copy action
    const copyBtn = row.querySelector(".copy-btn");
    copyBtn.addEventListener("click", () => {
      navigator.clipboard.writeText(data.text).then(() => {
        copyBtn.innerHTML = "✓";
        setTimeout(() => {
          copyBtn.innerHTML = `
            <svg viewBox="0 0 24 24" width="13" height="13" stroke="currentColor" stroke-width="2" fill="none"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
          `;
        }, 1500);
      });
    });

    chatStream.appendChild(row);
  }

  function appendConfirmationCard(confirmData, originalPrompt) {
    const row = document.createElement("div");
    row.className = "message-row assistant";

    const targetsStr = confirmData.targets && confirmData.targets.length
      ? confirmData.targets.join(", ")
      : "None specified";

    row.innerHTML = `
      <div class="avatar-circle">
        <img src="/static/emblem.svg" alt="Loki" class="avatar-img">
      </div>
      <div class="bubble" style="max-width: 90%;">
        <div class="confirmation-card">
          <div class="confirm-header">
            <span>⚠️ Security Confirmation Gate</span>
          </div>
          <div class="confirm-desc">
            ${confirmData.description}
          </div>
          <div class="confirm-targets">
            Targets: ${targetsStr}<br>
            Action ID: ${confirmData.action_id}
          </div>
          <div class="confirm-btn-row">
            <button class="btn-approve" id="approveBtn_${confirmData.action_id}">Approve Action</button>
            <button class="btn-cancel" id="cancelBtn_${confirmData.action_id}">Cancel</button>
          </div>
        </div>
        <div class="bubble-meta">
          <span class="bubble-badge badge-deterministic">SECURITY GATEKEEPER</span>
          <span class="bubble-time">${formatTime()}</span>
        </div>
      </div>
    `;

    chatStream.appendChild(row);

    const approveBtn = row.querySelector(`#approveBtn_${confirmData.action_id}`);
    const cancelBtn = row.querySelector(`#cancelBtn_${confirmData.action_id}`);

    approveBtn.addEventListener("click", async () => {
      approveBtn.disabled = true;
      cancelBtn.disabled = true;
      approveBtn.textContent = "Executing...";

      try {
        const res = await fetch("/api/confirm", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            action_id: confirmData.action_id,
            original_message: originalPrompt,
          }),
        });
        const result = await res.json();
        appendAssistantMessage(result);
      } catch (err) {
        appendAssistantMessage({
          text: `Failed to execute approved action: ${err.message}`,
          source: "system",
          success: false,
        });
      } finally {
        scrollToBottom();
      }
    });

    cancelBtn.addEventListener("click", async () => {
      approveBtn.disabled = true;
      cancelBtn.disabled = true;
      cancelBtn.textContent = "Canceled";

      await fetch("/api/cancel", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action_id: confirmData.action_id }),
      });

      appendAssistantMessage({
        text: `Action '${confirmData.action_id}' was canceled. No changes were made.`,
        source: "system",
        success: true,
      });
      scrollToBottom();
    });
  }

  async function resetConversation() {
    try {
      await fetch("/api/clear", { method: "POST" });
    } catch (e) {
      console.warn("Could not clear backend history", e);
    }
    chatStream.innerHTML = "";
    if (emptyState) {
      emptyState.style.display = "flex";
    }
    messageInput.value = "";
    messageInput.style.height = "auto";
    messageInput.focus();
  }

  // =========================================================================
  // 5. System Status Polling
  // =========================================================================
  async function fetchStatus() {
    try {
      const res = await fetch("/api/status");
      if (!res.ok) throw new Error("Status endpoint error");
      const data = await res.json();
      updateStatusUI(data);
    } catch (err) {
      updateStatusUI({ ollama_online: false, error: err.message });
    }
  }

  function updateStatusUI(data) {
    if (!data.ollama_online) {
      statusDot.className = "status-dot offline";
      statusPillText.textContent = "Ollama: Offline";
      drawerOllamaBadge.textContent = "Offline";
      drawerOllamaBadge.className = "metric-badge offline";
    } else if (data.model_available) {
      statusDot.className = "status-dot online";
      if (data.using_fallback) {
        statusPillText.textContent = `Ollama: Connected (${data.effective_model} [Fallback])`;
      } else {
        statusPillText.textContent = `Ollama: Connected (${data.active_model})`;
      }
      drawerOllamaBadge.textContent = "Online";
      drawerOllamaBadge.className = "metric-badge online";
    } else {
      // Ollama daemon is online, but active model is not installed locally
      statusDot.className = "status-dot warning";
      statusPillText.textContent = `Ollama: Connected (Model Missing: ${data.active_model})`;
      drawerOllamaBadge.textContent = "Model Missing";
      drawerOllamaBadge.className = "metric-badge warning";
    }

    if (data.ollama_host) drawerOllamaHost.textContent = data.ollama_host;
    if (data.ollama_version) drawerOllamaVersion.textContent = data.ollama_version;

    if (data.active_model) {
      if (data.model_available) {
        drawerActiveModel.textContent = data.using_fallback
          ? `${data.effective_model} (Fallback)`
          : data.active_model;
        drawerActiveModel.className = "detail-val gold-accent";
      } else {
        drawerActiveModel.textContent = `${data.active_model} (Not Installed)`;
        drawerActiveModel.className = "detail-val red-accent";
      }
    }
    if (data.hardware_tier) drawerHardwareTier.textContent = data.hardware_tier.toUpperCase();

    if (data.ram_used_gb !== undefined && data.ram_total_gb !== undefined) {
      drawerRamText.textContent = `${data.ram_used_gb} GB / ${data.ram_total_gb} GB (${data.ram_percent}%)`;
      drawerRamBar.style.width = `${Math.min(data.ram_percent, 100)}%`;
    }

    if (data.cpu_percent !== undefined) {
      drawerCpuText.textContent = `${data.cpu_percent}% across ${data.cpu_cores} cores`;
      drawerCpuBar.style.width = `${Math.min(data.cpu_percent, 100)}%`;
    }
  }

  // Initial status check & periodic refresh
  fetchStatus();
  setInterval(fetchStatus, 8000);
});
