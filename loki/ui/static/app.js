/**
 * LOKI Desktop UI Client Controller
 */

function initLokiApp() {
  console.log(`[LOKI UI] Initializing client controller (readyState=${document.readyState})...`);
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

  const recentList = document.getElementById("recentList");
  const searchChatsBtn = document.getElementById("searchChatsBtn");
  const recentSearchBar = document.getElementById("recentSearchBar");
  const recentSearchInput = document.getElementById("recentSearchInput");

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
  const filePicker = document.getElementById("filePicker");
  const attachmentPreview = document.getElementById("attachmentPreview");
  const attachmentName = document.getElementById("attachmentName");
  const attachmentSize = document.getElementById("attachmentSize");
  const removeAttachmentBtn = document.getElementById("removeAttachmentBtn");
  let currentAttachmentFile = null;

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

  // State Management
  function getSavedActiveConvId() {
    try {
      return localStorage.getItem("loki_active_conv_id") || null;
    } catch {
      return null;
    }
  }

  function saveActiveConvId(id) {
    try {
      if (id) {
        localStorage.setItem("loki_active_conv_id", id);
      } else {
        localStorage.removeItem("loki_active_conv_id");
      }
    } catch {
      // Safe fallback if local storage is restricted
    }
  }

  let activeConversationId = getSavedActiveConvId();
  let conversations = [];
  let isInferring = false;
  let lastPrompt = "";

  // =========================================================================
  // 1. Sidebar & Drawer Toggles
  // =========================================================================
  if (toggleSidebarBtn && sidebar) {
    toggleSidebarBtn.addEventListener("click", () => {
      sidebar.classList.toggle("collapsed");
    });
  }

  const openDrawer = () => {
    if (statusDrawer) {
      statusDrawer.classList.add("open");
      fetchStatus();
    }
  };

  const closeDrawer = () => {
    if (statusDrawer) {
      statusDrawer.classList.remove("open");
    }
  };

  if (toggleDrawerBtn && statusDrawer) {
    toggleDrawerBtn.addEventListener("click", () => {
      if (statusDrawer.classList.contains("open")) {
        closeDrawer();
      } else {
        openDrawer();
      }
    });
  }

  if (closeDrawerBtn) closeDrawerBtn.addEventListener("click", closeDrawer);
  if (navSettingsBtn) navSettingsBtn.addEventListener("click", openDrawer);
  if (refreshStatusBtn) refreshStatusBtn.addEventListener("click", fetchStatus);

  // New Chat and Clear Actions
  if (newChatBtn) {
    newChatBtn.addEventListener("click", (e) => {
      e.preventDefault();
      createNewConversation();
    });
  }

  if (clearChatBtn) {
    clearChatBtn.addEventListener("click", (e) => {
      e.preventDefault();
      clearActiveChat();
    });
  }

  // Search Toggle and Filter
  if (searchChatsBtn && recentSearchBar) {
    searchChatsBtn.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      console.log("[LOKI UI] Search toggle clicked");

      // If sidebar is collapsed, expand it so the search bar is clearly visible
      if (sidebar && sidebar.classList.contains("collapsed")) {
        sidebar.classList.remove("collapsed");
      }

      const isHidden = window.getComputedStyle(recentSearchBar).display === "none";
      if (isHidden) {
        recentSearchBar.style.display = "block";
        if (recentSearchInput) {
          recentSearchInput.value = "";
          recentSearchInput.focus();
        }
      } else {
        recentSearchBar.style.display = "none";
        if (recentSearchInput) {
          recentSearchInput.value = "";
        }
        renderRecentList("");
      }
    });
  }

  if (recentSearchInput) {
    recentSearchInput.addEventListener("input", (e) => {
      renderRecentList(e.target.value);
    });
    recentSearchInput.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        e.preventDefault();
        recentSearchInput.value = "";
        if (recentSearchBar) recentSearchBar.style.display = "none";
        renderRecentList("");
      }
    });
  }

  // Prompt Starter Chips
  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const prompt = chip.getAttribute("data-prompt");
      if (prompt && !isInferring && messageInput) {
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

  // File Attachment Handling
  function clearAttachment() {
    currentAttachmentFile = null;
    if (filePicker) filePicker.value = "";
    if (attachmentPreview) attachmentPreview.style.display = "none";
  }

  if (attachBtn && filePicker) {
    attachBtn.addEventListener("click", () => {
      if (isInferring) return;
      filePicker.click();
    });

    filePicker.addEventListener("change", () => {
      if (!filePicker.files || filePicker.files.length === 0) return;
      const file = filePicker.files[0];
      const ext = "." + (file.name.split(".").pop() || "").toLowerCase();
      const allowedExts = [".txt", ".pdf", ".csv", ".md"];

      if (!allowedExts.includes(ext)) {
        appendAssistantMessage({
          text: `⚠️ **Unsupported file type** (\`${escapeHtml(ext)}\`). LOKI supports plain-text (\`.txt\`, \`.md\`, \`.csv\`) and PDF (\`.pdf\`) files.`,
          source: "system",
          success: false,
        });
        scrollToBottom();
        clearAttachment();
        return;
      }

      const maxBytes = 20 * 1024 * 1024; // 20 MB
      if (file.size > maxBytes) {
        appendAssistantMessage({
          text: `⚠️ **File too large**: "${escapeHtml(file.name)}" (${formatBytes(file.size)}) exceeds the 20 MB size limit.`,
          source: "system",
          success: false,
        });
        scrollToBottom();
        clearAttachment();
        return;
      }

      if (file.size === 0) {
        appendAssistantMessage({
          text: `⚠️ **Empty file**: "${escapeHtml(file.name)}" contains 0 bytes.`,
          source: "system",
          success: false,
        });
        scrollToBottom();
        clearAttachment();
        return;
      }

      currentAttachmentFile = file;
      if (attachmentName) attachmentName.textContent = file.name;
      if (attachmentSize) attachmentSize.textContent = formatBytes(file.size);
      if (attachmentPreview) attachmentPreview.style.display = "flex";
      if (messageInput) messageInput.focus();
    });
  }

  if (removeAttachmentBtn) {
    removeAttachmentBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      clearAttachment();
      if (messageInput) messageInput.focus();
    });
  }

  // Keyboard Shortcuts
  document.addEventListener("keydown", (e) => {
    if (e.ctrlKey && e.key.toLowerCase() === "b") {
      e.preventDefault();
      if (sidebar) sidebar.classList.toggle("collapsed");
    }
    if (e.ctrlKey && e.key.toLowerCase() === "l") {
      e.preventDefault();
      clearActiveChat();
    }
    if (e.key === "Escape" && statusDrawer && statusDrawer.classList.contains("open")) {
      closeDrawer();
    }
  });

  // =========================================================================
  // 2. Input Sizing & Submission
  // =========================================================================
  if (messageInput) {
    messageInput.addEventListener("input", () => {
      messageInput.style.height = "auto";
      messageInput.style.height = Math.min(messageInput.scrollHeight, 140) + "px";
    });

    messageInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        if (!isInferring && composerForm) {
          composerForm.dispatchEvent(new Event("submit"));
        }
      }
    });
  }

  if (composerForm) {
    composerForm.addEventListener("submit", (e) => {
      e.preventDefault();
      if (!messageInput) return;
      const text = messageInput.value.trim();
      if ((text || currentAttachmentFile) && !isInferring) {
        submitMessage(text);
      }
    });
  }

  // =========================================================================
  // 3. Conversation Management
  // =========================================================================
  function escapeHtml(str) {
    if (!str) return "";
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function formatBytes(bytes) {
    if (!bytes || bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
  }

  function formatRelativeTime(isoStr) {
    if (!isoStr) return "";
    try {
      const date = new Date(isoStr);
      const now = new Date();
      const diffSecs = Math.floor((now - date) / 1000);
      if (diffSecs < 60) return "Just now";
      const diffMins = Math.floor(diffSecs / 60);
      if (diffMins < 60) return `${diffMins}m ago`;
      const diffHours = Math.floor(diffMins / 60);
      if (diffHours < 24) return `${diffHours}h ago`;
      return date.toLocaleDateString([], { month: "short", day: "numeric" });
    } catch {
      return "";
    }
  }

  async function loadConversations() {
    try {
      console.log("[LOKI UI] Fetching conversations from /api/conversations...");
      const res = await fetch("/api/conversations");
      if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to fetch conversations`);
      const data = await res.json();
      conversations = data.conversations || [];
      console.log(`[LOKI UI] Successfully loaded ${conversations.length} conversation(s).`);

      if (conversations.length === 0) {
        await createNewConversation();
      } else {
        renderRecentList();
        const exists = conversations.some((c) => c.id === activeConversationId);
        const targetId = exists ? activeConversationId : conversations[0].id;
        await selectConversation(targetId);
      }
    } catch (err) {
      console.error("[LOKI UI] Error loading conversations:", err);
      if (recentList) {
        recentList.innerHTML = `
          <div class="recent-empty error">
            <span>Unable to load conversations.</span>
            <button type="button" class="recent-retry-btn" id="retryLoadConvBtn">Retry</button>
          </div>
        `;
        const retryBtn = document.getElementById("retryLoadConvBtn");
        if (retryBtn) {
          retryBtn.addEventListener("click", () => loadConversations());
        }
      }
    }
  }

  async function fetchConversationsListSilently() {
    try {
      const res = await fetch("/api/conversations");
      if (res.ok) {
        const data = await res.json();
        conversations = data.conversations || [];
        const query = recentSearchInput ? recentSearchInput.value : "";
        renderRecentList(query);
      }
    } catch (e) {
      console.warn("Silent conversation refresh failed:", e);
    }
  }

  function renderRecentList(filterQuery = "") {
    if (!recentList) return;
    recentList.innerHTML = "";

    const q = filterQuery.trim().toLowerCase();
    const filtered = q
      ? conversations.filter((c) => (c.title || "").toLowerCase().includes(q))
      : conversations;

    if (filtered.length === 0) {
      const emptyNotice = document.createElement("div");
      emptyNotice.className = "recent-empty";
      emptyNotice.textContent = q ? "No matching chats" : "No recent chats";
      recentList.appendChild(emptyNotice);
      return;
    }

    // Defensive deduplication by conversation ID
    const seenIds = new Set();
    const uniqueFiltered = [];
    for (const c of filtered) {
      if (c && c.id && !seenIds.has(c.id)) {
        seenIds.add(c.id);
        uniqueFiltered.push(c);
      }
    }

    uniqueFiltered.forEach((conv) => {
      const item = document.createElement("div");
      item.className = "recent-item" + (conv.id === activeConversationId ? " active" : "");
      item.dataset.chatId = conv.id;

      item.innerHTML = `
        <span class="recent-icon">
          <svg viewBox="0 0 24 24" width="14" height="14" stroke="currentColor" stroke-width="2" fill="none"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
        </span>
        <div class="recent-info">
          <span class="recent-name" title="${escapeHtml(conv.title)}">${escapeHtml(conv.title)}</span>
          <span class="recent-time">${formatRelativeTime(conv.updated_at)}</span>
        </div>
        <div class="recent-actions">
          <button type="button" class="recent-action-btn rename-btn" title="Rename chat" aria-label="Rename chat">
            <svg viewBox="0 0 24 24" width="12" height="12" stroke="currentColor" stroke-width="2" fill="none"><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/></svg>
          </button>
          <button type="button" class="recent-action-btn delete-btn" title="Delete chat" aria-label="Delete chat">
            <svg viewBox="0 0 24 24" width="12" height="12" stroke="currentColor" stroke-width="2" fill="none"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
          </button>
        </div>
      `;

      item.addEventListener("click", (e) => {
        if (e.target.closest(".recent-actions") || e.target.closest(".recent-edit-input")) {
          return;
        }
        if (conv.id !== activeConversationId) {
          selectConversation(conv.id);
        }
      });

      const renameBtn = item.querySelector(".rename-btn");
      if (renameBtn) {
        renameBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          const nameSpan = item.querySelector(".recent-name");
          if (nameSpan) {
            startRenameConversation(conv.id, nameSpan, conv.title);
          }
        });
      }

      const deleteBtn = item.querySelector(".delete-btn");
      if (deleteBtn) {
        deleteBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          deleteConversation(conv.id, conv.title);
        });
      }

      recentList.appendChild(item);
    });
  }

  function startRenameConversation(convId, nameSpan, currentTitle) {
    const input = document.createElement("input");
    input.type = "text";
    input.className = "recent-edit-input";
    input.value = currentTitle;

    const parent = nameSpan.parentElement;
    parent.replaceChild(input, nameSpan);
    input.focus();
    input.select();

    let finished = false;
    const finish = async (save) => {
      if (finished) return;
      finished = true;
      const newTitle = input.value.trim();

      if (save && newTitle && newTitle !== currentTitle) {
        try {
          const res = await fetch(`/api/conversations/${convId}`, {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ title: newTitle }),
          });
          if (res.ok) {
            const data = await res.json();
            const idx = conversations.findIndex((c) => c.id === convId);
            if (idx !== -1 && data.conversation) {
              conversations[idx].title = data.conversation.title;
            }
          }
        } catch (err) {
          console.error("Rename conversation failed:", err);
        }
      }
      renderRecentList(recentSearchInput ? recentSearchInput.value : "");
    };

    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        finish(true);
      } else if (e.key === "Escape") {
        e.preventDefault();
        finish(false);
      }
    });

    input.addEventListener("blur", () => {
      finish(true);
    });
  }

  async function deleteConversation(convId, title) {
    const confirmed = window.confirm(`Delete conversation "${title}"? This cannot be undone.`);
    if (!confirmed) return;

    try {
      const res = await fetch(`/api/conversations/${convId}`, {
        method: "DELETE",
      });
      if (res.ok) {
        if (activeConversationId === convId) {
          activeConversationId = null;
          saveActiveConvId(null);
          if (chatStream) chatStream.innerHTML = "";
          if (emptyState) emptyState.style.display = "flex";
        }
        await loadConversations();
      } else {
        let errorDetail = `HTTP ${res.status}`;
        try {
          const errData = await res.json();
          if (errData && errData.detail) errorDetail = errData.detail;
        } catch {
          if (res.statusText) errorDetail = `HTTP ${res.status}: ${res.statusText}`;
        }
        console.error(`[LOKI UI] Delete conversation failed: ${errorDetail}`);
        alert(`Could not delete conversation: ${errorDetail}`);
      }
    } catch (err) {
      console.error("[LOKI UI] Delete conversation failed due to network error:", err);
      alert(`Could not delete conversation due to a network error: ${err.message || err}`);
    }
  }

  async function selectConversation(convId) {
    activeConversationId = convId;
    saveActiveConvId(convId);

    document.querySelectorAll(".recent-item").forEach((el) => {
      if (el.dataset.chatId === convId) {
        el.classList.add("active");
      } else {
        el.classList.remove("active");
      }
    });

    try {
      const res = await fetch(`/api/conversations/${convId}`);
      if (!res.ok) {
        if (res.status === 404) {
          saveActiveConvId(null);
          activeConversationId = null;
          await loadConversations();
          return;
        }
        throw new Error("Failed to load conversation");
      }

      const data = await res.json();
      const detail = data.conversation;
      if (chatStream) chatStream.innerHTML = "";

      if (!detail || !detail.messages || detail.messages.length === 0) {
        if (emptyState) emptyState.style.display = "flex";
      } else {
        if (emptyState) emptyState.style.display = "none";
        detail.messages.forEach((msg) => {
          if (msg.role === "user") {
            const att = (msg.metadata && msg.metadata.attachment) ? msg.metadata.attachment : null;
            appendUserMessage(msg.content, msg.created_at, att);
          } else if (msg.role === "assistant") {
            appendAssistantMessage({
              text: msg.content,
              source: msg.source,
              model_used: msg.model_used,
              matched_command: msg.metadata ? msg.metadata.command_name : null,
              success: true,
            }, msg.created_at);
          }
        });
        scrollToBottom();
      }
    } catch (err) {
      console.error("Error selecting conversation:", err);
    }
  }

  async function createNewConversation(title = "New Chat") {
    if (isInferring) return;
    // If the active conversation is already empty and titled "New Chat", stay on it
    if (activeConversationId) {
      const activeItem = conversations.find((c) => c.id === activeConversationId);
      if (activeItem && activeItem.title === "New Chat" && (!activeItem.message_count || activeItem.message_count === 0)) {
        if (chatStream) chatStream.innerHTML = "";
        if (emptyState) emptyState.style.display = "flex";
        if (messageInput) {
          messageInput.value = "";
          messageInput.style.height = "auto";
          messageInput.focus();
        }
        return;
      }
    }
    try {
      const res = await fetch("/api/conversations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title }),
      });
      if (!res.ok) throw new Error("Failed to create conversation");
      const data = await res.json();
      const newConv = data.conversation;
      activeConversationId = newConv.id;
      saveActiveConvId(activeConversationId);

      if (chatStream) chatStream.innerHTML = "";
      if (emptyState) emptyState.style.display = "flex";
      if (messageInput) {
        messageInput.value = "";
        messageInput.style.height = "auto";
        messageInput.focus();
      }

      await fetchConversationsListSilently();
    } catch (err) {
      console.error("Failed to create new conversation:", err);
    }
  }

  async function clearActiveChat() {
    if (isInferring) return;
    try {
      await fetch("/api/clear", { method: "POST" });
    } catch (e) {
      console.warn("Could not clear backend history", e);
    }
    if (chatStream) chatStream.innerHTML = "";
    if (emptyState) emptyState.style.display = "flex";
    if (messageInput) {
      messageInput.value = "";
      messageInput.style.height = "auto";
      messageInput.focus();
    }
    await fetchConversationsListSilently();
  }

  // =========================================================================
  // 4. Message Handling & Backend Communication
  // =========================================================================
  async function submitMessage(text) {
    const fileToUpload = currentAttachmentFile;
    // Clear composer attachment selection immediately
    clearAttachment();

    lastPrompt = text;
    if (messageInput) {
      messageInput.value = "";
      messageInput.style.height = "auto";
    }
    setInferringState(true);

    if (emptyState) {
      emptyState.style.display = "none";
    }

    let uploadedPayload = null;

    // Stage attachment file to backend if attached
    if (fileToUpload) {
      const uploadLoadingElem = appendLoadingIndicator();
      scrollToBottom();
      try {
        const formData = new FormData();
        formData.append("file", fileToUpload);
        const upRes = await fetch("/api/upload", {
          method: "POST",
          body: formData,
        });
        if (!upRes.ok) {
          let errDetail = `HTTP ${upRes.status}`;
          try {
            const errData = await upRes.json();
            if (errData && errData.detail) errDetail = errData.detail;
          } catch {}
          throw new Error(errDetail);
        }
        const upData = await upRes.json();
        uploadedPayload = {
          filename: upData.filename,
          path: upData.path,
          size_bytes: upData.size_bytes,
        };
      } catch (uploadErr) {
        if (uploadLoadingElem) uploadLoadingElem.remove();
        appendAssistantMessage({
          text: `⚠️ **Attachment Upload Failed**: ${uploadErr.message}`,
          source: "system",
          success: false,
        });
        setInferringState(false);
        scrollToBottom();
        return;
      }
      if (uploadLoadingElem) uploadLoadingElem.remove();
    }

    const effectivePrompt = text || (uploadedPayload ? `Summarize ${uploadedPayload.filename}.` : "");

    // Append User Message with attachment pill if attached
    appendUserMessage(effectivePrompt, new Date(), uploadedPayload);

    // Append Loading Indicator
    const loadingElem = appendLoadingIndicator();
    scrollToBottom();

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: effectivePrompt,
          conversation_id: activeConversationId,
          attachment: uploadedPayload,
        }),
      });

      const data = await res.json();
      if (loadingElem) loadingElem.remove();

      if (data.conversation_id && data.conversation_id !== activeConversationId) {
        activeConversationId = data.conversation_id;
        saveActiveConvId(activeConversationId);
      }

      if (data.pending_confirmation) {
        appendConfirmationCard(data.pending_confirmation, effectivePrompt);
      } else {
        appendAssistantMessage(data);
      }

      // Update sidebar list (e.g. auto-title updated, activity timestamp)
      fetchConversationsListSilently();
    } catch (err) {
      if (loadingElem) loadingElem.remove();
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
    if (sendBtn) sendBtn.disabled = loading;
    if (messageInput) {
      messageInput.disabled = loading;
      if (!loading) {
        messageInput.focus();
      }
    }
  }

  function scrollToBottom() {
    if (messagesViewport) {
      messagesViewport.scrollTop = messagesViewport.scrollHeight;
    }
  }

  function formatTime(dateVal = new Date()) {
    const d = typeof dateVal === "string" ? new Date(dateVal) : dateVal;
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  // =========================================================================
  // 5. DOM Rendering for Messages
  // =========================================================================
  function appendUserMessage(text, timestamp, attachment) {
    if (!chatStream) return;
    const row = document.createElement("div");
    row.className = "message-row user";

    const bubble = document.createElement("div");
    bubble.className = "bubble";

    if (attachment && (attachment.filename || typeof attachment === "string")) {
      const attFilename = typeof attachment === "string" ? attachment : attachment.filename;
      const attSize = (typeof attachment === "object" && attachment.size_bytes) ? formatBytes(attachment.size_bytes) : "";

      const attPill = document.createElement("div");
      attPill.className = "user-attachment-pill";
      attPill.innerHTML = `
        <svg viewBox="0 0 24 24" width="14" height="14" stroke="currentColor" stroke-width="2" fill="none"><path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/></svg>
        <span class="user-attachment-name" title="${escapeHtml(attFilename)}">${escapeHtml(attFilename)}</span>
        ${attSize ? `<span class="user-attachment-size">${escapeHtml(attSize)}</span>` : ""}
      `;
      bubble.appendChild(attPill);
    }

    const textDiv = document.createElement("div");
    textDiv.className = "user-message-text";
    textDiv.textContent = text;
    bubble.appendChild(textDiv);

    const meta = document.createElement("div");
    meta.className = "bubble-meta";
    meta.innerHTML = `<span></span><span>${formatTime(timestamp)}</span>`;

    bubble.appendChild(meta);
    row.appendChild(bubble);
    chatStream.appendChild(row);
  }

  function appendLoadingIndicator() {
    if (!chatStream) return null;
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

  function appendAssistantMessage(data, timestamp) {
    if (!chatStream) return;
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
          <span class="bubble-badge ${badgeClass}">${escapeHtml(badgeLabel)}</span>
          <div class="bubble-actions">
            <button type="button" class="copy-btn" title="Copy response" aria-label="Copy response">
              <svg viewBox="0 0 24 24" width="13" height="13" stroke="currentColor" stroke-width="2" fill="none"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
            </button>
            <span class="bubble-time">${formatTime(timestamp)}</span>
          </div>
        </div>
      </div>
    `;

    // Copy action
    const copyBtn = row.querySelector(".copy-btn");
    if (copyBtn) {
      copyBtn.addEventListener("click", () => {
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(data.text).then(() => {
            copyBtn.innerHTML = "✓";
            setTimeout(() => {
              copyBtn.innerHTML = `
                <svg viewBox="0 0 24 24" width="13" height="13" stroke="currentColor" stroke-width="2" fill="none"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
              `;
            }, 1500);
          }).catch((err) => {
            console.warn("[LOKI UI] Clipboard copy failed:", err);
          });
        }
      });
    }

    chatStream.appendChild(row);
  }

  function appendConfirmationCard(confirmData, originalPrompt) {
    if (!chatStream) return;
    const row = document.createElement("div");
    row.className = "message-row assistant";

    const targetsStr = confirmData.targets && confirmData.targets.length
      ? confirmData.targets.join(", ")
      : "None specified";

    const safeDesc = escapeHtml(confirmData.description);
    const safeTargets = escapeHtml(targetsStr);
    const safeActionId = escapeHtml(confirmData.action_id);

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
            ${safeDesc}
          </div>
          <div class="confirm-targets">
            Targets: ${safeTargets}<br>
            Action ID: ${safeActionId}
          </div>
          <div class="confirm-btn-row">
            <button type="button" class="btn-approve" id="approveBtn_${safeActionId}">Approve Action</button>
            <button type="button" class="btn-cancel" id="cancelBtn_${safeActionId}">Cancel</button>
          </div>
        </div>
        <div class="bubble-meta">
          <span class="bubble-badge badge-deterministic">SECURITY GATEKEEPER</span>
          <span class="bubble-time">${formatTime()}</span>
        </div>
      </div>
    `;

    chatStream.appendChild(row);

    const approveBtn = row.querySelector(`#approveBtn_${safeActionId}`);
    const cancelBtn = row.querySelector(`#cancelBtn_${safeActionId}`);

    if (approveBtn) {
      approveBtn.addEventListener("click", async () => {
        approveBtn.disabled = true;
        if (cancelBtn) cancelBtn.disabled = true;
        approveBtn.textContent = "Executing...";

        try {
          const res = await fetch("/api/confirm", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              action_id: confirmData.action_id,
              original_message: originalPrompt,
              conversation_id: activeConversationId,
            }),
          });
          const result = await res.json();
          appendAssistantMessage(result);
          fetchConversationsListSilently();
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
    }

    if (cancelBtn) {
      cancelBtn.addEventListener("click", async () => {
        if (approveBtn) approveBtn.disabled = true;
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
  }

  // =========================================================================
  // 6. System Status Polling
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
    if (!statusDot || !statusPillText) return;

    if (!data.ollama_online) {
      statusDot.className = "status-dot offline";
      statusPillText.textContent = "Ollama: Offline";
      if (drawerOllamaBadge) {
        drawerOllamaBadge.textContent = "Offline";
        drawerOllamaBadge.className = "metric-badge offline";
      }
    } else if (data.model_available) {
      statusDot.className = "status-dot online";
      if (data.using_fallback) {
        statusPillText.textContent = `Ollama: Connected (${data.effective_model} [Fallback])`;
      } else {
        statusPillText.textContent = `Ollama: Connected (${data.active_model})`;
      }
      if (drawerOllamaBadge) {
        drawerOllamaBadge.textContent = "Online";
        drawerOllamaBadge.className = "metric-badge online";
      }
    } else {
      statusDot.className = "status-dot warning";
      statusPillText.textContent = `Ollama: Connected (Model Missing: ${data.active_model})`;
      if (drawerOllamaBadge) {
        drawerOllamaBadge.textContent = "Model Missing";
        drawerOllamaBadge.className = "metric-badge warning";
      }
    }

    if (data.ollama_host && drawerOllamaHost) drawerOllamaHost.textContent = data.ollama_host;
    if (data.ollama_version && drawerOllamaVersion) drawerOllamaVersion.textContent = data.ollama_version;

    if (data.active_model && drawerActiveModel) {
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
    if (data.hardware_tier && drawerHardwareTier) {
      drawerHardwareTier.textContent = data.hardware_tier.toUpperCase();
    }

    if (data.ram_used_gb !== undefined && data.ram_total_gb !== undefined && drawerRamText && drawerRamBar) {
      drawerRamText.textContent = `${data.ram_used_gb} GB / ${data.ram_total_gb} GB (${data.ram_percent}%)`;
      drawerRamBar.style.width = `${Math.min(data.ram_percent, 100)}%`;
    }

    if (data.cpu_percent !== undefined && drawerCpuText && drawerCpuBar) {
      drawerCpuText.textContent = `${data.cpu_percent}% across ${data.cpu_cores} cores`;
      drawerCpuBar.style.width = `${Math.min(data.cpu_percent, 100)}%`;
    }
  }

  // Initial setup: load conversations and telemetry
  loadConversations();
  fetchStatus();
  setInterval(fetchStatus, 8000);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initLokiApp);
} else {
  initLokiApp();
}
