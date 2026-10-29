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
  let currentAttachmentPayload = null;
  let activeTab = "chat";
  let libraryDocuments = [];
  let storedMemories = [];
  let editingMemoryId = null;

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
  if (navSettingsBtn) navSettingsBtn.addEventListener("click", () => openSettingsModal());
  if (refreshStatusBtn) refreshStatusBtn.addEventListener("click", fetchStatus);

  // New Chat and Clear Actions
  if (newChatBtn) {
    newChatBtn.addEventListener("click", (e) => {
      e.preventDefault();
      switchTab("chat");
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
    currentAttachmentPayload = null;
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
      if ((text || currentAttachmentFile || currentAttachmentPayload) && !isInferring) {
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
        switchTab("chat");
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
    let uploadedPayload = currentAttachmentPayload;
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

    // Stage attachment file to backend if attached
    if (fileToUpload && !uploadedPayload) {
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


  // =========================================================================
  // VIEW PANELS & TAB CONTROLLER
  // =========================================================================
  const chatView = document.getElementById("chatView");
  const exploreView = document.getElementById("exploreView");
  const libraryView = document.getElementById("libraryView");
  const navTabs = document.querySelectorAll(".nav-menu .nav-item[data-tab]");

  function switchTab(tabName) {
    activeTab = tabName;
    navTabs.forEach((btn) => {
      if (btn.dataset.tab === tabName) {
        btn.classList.add("active");
      } else {
        btn.classList.remove("active");
      }
    });

    if (chatView) chatView.style.display = (tabName === "chat") ? "flex" : "none";
    if (exploreView) exploreView.style.display = (tabName === "explore") ? "flex" : "none";
    if (libraryView) libraryView.style.display = (tabName === "library") ? "flex" : "none";

    if (chatView) {
      if (tabName === "chat") chatView.classList.add("active");
      else chatView.classList.remove("active");
    }
    if (exploreView) {
      if (tabName === "explore") exploreView.classList.add("active");
      else exploreView.classList.remove("active");
    }
    if (libraryView) {
      if (tabName === "library") libraryView.classList.add("active");
      else libraryView.classList.remove("active");
    }

    if (tabName === "library") {
      loadLibraryDocuments();
    } else if (tabName === "explore") {
      renderExploreCapabilities();
    } else if (tabName === "chat") {
      if (messageInput) messageInput.focus();
    }
  }

  navTabs.forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      const tab = btn.dataset.tab;
      if (tab) switchTab(tab);
    });
  });

  // =========================================================================
  // EXPLORE CAPABILITIES CONTROLLER
  // =========================================================================
  const AVAILABLE_CAPABILITIES = [
    {
      id: "ai_chat",
      icon: "🧠",
      title: "Local AI Chat & Reasoning",
      tag: "Ready",
      desc: "Conversational intelligence running 100% locally via Ollama with complete privacy and zero data leakage.",
      actionLabel: "Try Chat Prompt",
      actionType: "prompt",
      actionValue: "Explain the architectural advantages of local offline AI.",
    },
    {
      id: "telemetry",
      icon: "⚡",
      title: "Local System Telemetry",
      tag: "Ready",
      desc: "Deterministic zero-LLM monitoring of CPU load, RAM allocation, battery percentage, and disk storage.",
      actionLabel: "Check Battery",
      actionType: "prompt",
      actionValue: "what is my battery percentage",
    },
    {
      id: "file_search",
      icon: "🔍",
      title: "Safe File Discovery",
      tag: "Ready",
      desc: "Instant local search for files and folders strictly constrained within user-approved search roots.",
      actionLabel: "Search Files",
      actionType: "prompt",
      actionValue: "find file report.pdf",
    },
    {
      id: "file_create",
      icon: "📝",
      title: "Safe File & Folder Creation",
      tag: "Ready",
      desc: "Create new files and directories deterministically with strict boundary validation and safety checks.",
      actionLabel: "Create File",
      actionType: "prompt",
      actionValue: "create file notes.txt with hello world",
    },
    {
      id: "file_delete",
      icon: "🛡️",
      title: "Gated Destructive Operations",
      tag: "Ready",
      desc: "Destructive operations are fail-closed and require an explicit single-use cryptographic confirmation token.",
      actionLabel: "Learn Safety Flow",
      actionType: "prompt",
      actionValue: "How does Loki protect my files with confirmation tokens?",
    },
    {
      id: "doc_intel",
      icon: "📄",
      title: "Document Intelligence",
      tag: "Ready",
      desc: "Offline text extraction and grounded summarization for PDF, TXT, CSV, and Markdown files.",
      actionLabel: "Open Library",
      actionType: "library",
    },
    {
      id: "memory_store",
      icon: "💾",
      title: "Persistent User Memory",
      tag: "Ready",
      desc: "Deterministic SQLite memory store remembering user preferences and personal facts across sessions.",
      actionLabel: "Manage Memories",
      actionType: "settings_memory",
    },
    {
      id: "memory_aware",
      icon: "🎯",
      title: "Memory-Aware Responses",
      tag: "Ready",
      desc: "Contextually personalizes LLM responses by injecting relevant saved facts into system prompts.",
      actionLabel: "Query Memory",
      actionType: "prompt",
      actionValue: "What do you remember about me?",
    },
    {
      id: "app_launcher",
      icon: "🚀",
      title: "Application Launcher",
      tag: "Ready",
      desc: "Safely launch allowed desktop applications like Notepad, Calculator, or Terminal deterministically.",
      actionLabel: "Launch Notepad",
      actionType: "prompt",
      actionValue: "launch notepad",
    },
  ];

  const PLANNED_CAPABILITIES = [
    {
      id: "voice_in",
      icon: "🎤",
      title: "Offline Voice Input",
      tag: "Phase 6",
      desc: "Local, private speech-to-text transcription powered by OpenAI Whisper running entirely on-device.",
    },
    {
      id: "voice_out",
      icon: "🔊",
      title: "Neural Voice Synthesis",
      tag: "Phase 7",
      desc: "High-performance, offline neural speech synthesis using Piper TTS for natural voice interaction.",
    },
    {
      id: "scheduler",
      icon: "⏱️",
      title: "Automated Scheduled Tasks",
      tag: "Roadmap",
      desc: "Background recurring schedules, reminders, and automated local workflow triggers.",
    },
    {
      id: "vision",
      icon: "👁️",
      title: "Vision & Multimodal QA",
      tag: "Roadmap",
      desc: "Local image comprehension, OCR extraction, and visual document reasoning.",
    },
    {
      id: "model_switcher",
      icon: "🔄",
      title: "Dynamic Model Switcher",
      tag: "Roadmap",
      desc: "Hot-switching between installed local Ollama models tailored for specialized tasks.",
    },
  ];

  const availableGrid = document.getElementById("availableCapabilitiesGrid");
  const plannedGrid = document.getElementById("plannedCapabilitiesGrid");
  const exploreFilterInput = document.getElementById("exploreFilterInput");

  function renderExploreCapabilities(filter = "") {
    const q = filter.trim().toLowerCase();

    if (availableGrid) {
      availableGrid.innerHTML = "";
      const filteredAvail = q
        ? AVAILABLE_CAPABILITIES.filter((c) => c.title.toLowerCase().includes(q) || c.desc.toLowerCase().includes(q))
        : AVAILABLE_CAPABILITIES;

      filteredAvail.forEach((cap) => {
        const card = document.createElement("div");
        card.className = "capability-card";
        card.innerHTML = `
          <div class="cap-header">
            <div class="cap-title-group">
              <span class="cap-icon">${cap.icon}</span>
              <span class="cap-title">${escapeHtml(cap.title)}</span>
            </div>
            <span class="cap-tag ready">${cap.tag}</span>
          </div>
          <p class="cap-desc">${escapeHtml(cap.desc)}</p>
          <div class="cap-footer">
            <button type="button" class="cap-action-btn" data-cap-id="${cap.id}">
              <span>${escapeHtml(cap.actionLabel)}</span>
              <svg viewBox="0 0 24 24" width="12" height="12" stroke="currentColor" stroke-width="2" fill="none"><polyline points="9 18 15 12 9 6"/></svg>
            </button>
          </div>
        `;

        const btn = card.querySelector(".cap-action-btn");
        if (btn) {
          btn.addEventListener("click", () => {
            if (cap.actionType === "prompt") {
              switchTab("chat");
              if (messageInput) {
                messageInput.value = cap.actionValue;
                submitMessage(cap.actionValue);
              }
            } else if (cap.actionType === "library") {
              switchTab("library");
            } else if (cap.actionType === "settings_memory") {
              openSettingsModal();
            }
          });
        }
        availableGrid.appendChild(card);
      });
    }

    if (plannedGrid) {
      plannedGrid.innerHTML = "";
      const filteredPlanned = q
        ? PLANNED_CAPABILITIES.filter((c) => c.title.toLowerCase().includes(q) || c.desc.toLowerCase().includes(q))
        : PLANNED_CAPABILITIES;

      filteredPlanned.forEach((cap) => {
        const card = document.createElement("div");
        card.className = "capability-card planned-card";
        card.innerHTML = `
          <div class="cap-header">
            <div class="cap-title-group">
              <span class="cap-icon">${cap.icon}</span>
              <span class="cap-title">${escapeHtml(cap.title)}</span>
            </div>
            <span class="cap-tag planned">${cap.tag}</span>
          </div>
          <p class="cap-desc">${escapeHtml(cap.desc)}</p>
        `;
        plannedGrid.appendChild(card);
      });
    }
  }

  if (exploreFilterInput) {
    exploreFilterInput.addEventListener("input", (e) => {
      renderExploreCapabilities(e.target.value);
    });
  }

  // =========================================================================
  // DOCUMENT LIBRARY CONTROLLER
  // =========================================================================
  const libraryContainer = document.getElementById("libraryContainer");
  const libraryDocCount = document.getElementById("libraryDocCount");
  const libraryFileInput = document.getElementById("libraryFileInput");
  const libraryUploadBtn = document.getElementById("libraryUploadBtn");
  const refreshLibraryBtn = document.getElementById("refreshLibraryBtn");
  const librarySearchInput = document.getElementById("librarySearchInput");

  async function loadLibraryDocuments() {
    try {
      const res = await fetch("/api/documents");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      libraryDocuments = data.documents || [];
      renderLibraryDocuments(librarySearchInput ? librarySearchInput.value : "");
    } catch (e) {
      console.warn("[LOKI UI] Failed to load documents library:", e);
      if (libraryContainer) {
        libraryContainer.innerHTML = `
          <div class="library-empty">
            <span class="library-empty-title">Error loading library</span>
            <span class="library-empty-desc">${escapeHtml(e.message)}</span>
          </div>
        `;
      }
    }
  }

  function renderLibraryDocuments(filter = "") {
    if (!libraryContainer) return;
    const q = filter.trim().toLowerCase();
    const filtered = q
      ? libraryDocuments.filter((d) => (d.filename || "").toLowerCase().includes(q))
      : libraryDocuments;

    if (libraryDocCount) {
      libraryDocCount.textContent = `${filtered.length} document${filtered.length === 1 ? "" : "s"}`;
    }

    if (filtered.length === 0) {
      libraryContainer.innerHTML = `
        <div class="library-empty">
          <div class="library-empty-icon">
            <svg viewBox="0 0 24 24" width="28" height="28" stroke="currentColor" stroke-width="2" fill="none"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
          </div>
          <span class="library-empty-title">${q ? "No matching documents" : "No documents in library"}</span>
          <span class="library-empty-desc">
            ${q ? "Try clearing your search query." : "Upload a .txt, .pdf, .csv, or .md document to analyze it locally with LOKI."}
          </span>
        </div>
      `;
      return;
    }

    const grid = document.createElement("div");
    grid.className = "documents-grid";

    filtered.forEach((doc) => {
      const ext = (doc.extension || "").replace(".", "").toLowerCase();
      const card = document.createElement("div");
      card.className = "document-card";
      card.dataset.docId = doc.id;

      card.innerHTML = `
        <div class="doc-top">
          <div class="doc-icon-wrapper ${ext}">${ext || "doc"}</div>
          <div class="doc-info">
            <div class="doc-name" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</div>
            <div class="doc-meta">${formatBytes(doc.size_bytes)} • ${formatRelativeTime(doc.modified_at)}</div>
          </div>
        </div>
        <div class="doc-actions">
          <button type="button" class="doc-btn analyze-btn" title="Analyze document in chat">
            <svg viewBox="0 0 24 24" width="12" height="12" stroke="currentColor" stroke-width="2" fill="none"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
            <span>Analyze in Chat</span>
          </button>
          <button type="button" class="doc-btn delete-btn" title="Delete document">
            <svg viewBox="0 0 24 24" width="12" height="12" stroke="currentColor" stroke-width="2" fill="none"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
            <span>Delete</span>
          </button>
        </div>
      `;

      const analyzeBtn = card.querySelector(".analyze-btn");
      if (analyzeBtn) {
        analyzeBtn.addEventListener("click", () => {
          stageLibraryDocumentForChat(doc);
        });
      }

      const deleteBtn = card.querySelector(".delete-btn");
      if (deleteBtn) {
        deleteBtn.addEventListener("click", () => {
          deleteLibraryDocument(doc.id, doc.filename);
        });
      }

      grid.appendChild(card);
    });

    libraryContainer.innerHTML = "";
    libraryContainer.appendChild(grid);
  }

  function stageLibraryDocumentForChat(doc) {
    switchTab("chat");
    currentAttachmentPayload = {
      filename: doc.filename,
      path: doc.path,
      size_bytes: doc.size_bytes,
    };
    currentAttachmentFile = null;

    if (attachmentName) attachmentName.textContent = doc.filename;
    if (attachmentSize) attachmentSize.textContent = formatBytes(doc.size_bytes);
    if (attachmentPreview) attachmentPreview.style.display = "flex";

    if (messageInput) {
      messageInput.value = `Summarize ${doc.filename}.`;
      messageInput.focus();
    }
  }

  async function deleteLibraryDocument(docId, filename) {
    const confirmed = window.confirm(`Delete document "${filename}" from local library? This cannot be undone.`);
    if (!confirmed) return;

    try {
      const res = await fetch(`/api/documents/${encodeURIComponent(docId)}`, {
        method: "DELETE",
      });
      if (res.ok) {
        await loadLibraryDocuments();
      } else {
        const err = await res.json().catch(() => ({}));
        alert(`Failed to delete document: ${err.detail || res.statusText}`);
      }
    } catch (e) {
      alert(`Error deleting document: ${e.message}`);
    }
  }

  if (libraryUploadBtn && libraryFileInput) {
    libraryUploadBtn.addEventListener("click", () => {
      libraryFileInput.click();
    });

    libraryFileInput.addEventListener("change", async () => {
      if (!libraryFileInput.files || libraryFileInput.files.length === 0) return;
      const file = libraryFileInput.files[0];
      const formData = new FormData();
      formData.append("file", file);

      try {
        const res = await fetch("/api/upload", {
          method: "POST",
          body: formData,
        });
        if (res.ok) {
          libraryFileInput.value = "";
          await loadLibraryDocuments();
        } else {
          const err = await res.json().catch(() => ({}));
          alert(`Upload failed: ${err.detail || res.statusText}`);
        }
      } catch (e) {
        alert(`Upload failed: ${e.message}`);
      }
    });
  }

  if (refreshLibraryBtn) {
    refreshLibraryBtn.addEventListener("click", () => {
      loadLibraryDocuments();
    });
  }

  if (librarySearchInput) {
    librarySearchInput.addEventListener("input", (e) => {
      renderLibraryDocuments(e.target.value);
    });
  }

  // =========================================================================
  // SETTINGS & MEMORY MANAGEMENT CONTROLLER
  // =========================================================================
  const settingsModal = document.getElementById("settingsModal");
  const closeSettingsModalBtn = document.getElementById("closeSettingsModalBtn");
  const settingsTabMemory = document.getElementById("settingsTabMemory");
  const settingsTabDiagnostics = document.getElementById("settingsTabDiagnostics");
  const settingsMemoryPanel = document.getElementById("settingsMemoryPanel");
  const settingsDiagnosticsPanel = document.getElementById("settingsDiagnosticsPanel");

  const memoryEnabledToggle = document.getElementById("memoryEnabledToggle");
  const memoryCountBadge = document.getElementById("memoryCountBadge");
  const memorySearchInput = document.getElementById("memorySearchInput");
  const showAddMemoryBtn = document.getElementById("showAddMemoryBtn");
  const memoryFormCard = document.getElementById("memoryFormCard");
  const memoryFormTitle = document.getElementById("memoryFormTitle");
  const memoryCategorySelect = document.getElementById("memoryCategorySelect");
  const memoryKeyInput = document.getElementById("memoryKeyInput");
  const memoryContentInput = document.getElementById("memoryContentInput");
  const cancelMemoryFormBtn = document.getElementById("cancelMemoryFormBtn");
  const saveMemoryFormBtn = document.getElementById("saveMemoryFormBtn");
  const memoryFormError = document.getElementById("memoryFormError");
  const memoriesList = document.getElementById("memoriesList");
  const clearAllMemoriesBtn = document.getElementById("clearAllMemoriesBtn");

  const settingsOllamaStatus = document.getElementById("settingsOllamaStatus");
  const settingsActiveModel = document.getElementById("settingsActiveModel");
  const settingsRamText = document.getElementById("settingsRamText");
  const settingsCpuText = document.getElementById("settingsCpuText");

  async function openSettingsModal() {
    if (settingsModal) {
      settingsModal.style.display = "flex";
      await loadSettings();
      await loadMemories();
    }
  }

  function closeSettingsModal() {
    if (settingsModal) {
      settingsModal.style.display = "none";
      if (memoryFormCard) memoryFormCard.style.display = "none";
      editingMemoryId = null;
    }
  }

  if (closeSettingsModalBtn) closeSettingsModalBtn.addEventListener("click", closeSettingsModal);
  if (settingsModal) {
    settingsModal.addEventListener("click", (e) => {
      if (e.target === settingsModal) closeSettingsModal();
    });
  }

  if (settingsTabMemory && settingsTabDiagnostics) {
    settingsTabMemory.addEventListener("click", () => {
      settingsTabMemory.classList.add("active");
      settingsTabDiagnostics.classList.remove("active");
      if (settingsMemoryPanel) settingsMemoryPanel.style.display = "flex";
      if (settingsDiagnosticsPanel) settingsDiagnosticsPanel.style.display = "none";
    });

    settingsTabDiagnostics.addEventListener("click", () => {
      settingsTabDiagnostics.classList.add("active");
      settingsTabMemory.classList.remove("active");
      if (settingsDiagnosticsPanel) settingsDiagnosticsPanel.style.display = "flex";
      if (settingsMemoryPanel) settingsMemoryPanel.style.display = "none";
    });
  }

  async function loadSettings() {
    try {
      const res = await fetch("/api/settings");
      if (res.ok) {
        const data = await res.json();
        if (memoryEnabledToggle) {
          memoryEnabledToggle.checked = !!data.memory_enabled;
        }
        if (settingsActiveModel) settingsActiveModel.textContent = data.active_model || "—";
      }
    } catch (e) {
      console.warn("Failed to load settings:", e);
    }

    try {
      const sRes = await fetch("/api/status");
      if (sRes.ok) {
        const sData = await sRes.json();
        if (settingsOllamaStatus) {
          settingsOllamaStatus.textContent = sData.ollama_online ? `Online (${sData.ollama_version || "connected"})` : "Offline";
        }
        if (settingsRamText && sData.ram_used_gb !== undefined) {
          settingsRamText.textContent = `${sData.ram_used_gb} GB / ${sData.ram_total_gb} GB (${sData.ram_percent}%)`;
        }
        if (settingsCpuText && sData.cpu_percent !== undefined) {
          settingsCpuText.textContent = `${sData.cpu_percent}% across ${sData.cpu_cores} cores`;
        }
      }
    } catch {}
  }

  if (memoryEnabledToggle) {
    memoryEnabledToggle.addEventListener("change", async () => {
      const enabled = memoryEnabledToggle.checked;
      try {
        await fetch("/api/settings", {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ memory_enabled: enabled }),
        });
      } catch (e) {
        console.warn("Failed to update memory setting:", e);
      }
    });
  }

  async function loadMemories(query = "") {
    try {
      const url = query ? `/api/memories?query=${encodeURIComponent(query)}` : "/api/memories";
      const res = await fetch(url);
      if (res.ok) {
        const data = await res.json();
        storedMemories = data.memories || [];
        renderMemoriesList(storedMemories);
      }
    } catch (e) {
      console.warn("Failed to fetch memories:", e);
    }
  }

  function renderMemoriesList(memories) {
    if (!memoriesList) return;
    if (memoryCountBadge) {
      memoryCountBadge.textContent = `${memories.length} memor${memories.length === 1 ? "y" : "ies"}`;
    }

    if (memories.length === 0) {
      memoriesList.innerHTML = `
        <div class="memory-empty-state">
          No memories stored yet. You can tell LOKI "Remember that..." in chat, or click "+ Add Memory" above.
        </div>
      `;
      return;
    }

    memoriesList.innerHTML = "";
    memories.forEach((mem) => {
      const card = document.createElement("div");
      card.className = "memory-card";
      card.dataset.memoryId = mem.id;

      card.innerHTML = `
        <div class="mem-header">
          <div class="mem-tags">
            <span class="mem-category-badge">${escapeHtml(mem.category)}</span>
            ${mem.key ? `<span class="mem-key">${escapeHtml(mem.key)}</span>` : ""}
          </div>
          <div class="mem-actions">
            <button type="button" class="mem-btn edit" title="Edit memory">
              <svg viewBox="0 0 24 24" width="12" height="12" stroke="currentColor" stroke-width="2" fill="none"><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/></svg>
            </button>
            <button type="button" class="mem-btn delete" title="Delete memory">
              <svg viewBox="0 0 24 24" width="12" height="12" stroke="currentColor" stroke-width="2" fill="none"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
            </button>
          </div>
        </div>
        <div class="mem-content">${escapeHtml(mem.content)}</div>
        <div class="mem-time">${formatRelativeTime(mem.updated_at)}</div>
      `;

      const editBtn = card.querySelector(".mem-btn.edit");
      if (editBtn) {
        editBtn.addEventListener("click", () => {
          startEditMemory(mem);
        });
      }

      const delBtn = card.querySelector(".mem-btn.delete");
      if (delBtn) {
        delBtn.addEventListener("click", () => {
          deleteSingleMemory(mem.id, mem.content);
        });
      }

      memoriesList.appendChild(card);
    });
  }

  function startEditMemory(mem) {
    editingMemoryId = mem.id;
    if (memoryFormTitle) memoryFormTitle.textContent = "Edit Memory";
    if (memoryCategorySelect) memoryCategorySelect.value = mem.category || "preference";
    if (memoryKeyInput) memoryKeyInput.value = mem.key || "";
    if (memoryContentInput) memoryContentInput.value = mem.content || "";
    if (memoryFormError) memoryFormError.style.display = "none";
    if (memoryFormCard) memoryFormCard.style.display = "flex";
    if (memoryContentInput) memoryContentInput.focus();
  }

  if (showAddMemoryBtn) {
    showAddMemoryBtn.addEventListener("click", () => {
      editingMemoryId = null;
      if (memoryFormTitle) memoryFormTitle.textContent = "Add New Memory";
      if (memoryCategorySelect) memoryCategorySelect.value = "preference";
      if (memoryKeyInput) memoryKeyInput.value = "";
      if (memoryContentInput) memoryContentInput.value = "";
      if (memoryFormError) memoryFormError.style.display = "none";
      if (memoryFormCard) {
        memoryFormCard.style.display = memoryFormCard.style.display === "none" ? "flex" : "none";
      }
      if (memoryContentInput) memoryContentInput.focus();
    });
  }

  if (cancelMemoryFormBtn && memoryFormCard) {
    cancelMemoryFormBtn.addEventListener("click", () => {
      memoryFormCard.style.display = "none";
      editingMemoryId = null;
    });
  }

  if (saveMemoryFormBtn) {
    saveMemoryFormBtn.addEventListener("click", async () => {
      const content = memoryContentInput ? memoryContentInput.value.trim() : "";
      const category = memoryCategorySelect ? memoryCategorySelect.value : "preference";
      const key = memoryKeyInput ? memoryKeyInput.value.trim() || null : null;

      if (!content) {
        if (memoryFormError) {
          memoryFormError.textContent = "Memory content cannot be empty.";
          memoryFormError.style.display = "block";
        }
        return;
      }

      try {
        let res;
        if (editingMemoryId) {
          res = await fetch(`/api/memories/${editingMemoryId}`, {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ content, category, key }),
          });
        } else {
          res = await fetch("/api/memories", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ content, category, key }),
          });
        }

        if (res.ok) {
          if (memoryFormCard) memoryFormCard.style.display = "none";
          editingMemoryId = null;
          await loadMemories(memorySearchInput ? memorySearchInput.value : "");
        } else {
          const errData = await res.json().catch(() => ({}));
          if (memoryFormError) {
            memoryFormError.textContent = errData.detail || "Failed to save memory.";
            memoryFormError.style.display = "block";
          }
        }
      } catch (err) {
        if (memoryFormError) {
          memoryFormError.textContent = err.message || "Network error saving memory.";
          memoryFormError.style.display = "block";
        }
      }
    });
  }

  async function deleteSingleMemory(memoryId, snippet) {
    const preview = snippet.length > 40 ? snippet.slice(0, 40) + "..." : snippet;
    const confirmed = window.confirm(`Forget memory: "${preview}"?`);
    if (!confirmed) return;

    try {
      const res = await fetch(`/api/memories/${memoryId}`, { method: "DELETE" });
      if (res.ok) {
        await loadMemories(memorySearchInput ? memorySearchInput.value : "");
      } else {
        alert("Failed to delete memory.");
      }
    } catch (e) {
      alert(`Error deleting memory: ${e.message}`);
    }
  }

  if (clearAllMemoriesBtn) {
    clearAllMemoriesBtn.addEventListener("click", async () => {
      const confirmed = window.confirm(
        "Are you sure you want to forget ALL stored memories? This cannot be undone."
      );
      if (!confirmed) return;

      try {
        const res = await fetch("/api/memories/clear", { method: "POST" });
        if (res.ok) {
          await loadMemories();
        } else {
          alert("Failed to clear memories.");
        }
      } catch (e) {
        alert(`Error clearing memories: ${e.message}`);
      }
    });
  }

  if (memorySearchInput) {
    memorySearchInput.addEventListener("input", (e) => {
      loadMemories(e.target.value);
    });
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
