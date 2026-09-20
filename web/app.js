// ==========================================================================
// ChatGPT Client Application Logic
// ==========================================================================

document.addEventListener("DOMContentLoaded", () => {
  const scrollContainer = document.getElementById("chatScrollContainer");
  const messagesWrapper = document.getElementById("messagesWrapper");
  const heroScreen = document.getElementById("heroScreen");
  const chatForm = document.getElementById("chatForm");
  const messageInput = document.getElementById("messageInput");
  const sendBtn = document.getElementById("sendBtn");
  const clearChatBtn = document.getElementById("clearChatBtn");
  const newChatBtn = document.getElementById("newChatBtn");
  const sidebarCollapseBtn = document.getElementById("sidebarCollapseBtn");
  const sidebarOpenBtn = document.getElementById("sidebarOpenBtn");
  const sidebar = document.getElementById("sidebar");
  const chipCards = document.querySelectorAll(".chip-card");

  let conversationHistory = [];
  let isGenerating = false;

  // Configure Marked markdown parser
  marked.setOptions({
    breaks: true,
    gfm: true,
    highlight: function (code, lang) {
      const language = hljs.getLanguage(lang) ? lang : "plaintext";
      return hljs.highlight(code, { language }).value;
    },
  });

  // Enable/Disable Send button on typing & auto-resize
  messageInput.addEventListener("input", () => {
    sendBtn.disabled = !messageInput.value.trim() || isGenerating;
    messageInput.style.height = "auto";
    messageInput.style.height = Math.min(messageInput.scrollHeight, 180) + "px";
  });

  // Keyboard shortcut: Enter to submit, Shift+Enter for newline
  messageInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!sendBtn.disabled) {
        chatForm.dispatchEvent(new Event("submit"));
      }
    }
  });

  // Sidebar Controls
  sidebarCollapseBtn.addEventListener("click", () => {
    sidebar.classList.add("collapsed");
  });

  sidebarOpenBtn.addEventListener("click", () => {
    sidebar.classList.toggle("collapsed");
  });

  // Suggestion Chip Cards
  chipCards.forEach((card) => {
    card.addEventListener("click", () => {
      const prompt = card.getAttribute("data-prompt");
      messageInput.value = prompt;
      sendBtn.disabled = false;
      chatForm.dispatchEvent(new Event("submit"));
    });
  });

  // New Chat & Clear Chat
  function resetConversation() {
    conversationHistory = [];
    messagesWrapper.innerHTML = "";
    heroScreen.style.display = "flex";
    messageInput.value = "";
    messageInput.style.height = "auto";
    sendBtn.disabled = true;
    messageInput.focus();
  }

  if (newChatBtn) newChatBtn.addEventListener("click", resetConversation);
  if (clearChatBtn) clearChatBtn.addEventListener("click", resetConversation);

  // Form Submission (Real-Time SSE Streaming)
  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = messageInput.value.trim();
    if (!text || isGenerating) return;

    // Hide hero screen
    heroScreen.style.display = "none";

    // Append User Message
    appendMessage("user", text);
    conversationHistory.push({ role: "user", content: text });

    messageInput.value = "";
    messageInput.style.height = "auto";
    setGeneratingState(true);

    // Create Assistant Row with Typing Cursor
    const assistantRow = appendMessage("assistant", "");
    const bodyEl = assistantRow.querySelector(".message-body");
    const cursor = document.createElement("span");
    cursor.className = "streaming-cursor";
    bodyEl.appendChild(cursor);

    let fullAssistantResponse = "";

    try {
      const response = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: conversationHistory,
        }),
      });

      if (!response.ok) {
        throw new Error(`Server returned error ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split("\n");

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const dataStr = line.replace("data: ", "").trim();
            if (dataStr === "[DONE]") break;
            try {
              const data = JSON.parse(dataStr);
              if (data.token) {
                fullAssistantResponse += data.token;
                renderMarkdownContent(bodyEl, fullAssistantResponse, cursor);
                scrollContainer.scrollTop = scrollContainer.scrollHeight;
              }
            } catch (err) {
              fullAssistantResponse += dataStr;
              renderMarkdownContent(bodyEl, fullAssistantResponse, cursor);
            }
          }
        }
      }

      // Final render without cursor
      if (cursor.parentNode) cursor.remove();
      renderMarkdownContent(bodyEl, fullAssistantResponse);
      conversationHistory.push({ role: "assistant", content: fullAssistantResponse });
    } catch (err) {
      if (cursor.parentNode) cursor.remove();
      bodyEl.innerHTML = `<span style="color: #ef4444;">[Error: ${err.message}]</span>`;
    } finally {
      setGeneratingState(false);
      scrollContainer.scrollTop = scrollContainer.scrollHeight;
    }
  });

  function appendMessage(role, text) {
    const row = document.createElement("div");
    row.className = `message-row ${role}`;

    const body = document.createElement("div");
    body.className = "message-body";
    if (text) {
      if (role === "user") {
        body.textContent = text;
      } else {
        renderMarkdownContent(body, text);
      }
    }

    row.appendChild(body);
    messagesWrapper.appendChild(row);
    scrollContainer.scrollTop = scrollContainer.scrollHeight;
    return row;
  }

  function renderMarkdownContent(element, markdownText, cursorElement = null) {
    element.innerHTML = marked.parse(markdownText);
    if (cursorElement) {
      element.appendChild(cursorElement);
    }
    // Apply syntax highlighting & wrap code blocks
    element.querySelectorAll("pre code").forEach((block) => {
      hljs.highlightElement(block);
      wrapCodeBlock(block.parentElement);
    });
  }

  function wrapCodeBlock(preElement) {
    if (preElement.parentElement.classList.contains("code-container")) return;

    const wrapper = document.createElement("div");
    wrapper.className = "code-container";

    const header = document.createElement("div");
    header.className = "code-header-bar";

    const lang = preElement.querySelector("code").className.replace("hljs language-", "") || "code";
    const langSpan = document.createElement("span");
    langSpan.textContent = lang;

    const copyBtn = document.createElement("button");
    copyBtn.className = "copy-button";
    copyBtn.innerHTML = `
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
      </svg>
      <span>Copy code</span>
    `;

    copyBtn.addEventListener("click", () => {
      const code = preElement.querySelector("code").innerText;
      navigator.clipboard.writeText(code);
      copyBtn.innerHTML = `<span>✓ Copied!</span>`;
      setTimeout(() => {
        copyBtn.innerHTML = `
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
          </svg>
          <span>Copy code</span>
        `;
      }, 2000);
    });

    header.appendChild(langSpan);
    header.appendChild(copyBtn);

    preElement.parentNode.insertBefore(wrapper, preElement);
    wrapper.appendChild(header);
    wrapper.appendChild(preElement);
  }

  function setGeneratingState(generating) {
    isGenerating = generating;
    sendBtn.disabled = generating || !messageInput.value.trim();
    messageInput.disabled = generating;
    if (!generating) messageInput.focus();
  }
});
