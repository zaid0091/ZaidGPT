// ==========================================================================
// ZaidGPT Client Application Logic (SSE Streaming + Markdown Highlighting)
// ==========================================================================

document.addEventListener("DOMContentLoaded", () => {
  const chatMessages = document.getElementById("chatMessages");
  const welcomeScreen = document.getElementById("welcomeScreen");
  const chatForm = document.getElementById("chatForm");
  const messageInput = document.getElementById("messageInput");
  const sendBtn = document.getElementById("sendBtn");
  const clearChatBtn = document.getElementById("clearChatBtn");
  const toggleSidebarBtn = document.getElementById("toggleSidebarBtn");
  const sidebar = document.getElementById("sidebar");
  const engineSelect = document.getElementById("engineSelect");
  const activeEngineName = document.getElementById("activeEngineName");
  const suggestionCards = document.querySelectorAll(".suggestion-card");

  let conversationHistory = [];
  let isGenerating = false;

  // Configure Marked renderer for code blocks
  marked.setOptions({
    breaks: true,
    gfm: true,
    highlight: function (code, lang) {
      const language = hljs.getLanguage(lang) ? lang : "plaintext";
      return hljs.highlight(code, { language }).value;
    },
  });

  // Auto-resize textarea
  messageInput.addEventListener("input", () => {
    messageInput.style.height = "auto";
    messageInput.style.height = Math.min(messageInput.scrollHeight, 160) + "px";
  });

  // Keyboard shortcut: Enter to send, Shift+Enter for newline
  messageInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      chatForm.dispatchEvent(new Event("submit"));
    }
  });

  // Toggle Sidebar
  toggleSidebarBtn.addEventListener("click", () => {
    sidebar.classList.toggle("collapsed");
  });

  // Engine Switcher
  engineSelect.addEventListener("change", (e) => {
    const names = {
      instruct: "ZaidGPT Instruct Engine (SmolLM2)",
      finetuned: "ZaidGPT Fine-Tuned (GPT-2)",
      scratch: "ZaidGPT From-Scratch (Transformer+RAG)",
    };
    activeEngineName.textContent = names[e.target.value] || "ZaidGPT AI Engine";
  });

  // Suggestion Cards
  suggestionCards.forEach((card) => {
    card.addEventListener("click", () => {
      const prompt = card.getAttribute("data-prompt");
      messageInput.value = prompt;
      chatForm.dispatchEvent(new Event("submit"));
    });
  });

  // Clear Chat
  clearChatBtn.addEventListener("click", () => {
    if (confirm("Clear current conversation?")) {
      conversationHistory = [];
      chatMessages.innerHTML = "";
      chatMessages.appendChild(welcomeScreen);
      welcomeScreen.style.display = "block";
    }
  });

  // Form Submit (Streaming Chat)
  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = messageInput.value.trim();
    if (!text || isGenerating) return;

    // Hide welcome screen
    if (welcomeScreen && welcomeScreen.style.display !== "none") {
      welcomeScreen.style.display = "none";
    }

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
    cursor.className = "cursor-blink";
    bodyEl.appendChild(cursor);

    let fullAssistantResponse = "";
    const selectedEngine = engineSelect.value;

    try {
      const response = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: conversationHistory,
          engine: selectedEngine,
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
                chatMessages.scrollTop = chatMessages.scrollHeight;
              }
            } catch (err) {
              // Plain text fallback
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
      chatMessages.scrollTop = chatMessages.scrollHeight;
    }
  });

  function appendMessage(role, text) {
    const row = document.createElement("div");
    row.className = `message-row ${role}`;

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.innerHTML = role === "user" ? "👤" : "⚡";

    const body = document.createElement("div");
    body.className = "message-body";
    if (text) {
      renderMarkdownContent(body, text);
    }

    row.appendChild(avatar);
    row.appendChild(body);
    chatMessages.appendChild(row);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return row;
  }

  function renderMarkdownContent(element, markdownText, cursorElement = null) {
    element.innerHTML = marked.parse(markdownText);
    if (cursorElement) {
      element.appendChild(cursorElement);
    }
    // Apply syntax highlighting & add copy buttons
    element.querySelectorAll("pre code").forEach((block) => {
      hljs.highlightElement(block);
      wrapCodeBlock(block.parentElement);
    });
  }

  function wrapCodeBlock(preElement) {
    if (preElement.parentElement.classList.contains("code-block-wrapper")) return;

    const wrapper = document.createElement("div");
    wrapper.className = "code-block-wrapper";

    const header = document.createElement("div");
    header.className = "code-header";

    const lang = preElement.querySelector("code").className.replace("hljs language-", "") || "code";
    const langSpan = document.createElement("span");
    langSpan.textContent = lang;

    const copyBtn = document.createElement("button");
    copyBtn.className = "copy-code-btn";
    copyBtn.innerHTML = `
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
      </svg>
      <span>Copy</span>
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
          <span>Copy</span>
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
    sendBtn.disabled = generating;
    messageInput.disabled = generating;
    if (!generating) messageInput.focus();
  }
});
