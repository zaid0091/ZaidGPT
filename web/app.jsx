const { useState, useEffect, useRef } = React;

// Configure Marked options
marked.setOptions({
  breaks: true,
  gfm: true,
  highlight: function (code, lang) {
    const language = hljs.getLanguage(lang) ? lang : "plaintext";
    return hljs.highlight(code, { language }).value;
  },
});

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [currentStreamingText, setCurrentStreamingText] = useState("");
  
  const scrollRef = useRef(null);
  const textareaRef = useRef(null);

  // Auto-scroll on new message or stream chunk
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, currentStreamingText]);

  // Adjust textarea height
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 180) + "px";
    }
  }, [input]);

  const handleSend = async (textToSend) => {
    const text = (textToSend || input).trim();
    if (!text || isGenerating) return;

    const userMessage = { role: "user", content: text };
    const updatedHistory = [...messages, userMessage];

    setMessages(updatedHistory);
    setInput("");
    setIsGenerating(true);
    setCurrentStreamingText("");

    try {
      const response = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: updatedHistory }),
      });

      if (!response.ok) {
        throw new Error(`Server returned error ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let fullAssistantText = "";

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
                fullAssistantText += data.token;
                setCurrentStreamingText(fullAssistantText);
              }
            } catch (err) {
              fullAssistantText += dataStr;
              setCurrentStreamingText(fullAssistantText);
            }
          }
        }
      }

      setMessages((prev) => [...prev, { role: "assistant", content: fullAssistantText }]);
      setCurrentStreamingText("");
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `[Error: ${err.message}]` },
      ]);
      setCurrentStreamingText("");
    } finally {
      setIsGenerating(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleNewChat = () => {
    setMessages([]);
    setCurrentStreamingText("");
    setInput("");
    setIsGenerating(false);
    if (textareaRef.current) textareaRef.current.focus();
  };

  return (
    <div className="chatgpt-layout">
      {/* Collapsible Sidebar */}
      <aside className={`sidebar ${sidebarCollapsed ? "collapsed" : ""}`}>
        <div className="sidebar-top">
          <button
            className="icon-btn-ghost"
            onClick={() => setSidebarCollapsed(true)}
            title="Close sidebar"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
              <line x1="9" y1="3" x2="9" y2="21"></line>
            </svg>
          </button>
          <button
            className="new-chat-icon-btn"
            onClick={handleNewChat}
            title="New chat"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M12 20h9"></path>
              <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"></path>
            </svg>
          </button>
        </div>

        <div className="sidebar-content">
          <div className="history-group-title">Today</div>
          <div className="history-list">
            <div className="history-item active" onClick={handleNewChat}>
              <span className="history-item-text">
                {messages.length > 0 ? messages[0].content.slice(0, 26) + "..." : "New chat"}
              </span>
            </div>
          </div>
        </div>

        <div className="sidebar-bottom">
          <div className="user-profile-row">
            <div className="user-avatar-circle">U</div>
            <div className="user-details">
              <span className="user-name">Local User</span>
              <span className="user-plan">Free Plan</span>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Workspace */}
      <main className="chat-workspace">
        {/* Top Bar */}
        <header className="top-nav">
          <div className="nav-left">
            {sidebarCollapsed && (
              <button
                className="icon-btn-ghost"
                onClick={() => setSidebarCollapsed(false)}
                title="Open sidebar"
              >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                  <line x1="9" y1="3" x2="9" y2="21"></line>
                </svg>
              </button>
            )}
            <div className="model-pill">
              <span className="model-title">ChatGPT</span>
              <span className="model-sub">4o mini</span>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                <polyline points="6 9 12 15 18 9"></polyline>
              </svg>
            </div>
          </div>

          <div className="nav-right">
            <button className="clear-btn" onClick={handleNewChat} title="Clear chat">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="3 6 5 6 21 6"></polyline>
                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
              </svg>
            </button>
          </div>
        </header>

        {/* Scrollable Viewport */}
        <div className="chat-scroll-container" ref={scrollRef}>
          {messages.length === 0 && !currentStreamingText ? (
            <div className="hero-screen">
              <h1 className="hero-heading">What can I help with today?</h1>
              <div className="prompt-chips-grid">
                <button
                  className="chip-card"
                  onClick={() => handleSend("Explain Git workflows, commits, branching, and rebasing.")}
                >
                  <span className="chip-title">Explain Git workflows</span>
                  <span className="chip-desc">Commits, branching & rebasing</span>
                </button>
                <button
                  className="chip-card"
                  onClick={() => handleSend("Write a clean REST API in FastAPI with Pydantic validation and error handling.")}
                >
                  <span className="chip-title">FastAPI REST endpoint</span>
                  <span className="chip-desc">Pydantic models and CRUD</span>
                </button>
                <button
                  className="chip-card"
                  onClick={() => handleSend("Explain Rust ownership, borrowing, and lifetimes with code examples.")}
                >
                  <span className="chip-title">Rust ownership & borrowing</span>
                  <span className="chip-desc">Memory safety without GC</span>
                </button>
                <button
                  className="chip-card"
                  onClick={() => handleSend("What is Docker and how do containers differ from Virtual Machines?")}
                >
                  <span className="chip-title">Docker vs Virtual Machines</span>
                  <span className="chip-desc">Container architecture & namespaces</span>
                </button>
              </div>
            </div>
          ) : (
            <div className="messages-wrapper">
              {messages.map((msg, index) => (
                <div key={index} className={`message-row ${msg.role}`}>
                  <div className="message-body">
                    {msg.role === "user" ? (
                      msg.content
                    ) : (
                      <MarkdownRenderer content={msg.content} />
                    )}
                  </div>
                </div>
              ))}

              {/* 3-Dots Typing Indicator right when message is sent */}
              {isGenerating && !currentStreamingText && (
                <div className="message-row assistant">
                  <div className="message-body">
                    <div className="typing-dots-indicator">
                      <span className="dot"></span>
                      <span className="dot"></span>
                      <span className="dot"></span>
                    </div>
                  </div>
                </div>
              )}

              {/* Live Streaming Response with Inline Cursor */}
              {isGenerating && currentStreamingText && (
                <div className="message-row assistant">
                  <div className="message-body">
                    <MarkdownRenderer content={currentStreamingText} isStreaming={true} />
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Input Capsule */}
        <div className="input-area-wrapper">
          <form
            className="input-capsule-form"
            onSubmit={(e) => {
              e.preventDefault();
              handleSend();
            }}
          >
            <div className="input-capsule">
              <textarea
                ref={textareaRef}
                className="chat-input"
                placeholder="Message ChatGPT..."
                rows="1"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={isGenerating}
                autoFocus
              ></textarea>

              <div className="input-actions-bar">
                <button
                  type="submit"
                  className="send-circle-btn"
                  disabled={!input.trim() || isGenerating}
                  title="Send message"
                >
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                    <line x1="12" y1="19" x2="12" y2="5"></line>
                    <polyline points="5 12 12 5 19 12"></polyline>
                  </svg>
                </button>
              </div>
            </div>
            <div className="disclaimer-text">
              ChatGPT can make mistakes. Check important info.
            </div>
          </form>
        </div>
      </main>
    </div>
  );
}

// Markdown Renderer Component with proper code blocks & Inline Streaming Cursor
function MarkdownRenderer({ content, isStreaming = false }) {
  const containerRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current) return;

    // Attach copy button listeners to any rendered code blocks
    const codeBlocks = containerRef.current.querySelectorAll("pre code");
    codeBlocks.forEach((block) => {
      hljs.highlightElement(block);

      const pre = block.parentElement;
      if (pre.parentElement.classList.contains("code-container")) return;

      const wrapper = document.createElement("div");
      wrapper.className = "code-container";

      const header = document.createElement("div");
      header.className = "code-header-bar";

      const lang = block.className.replace("hljs language-", "") || "code";
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
        navigator.clipboard.writeText(block.innerText);
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

      pre.parentNode.insertBefore(wrapper, pre);
      wrapper.appendChild(header);
      wrapper.appendChild(pre);
    });
  }, [content, isStreaming]);

  // Insert streaming cursor inline into markdown HTML
  let parsedHtml = marked.parse(content || "");
  if (isStreaming) {
    // If parsed HTML ends with </p>, insert the cursor inside the last <p> tag to prevent line-breaking!
    if (parsedHtml.endsWith("</p>\n") || parsedHtml.endsWith("</p>")) {
      parsedHtml = parsedHtml.replace(/<\/p>(?:\n)?$/, '<span class="streaming-cursor"></span></p>');
    } else {
      parsedHtml += '<span class="streaming-cursor"></span>';
    }
  }

  return (
    <div
      ref={containerRef}
      className="markdown-content"
      dangerouslySetInnerHTML={{ __html: parsedHtml }}
    />
  );
}

// Mount React App
const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<App />);
