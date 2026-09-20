const { useState, useEffect, useRef } = React;

// Safe Escape HTML helper
function escapeHtml(str) {
  if (str === null || str === undefined) return "";
  const s = typeof str === "string" ? str : String(str.text || str.raw || str || "");
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// Custom Marked Renderer for Authentic ChatGPT Code Blocks
const renderer = new marked.Renderer();

renderer.code = function (tokenOrCode, maybeLang) {
  let codeStr = "";
  let langStr = "";

  if (typeof tokenOrCode === "object" && tokenOrCode !== null) {
    codeStr = tokenOrCode.text || tokenOrCode.raw || "";
    langStr = tokenOrCode.lang || "";
  } else {
    codeStr = typeof tokenOrCode === "string" ? tokenOrCode : String(tokenOrCode || "");
    langStr = typeof maybeLang === "string" ? maybeLang : "";
  }

  const rawLang = (langStr || "").trim();
  const cleanLang = rawLang.replace(/^language-/, "").toLowerCase();
  const displayLang = cleanLang || "code";
  const validLang = hljs.getLanguage(cleanLang) ? cleanLang : null;

  let highlighted = "";
  try {
    highlighted = validLang
      ? hljs.highlight(codeStr, { language: validLang }).value
      : escapeHtml(codeStr);
  } catch (err) {
    highlighted = escapeHtml(codeStr);
  }

  const encodedCode = encodeURIComponent(codeStr);

  return `
    <div class="code-container">
      <div class="code-header-bar">
        <span class="code-lang-text">${escapeHtml(displayLang)}</span>
        <button class="copy-button" onclick="(function(btn){
          navigator.clipboard.writeText(decodeURIComponent('${encodedCode}'));
          btn.innerHTML='<svg width=\\'14\\' height=\\'14\\' viewBox=\\'0 0 24 24\\' fill=\\'none\\' stroke=\\'currentColor\\' stroke-width=\\'2\\'><polyline points=\\'20 6 9 17 4 12\\'></polyline></svg><span>Copied!</span>';
          setTimeout(function(){
            btn.innerHTML='<svg width=\\'14\\' height=\\'14\\' viewBox=\\'0 0 24 24\\' fill=\\'none\\' stroke=\\'currentColor\\' stroke-width=\\'2\\'><rect x=\\'9\\' y=\\'9\\' width=\\'13\\' height=\\'13\\' rx=\\'2\\' ry=\\'2\\'></rect><path d=\\'M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1\\'></path></svg><span>Copy code</span>';
          }, 2000);
        })(this)">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
          </svg>
          <span>Copy code</span>
        </button>
      </div>
      <pre><code class="hljs ${validLang || ''}">${highlighted}</code></pre>
    </div>
  `;
};

marked.setOptions({
  renderer: renderer,
  breaks: true,
  gfm: true,
});

const STORAGE_SESSIONS_KEY = "chatgpt_local_sessions";
const STORAGE_ACTIVE_KEY = "chatgpt_local_active_id";

function generateId() {
  return "session_" + Date.now() + "_" + Math.random().toString(36).substr(2, 6);
}

function loadInitialSessions() {
  try {
    const saved = localStorage.getItem(STORAGE_SESSIONS_KEY);
    if (saved) {
      const parsed = JSON.parse(saved);
      if (Array.isArray(parsed) && parsed.length > 0) {
        return parsed;
      }
    }
  } catch (e) {
    console.error("Error loading sessions from localStorage", e);
  }
  const defaultSession = {
    id: generateId(),
    title: "New chat",
    messages: [],
    updatedAt: Date.now(),
  };
  return [defaultSession];
}

function loadInitialActiveId(sessions) {
  const savedActive = localStorage.getItem(STORAGE_ACTIVE_KEY);
  if (savedActive && sessions.some((s) => s.id === savedActive)) {
    return savedActive;
  }
  return sessions[0].id;
}

function App() {
  const [sessions, setSessions] = useState(loadInitialSessions);
  const [activeSessionId, setActiveSessionId] = useState(() => loadInitialActiveId(sessions));
  const [input, setInput] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [currentStreamingText, setCurrentStreamingText] = useState("");
  
  const scrollRef = useRef(null);
  const textareaRef = useRef(null);
  const abortControllerRef = useRef(null);
  const streamBufferRef = useRef("");
  const displayedStreamRef = useRef("");
  const streamTimerRef = useRef(null);

  const activeSession = sessions.find((s) => s.id === activeSessionId) || sessions[0] || {
    id: generateId(),
    title: "New chat",
    messages: [],
    updatedAt: Date.now(),
  };
  const messages = activeSession ? activeSession.messages || [] : [];

  const cleanupStream = () => {
    if (streamTimerRef.current) {
      clearInterval(streamTimerRef.current);
      streamTimerRef.current = null;
    }
    streamBufferRef.current = "";
    displayedStreamRef.current = "";
  };

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_SESSIONS_KEY, JSON.stringify(sessions));
      localStorage.setItem(STORAGE_ACTIVE_KEY, activeSessionId);
    } catch (e) {
      console.error("Failed to save to localStorage", e);
    }
  }, [sessions, activeSessionId]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, currentStreamingText, isGenerating]);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 180) + "px";
    }
  }, [input]);

  const handleNewChat = () => {
    cleanupStream();
    if (abortControllerRef.current) {
      try {
        abortControllerRef.current.abort();
      } catch (e) {}
      abortControllerRef.current = null;
    }
    setIsGenerating(false);
    setCurrentStreamingText("");
    setInput("");

    // If current session is already an empty new chat, just focus
    if (activeSession && (!activeSession.messages || activeSession.messages.length === 0)) {
      if (textareaRef.current) {
        textareaRef.current.focus();
      }
      return;
    }

    // Always create a clean fresh session at top and prune any unused empty chats
    const newSession = {
      id: generateId(),
      title: "New chat",
      messages: [],
      updatedAt: Date.now(),
    };
    setSessions((prev) => [newSession, ...prev.filter((s) => s.messages && s.messages.length > 0)]);
    setActiveSessionId(newSession.id);
    if (textareaRef.current) {
      setTimeout(() => textareaRef.current && textareaRef.current.focus(), 50);
    }
  };

  const handleSelectSession = (id) => {
    if (id === activeSessionId) return;
    cleanupStream();
    if (abortControllerRef.current) {
      try {
        abortControllerRef.current.abort();
      } catch (e) {}
      abortControllerRef.current = null;
    }
    setIsGenerating(false);
    setCurrentStreamingText("");
    setInput("");
    setActiveSessionId(id);
    if (textareaRef.current) {
      setTimeout(() => textareaRef.current && textareaRef.current.focus(), 50);
    }
  };

  const handleDeleteSession = (id, e) => {
    e.stopPropagation();
    if (id === activeSessionId) {
      cleanupStream();
      if (abortControllerRef.current) {
        try {
          abortControllerRef.current.abort();
        } catch (err) {}
        abortControllerRef.current = null;
      }
      setIsGenerating(false);
      setCurrentStreamingText("");
    }

    setSessions((prev) => {
      const remaining = prev.filter((s) => s.id !== id);
      if (remaining.length === 0) {
        const fresh = {
          id: generateId(),
          title: "New chat",
          messages: [],
          updatedAt: Date.now(),
        };
        setActiveSessionId(fresh.id);
        return [fresh];
      }
      if (id === activeSessionId) {
        setActiveSessionId(remaining[0].id);
      }
      return remaining;
    });
  };

  const handleSend = async (textToSend) => {
    const text = (textToSend || input).trim();
    if (!text || isGenerating) return;

    cleanupStream();
    if (abortControllerRef.current) {
      try {
        abortControllerRef.current.abort();
      } catch (e) {}
    }
    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    const userMessage = { role: "user", content: text };
    const currentMessages = activeSession.messages || [];
    const updatedMessages = [...currentMessages, userMessage];

    const newTitle =
      currentMessages.length === 0
        ? text.slice(0, 28) + (text.length > 28 ? "..." : "")
        : activeSession.title;

    const targetSessionId = activeSessionId;

    setSessions((prev) =>
      prev.map((s) =>
        s.id === targetSessionId
          ? { ...s, title: newTitle, messages: updatedMessages, updatedAt: Date.now() }
          : s
      )
    );

    setInput("");
    setIsGenerating(true);
    setCurrentStreamingText("");
    streamBufferRef.current = "";
    displayedStreamRef.current = "";

    // Start 60fps smooth typewriter ticker
    streamTimerRef.current = setInterval(() => {
      const buffer = streamBufferRef.current;
      const current = displayedStreamRef.current;
      if (buffer.length > current.length) {
        const diff = buffer.length - current.length;
        const step = diff > 80 ? 6 : diff > 30 ? 3 : diff > 8 ? 2 : 1;
        const nextLen = Math.min(buffer.length, current.length + step);
        const nextText = buffer.slice(0, nextLen);
        displayedStreamRef.current = nextText;
        setCurrentStreamingText(nextText);
      }
    }, 16);

    try {
      const response = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: updatedMessages }),
        signal: abortController.signal,
      });

      if (!response.ok) {
        throw new Error(`Server returned error ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let fullAssistantText = "";
      let streamFinished = false;

      while (!streamFinished) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split("\n");

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const dataStr = line.replace("data: ", "").trim();
            if (dataStr === "[DONE]") {
              streamFinished = true;
              break;
            }
            try {
              const data = JSON.parse(dataStr);
              if (data.token) {
                fullAssistantText += data.token;
                streamBufferRef.current = fullAssistantText;
              }
            } catch (err) {
              if (dataStr && !dataStr.startsWith("{")) {
                fullAssistantText += dataStr;
                streamBufferRef.current = fullAssistantText;
              }
            }
          }
        }
      }

      // Smoothly flush any remaining characters in the typing queue
      while (displayedStreamRef.current.length < streamBufferRef.current.length) {
        await new Promise((r) => setTimeout(r, 16));
      }

      cleanupStream();

      setSessions((prev) =>
        prev.map((s) =>
          s.id === targetSessionId
            ? {
                ...s,
                messages: [
                  ...updatedMessages,
                  { role: "assistant", content: fullAssistantText },
                ],
                updatedAt: Date.now(),
              }
            : s
        )
      );
      setCurrentStreamingText("");
    } catch (err) {
      cleanupStream();
      if (err.name === "AbortError") {
        console.log("Generation aborted by user");
      } else {
        const isNetworkErr = err.message && (
          err.message.toLowerCase().includes("failed to fetch") ||
          err.message.toLowerCase().includes("network")
        );
        const errorMsg = isNetworkErr
          ? "Connection to the local server was temporarily interrupted while restarting. Please try sending your message again."
          : `[Error: ${err.message}]`;

        setSessions((prev) =>
          prev.map((s) =>
            s.id === targetSessionId
              ? {
                  ...s,
                  messages: [
                    ...updatedMessages,
                    { role: "assistant", content: errorMsg },
                  ],
                  updatedAt: Date.now(),
                }
              : s
          )
        );
      }
      setCurrentStreamingText("");
    } finally {
      cleanupStream();
      setIsGenerating(false);
      abortControllerRef.current = null;
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
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
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
              <line x1="9" y1="3" x2="9" y2="21"></line>
            </svg>
          </button>
          <button
            className="new-chat-icon-btn"
            onClick={handleNewChat}
            title="New chat"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 20h9"></path>
              <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"></path>
            </svg>
          </button>
        </div>

        <div className="sidebar-new-chat-wrapper">
          <button className="sidebar-new-chat-btn" onClick={handleNewChat}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="12" y1="5" x2="12" y2="19"></line>
              <line x1="5" y1="12" x2="19" y2="12"></line>
            </svg>
            <span>New chat</span>
          </button>
        </div>

        <div className="sidebar-content">
          <div className="history-group-title">Recent Chats</div>
          <div className="history-list">
            {sessions.map((session) => (
              <div
                key={session.id}
                className={`history-item ${session.id === activeSessionId ? "active" : ""}`}
                onClick={() => handleSelectSession(session.id)}
              >
                <span className="history-item-text">{session.title || "New chat"}</span>
                <button
                  className="history-delete-btn"
                  onClick={(e) => handleDeleteSession(session.id, e)}
                  title="Delete chat"
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <polyline points="3 6 5 6 21 6"></polyline>
                    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                  </svg>
                </button>
              </div>
            ))}
          </div>
        </div>

        <div className="sidebar-bottom">
          <div className="user-profile-row">
            <div className="user-avatar-circle">U</div>
            <div className="user-details">
              <span className="user-name">Local User</span>
              <span className="user-plan">Persistent History</span>
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
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                  <line x1="9" y1="3" x2="9" y2="21"></line>
                </svg>
              </button>
            )}
            <div className="model-pill">
              <span className="model-title">ChatGPT</span>
              <span className="model-sub">4o mini</span>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <polyline points="6 9 12 15 18 9"></polyline>
              </svg>
            </div>
          </div>

          <div className="nav-right">
            <button className="clear-btn" onClick={handleNewChat} title="New chat">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M12 20h9"></path>
                <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"></path>
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
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
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

// Markdown Renderer Component with proper code blocks & Inline Streaming Cursor (Memoized for high FPS)
const MarkdownRenderer = React.memo(function MarkdownRenderer({ content, isStreaming = false }) {
  let parsedHtml = "";
  try {
    parsedHtml = marked.parse(content || "");
  } catch (err) {
    parsedHtml = escapeHtml(content || "");
  }

  if (isStreaming) {
    if (parsedHtml.endsWith("</p>\n") || parsedHtml.endsWith("</p>")) {
      parsedHtml = parsedHtml.replace(/<\/p>(?:\n)?$/, '<span class="streaming-cursor"></span></p>');
    } else {
      parsedHtml += '<span class="streaming-cursor"></span>';
    }
  }

  return (
    <div
      className="markdown-content"
      dangerouslySetInnerHTML={{ __html: parsedHtml }}
    />
  );
});

// Mount React App
const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<App />);
