# ChatGPT Local AI Assistant

A fast, private, offline-capable AI Assistant powered by a lightweight neural instruct engine with real-time word-by-word streaming and an authentic ChatGPT dark interface.

---

## 🚀 Quick Start

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Launch the Web Interface
```powershell
python app.py
```
Open your browser at: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 💻 Terminal CLI Mode

To chat directly inside your terminal:
```powershell
python chat_instruct.py
```

---

## ✨ Features

- **ChatGPT UI Replica**: Authentic dark theme (`#212121` workspace & `#171717` sidebar), floating input capsule, and responsive layout.
- **Real-Time Word Streaming**: Server-Sent Events (SSE) streaming with inline blinking cursor and 3-dots typing loader.
- **Full History Persistence**: Conversations and multi-session side panel history automatically persist in `localStorage`.
- **Code Highlighting & Copy**: Formatted code blocks with syntax highlighting and 1-click **Copy code** buttons.
- **Zero Cloud Costs / Offline Capable**: Runs locally on CPU/GPU without third-party API keys or recurring charges.

---

## 📁 Project Structure

```
├── app.py              # FastAPI full-stack streaming backend server
├── chat_instruct.py    # Terminal CLI chat runner
├── requirements.txt    # Python package dependencies
├── README.md           # Project documentation and quickstart
└── web/                # React JSX ChatGPT Frontend
    ├── index.html      # HTML5 container & React 18 / Babel mounting
    ├── app.jsx         # React components, SSE streaming & localStorage history
    └── style.css       # ChatGPT dark theme stylesheet
```

---

## 📜 License
MIT License
