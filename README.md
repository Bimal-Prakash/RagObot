# 🤖 RagObot: Multi-Agent Concurrent RAG Matrix
RagObot was Built by Bimal P B

RagObot is a headless, highly optimized multi-agent Retrieval-Augmented Generation (RAG) system built with **LangGraph**. It orchestrates local compute (**Ollama**) and cloud compute (**Groq LPU**) concurrently, strictly monitors rolling context window budgets, displays live thought calculations on a futuristic **React Cyber-Console** (or interactive terminal console via **Rich**), and dispatches final responses to native Windows Text-to-Speech via **WinSDK SAPI** (with toggleable browser Web Speech).

---

## 🗺️ System Architecture & Workflow

```mermaid
flowchart TD
    A[User query] --> B[Retrieve ChromaDB chunks]
    B --> C{Relevant evidence available?}
    C -- No --> D[Return abstention]
    C -- Yes --> E[Ollama and Groq workers]
    E --> F[Consensus arbiter]
    F --> G{Answer grounded in sources?}
    G -- No --> D
    G -- Yes --> H[Final response and TTS]
```

---

## 🛠️ Core Pillars & Implementation

1. **Self-Checking Concurrency Orchestration Layer (`langgraph`)**:
   - Follows strict evidence validation and factual grounding gates (`mermaid-diagram.png`).
   - `START` routes to `Retrieve ChromaDB chunks`.
   - **Relevant evidence available?**: If ChromaDB does not contain sufficient verified evidence, the graph short-circuits directly to `Return abstention` to prevent hallucinations.
   - **Ollama and Groq workers**: Executes concurrent compute nodes (`ollama_worker` & `groq_worker`) in parallel via `asyncio`.
   - **Consensus Arbiter**: Cross-examines worker outputs against ChromaDB chunks.
   - **Answer grounded in sources?**: Evaluates factual grounding against source documents. If ungrounded or hallucinated claims are detected, routes to `Return abstention`.
   - **Final response and TTS**: Dispatches the verified, grounded response to native Windows SAPI TTS and UI.
   - Ingestion protocol reads structured Markdown (`.md`) files using `MarkdownHeaderTextSplitter` to retain semantic header hierarchy (`#`, `##`, `###`).
2. **Dual-Engine Compute Matrix (`groq` + `ollama`)**:
   - **Groq Cloud Engine**: Powered by `ChatGroq` (`llama-3.3-70b-versatile`) as the structural supervisor.
   - **Ollama Local Matrix**: Powered by `ChatOllama` (`llama3.2:3b` / `llama3`) for localized verification.
3. **Consensus Arbiter Node**:
   - Neither engine sends raw output directly to the user.
   - The Consensus Arbiter cross-examines outputs against source documents, resolves conflicts, eliminates hallucinations, and synthesizes a single natural response.
4. **Context Window & Token Budget Management**:
   - Guarded by `trim_messages()` with `strategy="last"` and `include_system=True`.
   - Prunes history when total tokens exceed `MAX_CONTEXT_TOKENS` (default: 2,000) while permanently anchoring system RAG rules.
   - Live visual alert badge activates whenever trimming occurs.
5. **Headless Native Hardware Voice Pipeline (`win32com`) & Hybrid Web Speech**:
   - Dispatches synthesized text directly to Windows SAPI SpVoice asynchronously (`SVSFlagsAsync`) with zero UI lag.
   - React UI also features toggleable browser Web Speech synthesis for direct tab audio.
6. **Cyber-Console React Web UI (`frontend/`)**:
   - Glassmorphic cyberpunk aesthetic with neon glowing borders, dynamic thought streams, and real-time Server-Sent Events (SSE).
   - Side-by-side stream columns for Ollama (Cyan) and Groq (Yellow) in real time.
   - Live rolling token budget gauge bar with automatic pruning alerts.
   - Retrieved Markdown boundaries viewer with breadcrumbs and similarity scores.

---

## 🚀 Getting Started

### 1. Install Backend Dependencies
```powershell
.venv\Scripts\pip install -r requirements.txt
```

### 2. Configure Environment
Copy `.env.example` to `.env`:
```powershell
copy .env.example .env
```
Add your Groq API key and preferred models in `.env`.

### 3. Run the Backend API (FastAPI)
```powershell
.venv\Scripts\python -m uvicorn api:app --reload --port 8000
```

### 4. Run the React Frontend
In a new terminal window:
```powershell
cd frontend
npm install
npm run dev
```
Open **`http://localhost:5173`** in your browser.

*(Optional)* You can also still run the pure terminal cyber-console anytime:
```powershell
.venv\Scripts\python main.py
```
