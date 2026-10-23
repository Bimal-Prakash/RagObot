import asyncio
import json
import uuid
import urllib.request
import shutil
import os
import time
import hmac
import re
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Optional
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response, JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from langchain_core.messages import HumanMessage, AIMessage

import config
from core.graph import ragobot_graph, set_stream_listeners, is_document_referencing_query
from core.tts import tts_engine
from core.memory import prune_context_window
from core.thread_store import thread_store
from ingestion.rag_engine import rag_engine

app = FastAPI(
    title="RagObot Multi-Agent API",
    description="Hyper-concurrent LangGraph multi-agent RAG backend with SSE streaming"
)

# Enable CORS with explicit origin allowlist from configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Standardized Exception Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "error": exc.detail, "code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"status": "error", "error": "Internal server error", "code": 500}
    )

# Validation Constants
THREAD_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")
ALLOWED_UPLOAD_EXTENSIONS = {".md", ".markdown", ".txt", ".pdf"}
MAX_UPLOAD_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB

def validate_thread_id(thread_id: Optional[str]) -> str:
    """Validates thread ID format to prevent directory traversal and injection."""
    clean = (thread_id or "").strip()
    if not clean or not THREAD_ID_PATTERN.match(clean):
        raise HTTPException(
            status_code=400,
            detail="Invalid thread_id format. Must be 1-64 alphanumeric characters, underscores, or hyphens."
        )
    return clean

# In-memory session tracking
session_state: dict[str, Any] = {
    "thread_id": str(uuid.uuid4()),
    "token_count": 0,
    "was_trimmed": False,
    "last_uploaded_doc": None,
    "history": []
}

def extract_dev_key_from_headers(request: Request) -> Optional[str]:
    """
    Extracts developer secret key exclusively from standard HTTP headers.
    Accepted headers:
      - X-Dev-Key: <secret>
      - Authorization: Bearer <secret>
    Query parameters and JSON request body keys are strictly rejected.
    """
    header_key = request.headers.get("x-dev-key")
    if not header_key and request.headers.get("authorization"):
        auth_hdr = request.headers.get("authorization", "").strip()
        if auth_hdr.startswith("Bearer "):
            header_key = auth_hdr[7:].strip()
    return header_key

def is_dev_authorized(
    requested_dev_mode: bool,
    header_key: Optional[str] = None
) -> bool:
    """
    Validates whether developer diagnostic mode is authorized on the backend.
    - If requested_dev_mode is False: returns False.
    - If ENVIRONMENT is 'production': strictly enforces DEV_MODE_KEY verification via header,
      ignoring ALLOW_UNAUTHENTICATED_DEV_MODE to prevent production bypass.
    - If ENVIRONMENT is non-production AND ALLOW_UNAUTHENTICATED_DEV_MODE is True: returns True.
    - Otherwise: requires configured DEV_MODE_KEY matching header_key (X-Dev-Key or Authorization: Bearer).
    Fails closed if DEV_MODE_KEY is unset and unauthenticated access is disallowed.
    """
    if not requested_dev_mode:
        return False

    is_production = str(getattr(config, "ENVIRONMENT", "development")).strip().lower() == "production"

    # Only permit unauthenticated dev mode outside of production environments
    if not is_production and getattr(config, "ALLOW_UNAUTHENTICATED_DEV_MODE", False):
        return True

    configured_key = (config.DEV_MODE_KEY or "").strip()
    if not configured_key:
        # Fail closed: No authorization key configured and unauthenticated dev mode is disabled.
        return False

    token = (header_key or "").strip()
    return hmac.compare_digest(token, configured_key)

# Explicit Pydantic Schemas for API Requests & Responses
class PublicMessageModel(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    role: str
    content: str
    timestamp: Optional[Any] = None
    sources: list[Any] = Field(default_factory=list)
    evidenceAvailable: bool = False
    answerGrounded: bool = False
    isAbstained: bool = False
    targetDoc: Optional[str] = None

class DiagnosticMessageModel(PublicMessageModel):
    ollamaStream: Optional[str] = None
    groqStream: Optional[str] = None
    arbiterDebate: Optional[str] = None
    evidenceReason: Optional[str] = None
    groundingReason: Optional[str] = None
    groundedAnswer: Optional[str] = None
    abstentionReason: Optional[str] = None
    tokenCount: int = 0
    wasTrimmed: bool = False

class ThreadSummaryModel(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    title: str
    created_at: Optional[Any] = None
    updated_at: Optional[Any] = None
    message_count: int = 0
    last_uploaded_doc: Optional[str] = None

class ThreadListResponse(BaseModel):
    threads: list[ThreadSummaryModel]

class ThreadDetailModel(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    title: str
    created_at: Optional[Any] = None
    updated_at: Optional[Any] = None
    last_uploaded_doc: Optional[str] = None
    token_count: int = 0
    was_trimmed: bool = False
    is_diagnostic: bool = False
    messages: list[Any] = Field(default_factory=list)

class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query: str = Field(..., min_length=1)
    thread_id: Optional[str] = None
    uploaded_doc: Optional[str] = None
    trigger_backend_tts: bool = False  # False for web clients (browser plays audio), True for terminal
    model_mode: str = "dual"  # "dual", "local", "cloud"
    local_model: Optional[str] = None
    cloud_model: Optional[str] = None
    dev_mode: bool = False  # If True, requests developer diagnostic mode; must be authorized by backend header

class GroqSettingsRequest(BaseModel):
    api_key: Optional[str] = None
    model: Optional[str] = None

class GroqSettingsResponse(BaseModel):
    status: str
    groq_ready: bool
    groq_model: str
    message: str

class SynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice: Optional[str] = None

class ThreadTitleRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=120)

class CreateThreadRequest(BaseModel):
    title: Optional[str] = "New Chat"

class UploadResponse(BaseModel):
    status: str
    filename: str
    indexed_chunks: int

class ModelsResponse(BaseModel):
    local_models: list[str]
    cloud_models: list[str]
    default_local: Optional[str] = None
    default_cloud: Optional[str] = None
    groq_ready: bool

class StatusResponse(BaseModel):
    ollama_model: Optional[str] = None
    ollama_url: str
    groq_model: Optional[str] = None
    groq_ready: bool
    tts_enabled: bool
    token_budget: int
    indexed_chunks: int
    vector_db: str
    embedding_model: str
    current_tokens: int
    was_trimmed: bool

def format_message_for_client(msg: dict[str, Any], is_authorized_dev: bool) -> dict[str, Any]:
    """
    Applies strict allowlist schema filtering to conversation messages.
    - Public mode (is_authorized_dev=False):
      Returns ONLY public fields (id, role, content, timestamp, sources, evidenceAvailable, answerGrounded, isAbstained).
      Internal diagnostic fields (ollamaStream, groqStream, arbiterDebate, groundedAnswer, evidenceReason,
      groundingReason, abstentionReason, tokenCount, wasTrimmed) are completely excluded from the response.
    - Developer mode (is_authorized_dev=True):
      Returns all message fields including raw worker drafts, debate logs, and evaluation metrics.
    """
    if is_authorized_dev:
        return dict(msg)

    role = msg.get("role")
    filtered: dict[str, Any] = {
        "id": msg.get("id", ""),
        "role": role,
        "content": msg.get("content", ""),
        "timestamp": msg.get("timestamp", "")
    }
    if role == "assistant":
        filtered["sources"] = msg.get("sources", [])
        filtered["evidenceAvailable"] = bool(msg.get("evidenceAvailable", False))
        filtered["answerGrounded"] = bool(msg.get("answerGrounded", False))
        filtered["isAbstained"] = bool(msg.get("isAbstained", False))
        if "targetDoc" in msg:
            filtered["targetDoc"] = msg["targetDoc"]
    return filtered

def format_thread_for_client(thread: dict[str, Any], is_authorized_dev: bool) -> dict[str, Any]:
    """
    Formats thread response, enforcing strict separation of public fields from internal diagnostics.
    """
    raw_messages = thread.get("messages", [])
    filtered_messages = [format_message_for_client(m, is_authorized_dev) for m in raw_messages]

    formatted = {
        "id": thread.get("id"),
        "title": thread.get("title", "New Chat"),
        "created_at": thread.get("created_at"),
        "updated_at": thread.get("updated_at"),
        "last_uploaded_doc": thread.get("last_uploaded_doc"),
        "token_count": thread.get("token_count", 0),
        "was_trimmed": thread.get("was_trimmed", False),
        "is_diagnostic": is_authorized_dev,
        "messages": filtered_messages
    }
    return formatted

def update_env_file(updates: dict[str, str], env_path: Path = config.BASE_DIR / ".env") -> bool:
    lines = []
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
    updated_keys = set()
    new_lines = []
    
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            k, _ = stripped.split("=", 1)
            key = k.strip()
            if key in updates and updates[key] is not None:
                new_lines.append(f"{key}={updates[key]}\n")
                updated_keys.add(key)
                continue
        new_lines.append(line)
        
    for k, v in updates.items():
        if k not in updated_keys and v is not None:
            new_lines.append(f"{k}={v}\n")
            
    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
        
    return True

@app.on_event("startup")
async def startup_event():
    """Ingest markdown documents on server boot."""
    count = rag_engine.ingest_documents()
    print(f"[OK] RagObot API online. Ingested {count} chunks into ChromaDB (Nomic).")

@app.get("/api/models", response_model=ModelsResponse)
async def get_models():
    """Returns available local models from Ollama and cloud models from Groq."""
    local_models = []
    try:
        req = urllib.request.Request(f"{config.OLLAMA_BASE_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            if resp.status == 200:
                body = json.loads(resp.read().decode())
                local_models = [m["name"] for m in body.get("models", [])]
    except Exception:
        pass

    # Prioritize user's configured model from .env at the front of the list
    if config.OLLAMA_MODEL:
        if config.OLLAMA_MODEL in local_models:
            local_models.remove(config.OLLAMA_MODEL)
        local_models.insert(0, config.OLLAMA_MODEL)

    if not local_models:
        local_models = [config.OLLAMA_MODEL or "llama3.2"]

    cloud_models = [
        config.GROQ_MODEL,
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768"
    ]
    cloud_models = list(dict.fromkeys([m for m in cloud_models if m]))

    return {
        "local_models": local_models,
        "cloud_models": cloud_models,
        "default_local": config.OLLAMA_MODEL,
        "default_cloud": config.GROQ_MODEL,
        "groq_ready": bool(config.GROQ_API_KEY)
    }

@app.post("/api/settings/groq", response_model=GroqSettingsResponse)
async def update_groq_settings(req: GroqSettingsRequest, request: Request):
    """Saves Groq API Key and custom model directly into .env and updates in-memory config."""
    is_production = str(getattr(config, "ENVIRONMENT", "development")).strip().lower() == "production"
    if is_production:
        header_key = extract_dev_key_from_headers(request)
        if not is_dev_authorized(requested_dev_mode=True, header_key=header_key):
            raise HTTPException(
                status_code=403,
                detail="Modifying server configuration via API is restricted in production mode."
            )

    updates = {}
    if req.api_key is not None:
        clean_key = req.api_key.strip()
        updates["GROQ_API_KEY"] = clean_key
        os.environ["GROQ_API_KEY"] = clean_key
        config.GROQ_API_KEY = clean_key
        if hasattr(config, "settings"):
            config.settings.GROQ_API_KEY = clean_key

    if req.model is not None and req.model.strip():
        clean_model = req.model.strip()
        updates["GROQ_MODEL"] = clean_model
        os.environ["GROQ_MODEL"] = clean_model
        config.GROQ_MODEL = clean_model
        if hasattr(config, "settings"):
            config.settings.GROQ_MODEL = clean_model

    if updates:
        update_env_file(updates, config.BASE_DIR / ".env")

    return {
        "status": "success",
        "groq_ready": bool(config.GROQ_API_KEY),
        "groq_model": config.GROQ_MODEL or "",
        "message": "Groq settings saved to .env"
    }

@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """Returns real-time system configuration and model statuses."""
    return {
        "ollama_model": config.OLLAMA_MODEL,
        "ollama_url": config.OLLAMA_BASE_URL,
        "groq_model": config.GROQ_MODEL,
        "groq_ready": bool(config.GROQ_API_KEY),
        "tts_enabled": config.TTS_ENABLED,
        "token_budget": config.MAX_CONTEXT_TOKENS,
        "indexed_chunks": len(rag_engine.chunks),
        "vector_db": "ChromaDB",
        "embedding_model": config.EMBEDDING_MODEL,
        "current_tokens": session_state.get("token_count", 0),
        "was_trimmed": session_state.get("was_trimmed", False)
    }

@app.post("/api/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...), thread_id: Optional[str] = Form(None)):
    """Receives an uploaded PDF, Markdown, or TXT file, validates size/extension, saves and indexes it."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename missing in uploaded file.")

    # Guard against directory traversal by keeping only the basename
    safe_filename = Path(file.filename).name
    file_ext = Path(safe_filename).suffix.lower()
    if file_ext not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file_ext}'. Allowed types: {', '.join(sorted(ALLOWED_UPLOAD_EXTENSIONS))}"
        )

    # Validate thread_id if supplied
    if thread_id:
        thread_id = validate_thread_id(thread_id)

    # Read and validate payload size
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum allowed size of {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)}MB."
        )

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    target_path = config.DATA_DIR / safe_filename
    with open(target_path, "wb") as buffer:
        buffer.write(content)
    
    new_count = rag_engine.ingest_documents()
    
    session_state["last_uploaded_doc"] = safe_filename
    if thread_id:
        thread_store.set_thread_uploaded_doc(thread_id, safe_filename)
    elif session_state.get("thread_id"):
        thread_store.set_thread_uploaded_doc(session_state["thread_id"], safe_filename)

    return {
        "status": "success",
        "filename": safe_filename,
        "indexed_chunks": new_count
    }

@app.post("/api/reload-docs")
async def reload_docs():
    """Re-scans and indexes markdown files in data/ directory."""
    new_count = rag_engine.ingest_documents()
    return {"status": "success", "indexed_chunks": new_count}

@app.post("/api/voice/toggle")
async def toggle_voice():
    """Toggles backend Windows SAPI TTS."""
    new_state = tts_engine.toggle()
    return {"tts_enabled": new_state}

@app.post("/api/voice/stop")
async def stop_voice():
    """Stops any currently active Windows speech."""
    tts_engine.stop()
    return {"status": "stopped"}

@app.post("/api/voice/synthesize")
async def synthesize_voice(req: SynthesizeRequest):
    """Generates natural neural audio (MP3) for browser streaming."""
    try:
        audio_bytes = await tts_engine.generate_neural_audio_bytes(req.text, req.voice)
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Unable to synthesize audio")
        return Response(content=audio_bytes, media_type="audio/mpeg")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/threads", response_model=ThreadListResponse)
async def list_threads():
    """List all saved conversation threads."""
    return {"threads": thread_store.list_threads()}

@app.get("/api/threads/{thread_id}", response_model=ThreadDetailModel)
async def get_thread(
    thread_id: str,
    request: Request,
    dev_mode: bool = False
):
    """Get complete conversation thread, strictly separating public schema from developer diagnostics."""
    valid_id = validate_thread_id(thread_id)
    thread = thread_store.get_thread(valid_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
        
    header_key = extract_dev_key_from_headers(request)
    effective_dev_mode = is_dev_authorized(dev_mode, header_key=header_key)
    return format_thread_for_client(thread, effective_dev_mode)

@app.post("/api/threads")
async def create_thread(req: Optional[CreateThreadRequest] = None):
    """Create a new empty conversation thread."""
    title = req.title if req and req.title else "New Chat"
    thread = thread_store.create_thread(title=title)
    session_state["thread_id"] = thread["id"]
    session_state["token_count"] = 0
    session_state["was_trimmed"] = False
    return thread

@app.put("/api/threads/{thread_id}")
async def update_thread_title(thread_id: str, req: ThreadTitleRequest):
    """Update title of an existing conversation thread."""
    valid_id = validate_thread_id(thread_id)
    success = thread_store.update_thread_title(valid_id, req.title)
    if not success:
        raise HTTPException(status_code=404, detail="Thread not found")
    return {"status": "updated", "id": valid_id, "title": req.title}

@app.delete("/api/threads/{thread_id}")
async def delete_thread(thread_id: str):
    """Delete a conversation thread."""
    valid_id = validate_thread_id(thread_id)
    success = thread_store.delete_thread(valid_id)
    if not success:
        raise HTTPException(status_code=404, detail="Thread not found")
    return {"status": "deleted", "id": valid_id}

@app.post("/api/session/reset")
async def reset_session():
    """Resets conversation thread and memory state, creating a new thread."""
    new_thread = thread_store.create_thread(title="New Chat")
    session_state["thread_id"] = new_thread["id"]
    session_state["token_count"] = 0
    session_state["was_trimmed"] = False
    session_state["history"] = []
    return {"status": "reset", "thread_id": new_thread["id"], "thread": new_thread}

@app.post("/api/query")
async def stream_query(payload: QueryRequest, request: Request):
    """
    Executes LangGraph query and streams events to React via Server-Sent Events (SSE).
    Streams:
      - 'docs': Retrieved markdown chunks
      - 'ollama': Tokens from Ollama local worker (developer mode only)
      - 'groq': Tokens from Groq cloud worker (developer mode only)
      - 'consensus': Final synthesis from Arbiter (public response or diagnostic payload)
      - 'diagnostics': Token count and trim alert
    """
    raw_thread_id = payload.thread_id or session_state.get("thread_id")
    thread_id = validate_thread_id(raw_thread_id)
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    header_dev_key = extract_dev_key_from_headers(request)
    effective_dev_mode = is_dev_authorized(payload.dev_mode, header_key=header_dev_key)

    # Ensure thread exists in persistent storage
    thread_record = thread_store.get_thread(thread_id)
    if not thread_record:
        thread_record = thread_store.create_thread(thread_id=thread_id)
    
    # Determine effective uploaded_doc from request, thread, or session
    effective_uploaded_doc = (
        payload.uploaded_doc
        or (thread_record.get("last_uploaded_doc") if thread_record else None)
        or session_state.get("last_uploaded_doc")
    )

    async def event_generator() -> AsyncGenerator[str, None]:
        event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        
        # Stream listener hooks into core graph
        def stream_a(token: str):
            event_queue.put_nowait({"type": "ollama_token", "data": token})
            
        def stream_b(token: str):
            event_queue.put_nowait({"type": "groq_token", "data": token})
            
        def status_cb(event_name: str, payload_data: dict[str, Any]):
            event_queue.put_nowait({"type": event_name, "data": payload_data})
            
        set_stream_listeners(stream_a, stream_b, status_cb)
        graph_retrieved_docs: list[dict[str, Any]] = []
        
        # 1. Run graph in background task
        async def run_graph_task():
            try:
                config_dict = {"configurable": {"thread_id": thread_id}}
                
                # Check checkpointer memory state
                current_graph_state = ragobot_graph.get_state(config_dict)
                existing_messages = current_graph_state.values.get("messages", [])
                
                # If graph memory was cold (e.g. server reboot or older thread loaded), prime with stored messages
                if not existing_messages and thread_record and thread_record.get("messages"):
                    primed_messages = []
                    for m in thread_record["messages"]:
                        role = m.get("role")
                        content = m.get("content", "")
                        if role == "user":
                            primed_messages.append(HumanMessage(content=content))
                        elif role == "assistant":
                            primed_messages.append(AIMessage(content=content))
                    initial_messages = primed_messages + [HumanMessage(content=query)]
                else:
                    initial_messages = [HumanMessage(content=query)]
                
                initial_input = {
                    "query": query,
                    "messages": initial_messages,
                    "model_mode": payload.model_mode,
                    "local_model": payload.local_model,
                    "cloud_model": payload.cloud_model,
                    "trigger_backend_tts": payload.trigger_backend_tts,
                    "uploaded_doc": effective_uploaded_doc
                }
                
                # Check trimming state before execution
                res = await ragobot_graph.ainvoke(initial_input, config=config_dict)
                await event_queue.put({"type": "graph_result", "data": res})
            except Exception as e:
                await event_queue.put({"type": "error", "data": str(e)})
            finally:
                await event_queue.put({"type": "done", "data": {}})
                
        task = asyncio.create_task(run_graph_task())
        
        # Stream events from queue to SSE
        while True:
            item = await event_queue.get()
            event_type = item["type"]
            data = item["data"]
            
            if event_type == "retrieved_docs":
                graph_retrieved_docs = data.get("docs", [])
                yield f"event: docs\ndata: {json.dumps(graph_retrieved_docs)}\n\n"
            elif event_type == "stage":
                yield f"event: stage\ndata: {json.dumps(data)}\n\n"
            elif event_type == "ollama_token":
                # Only stream raw unverified tokens if developer mode is explicitly authorized on backend
                if effective_dev_mode:
                    yield f"event: ollama_token\ndata: {json.dumps({'token': data})}\n\n"
            elif event_type == "groq_token":
                # Only stream raw unverified tokens if developer mode is explicitly authorized on backend
                if effective_dev_mode:
                    yield f"event: groq_token\ndata: {json.dumps({'token': data})}\n\n"
            elif event_type == "evidence_status":
                if effective_dev_mode:
                    yield f"event: evidence_status\ndata: {json.dumps(data)}\n\n"
                else:
                    yield f"event: evidence_status\ndata: {json.dumps({'available': bool(data.get('available', False))})}\n\n"
            elif event_type == "validation_status":
                if effective_dev_mode:
                    yield f"event: validation_status\ndata: {json.dumps(data)}\n\n"
                else:
                    yield f"event: validation_status\ndata: {json.dumps({'grounded': bool(data.get('grounded', False))})}\n\n"
            elif event_type == "graph_result":
                if not graph_retrieved_docs and data.get("retrieved_docs"):
                    graph_retrieved_docs = data.get("retrieved_docs", [])
                    yield f"event: docs\ndata: {json.dumps(graph_retrieved_docs)}\n\n"
                token_count = data.get("token_count", 0)
                was_trimmed = data.get("was_trimmed", False)
                session_state["token_count"] = token_count
                session_state["was_trimmed"] = was_trimmed
                
                # Persist turn to server-side thread_store with all raw model metadata
                user_msg_entry = {
                    "id": f"user-{int(time.time()*1000)}",
                    "role": "user",
                    "content": query,
                    "timestamp": datetime.now().strftime("%I:%M %p")
                }
                assistant_msg_entry = {
                    "id": f"assistant-{int(time.time()*1000)}",
                    "role": "assistant",
                    "content": data.get("final_response", ""),
                    "sources": data.get("retrieved_docs") or graph_retrieved_docs,
                    "ollamaStream": data.get("ollama_output", ""),
                    "groqStream": data.get("groq_output", ""),
                    "arbiterDebate": data.get("arbiter_debate", ""),
                    "evidenceAvailable": data.get("evidence_available", False),
                    "evidenceReason": data.get("evidence_reason", ""),
                    "answerGrounded": data.get("answer_grounded", False),
                    "groundingReason": data.get("grounding_reason", ""),
                    "groundedAnswer": data.get("grounded_answer", ""),
                    "isAbstained": data.get("is_abstained", False),
                    "abstentionReason": data.get("abstention_reason", ""),
                    "tokenCount": token_count,
                    "wasTrimmed": was_trimmed,
                    "timestamp": datetime.now().strftime("%I:%M %p")
                }
                
                saved_thread = thread_store.append_turn(
                    thread_id=thread_id,
                    user_msg=user_msg_entry,
                    assistant_msg=assistant_msg_entry,
                    token_count=token_count,
                    was_trimmed=was_trimmed
                )
                
                if not effective_dev_mode:
                    # Public consensus schema: strictly omit worker drafts, CoT reasoning, and debate resolution notes
                    payload_out = {
                        "final_response": data.get("final_response", ""),
                        "evidence_available": data.get("evidence_available", False),
                        "answer_grounded": data.get("answer_grounded", False),
                        "is_abstained": data.get("is_abstained", False),
                        "thread_id": thread_id,
                        "thread_title": saved_thread.get("title", "New Chat"),
                        "uploaded_doc": effective_uploaded_doc,
                        "target_doc": data.get("target_doc", "")
                    }
                else:
                    # Diagnostic schema: provide full transparency for authorized developer requests
                    payload_out = {
                        "final_response": data.get("final_response", ""),
                        "arbiter_debate": data.get("arbiter_debate", ""),
                        "ollama_output": data.get("ollama_output", ""),
                        "groq_output": data.get("groq_output", ""),
                        "evidence_available": data.get("evidence_available", False),
                        "evidence_reason": data.get("evidence_reason", ""),
                        "answer_grounded": data.get("answer_grounded", False),
                        "grounding_reason": data.get("grounding_reason", ""),
                        "grounded_answer": data.get("grounded_answer", ""),
                        "is_abstained": data.get("is_abstained", False),
                        "abstention_reason": data.get("abstention_reason", ""),
                        "token_count": token_count,
                        "was_trimmed": was_trimmed,
                        "thread_id": thread_id,
                        "thread_title": saved_thread.get("title", "New Chat"),
                        "uploaded_doc": effective_uploaded_doc,
                        "target_doc": data.get("target_doc", "")
                    }
                yield f"event: consensus\ndata: {json.dumps(payload_out)}\n\n"
            elif event_type == "error":
                yield f"event: error\ndata: {json.dumps({'error': data})}\n\n"
            elif event_type == "done":
                yield f"event: done\ndata: {json.dumps({'status': 'complete', 'thread_id': thread_id})}\n\n"
                break
                
        await task

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host=config.HOST, port=config.PORT, reload=True)
