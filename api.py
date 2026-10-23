import asyncio
import json
import uuid
import urllib.request
import shutil
import os
import time
import hmac
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Optional
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
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

# In-memory session tracking
session_state: dict[str, Any] = {
    "thread_id": str(uuid.uuid4()),
    "token_count": 0,
    "was_trimmed": False,
    "last_uploaded_doc": None,
    "history": []
}

def is_dev_authorized(
    requested_dev_mode: bool,
    supplied_key: Optional[str] = None,
    header_key: Optional[str] = None
) -> bool:
    """
    Validates whether developer diagnostic mode is authorized on the backend.
    - If requested_dev_mode is False: returns False.
    - If ENVIRONMENT is 'production': strictly enforces DEV_MODE_KEY verification,
      ignoring ALLOW_UNAUTHENTICATED_DEV_MODE to prevent production bypass.
    - If ENVIRONMENT is non-production AND ALLOW_UNAUTHENTICATED_DEV_MODE is True: returns True.
    - Otherwise: requires configured DEV_MODE_KEY matching supplied_key or X-Dev-Key header.
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

    token = (supplied_key or header_key or "").strip()
    return hmac.compare_digest(token, configured_key)

class QueryRequest(BaseModel):
    query: str
    thread_id: str | None = None
    uploaded_doc: str | None = None
    trigger_backend_tts: bool = False  # False for web clients (browser plays audio), True for terminal
    model_mode: str = "dual"  # "dual", "local", "cloud"
    local_model: str | None = None
    cloud_model: str | None = None
    dev_mode: bool = False  # If True, requests developer diagnostic mode; must be authorized by backend
    dev_key: str | None = None  # Optional secret developer authorization key

class GroqSettingsRequest(BaseModel):
    api_key: str | None = None
    model: str | None = None

class SynthesizeRequest(BaseModel):
    text: str
    voice: str | None = None

class ThreadTitleRequest(BaseModel):
    title: str

class CreateThreadRequest(BaseModel):
    title: str | None = "New Chat"

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

@app.get("/api/models")
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

@app.post("/api/settings/groq")
async def update_groq_settings(req: GroqSettingsRequest):
    """Saves Groq API Key and custom model directly into .env and updates in-memory config."""
    try:
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
            "groq_model": config.GROQ_MODEL,
            "message": "Groq settings saved to .env"
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/api/status")
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

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), thread_id: Optional[str] = Form(None)):
    """Receives an uploaded PDF, Markdown, or TXT file, saves it into data/ and indexes it."""
    try:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        target_path = config.DATA_DIR / file.filename
        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        new_count = rag_engine.ingest_documents()
        
        # Track last uploaded document in session and thread store
        session_state["last_uploaded_doc"] = file.filename
        if thread_id:
            thread_store.set_thread_uploaded_doc(thread_id, file.filename)
        elif session_state.get("thread_id"):
            thread_store.set_thread_uploaded_doc(session_state["thread_id"], file.filename)

        return {
            "status": "success",
            "filename": file.filename,
            "indexed_chunks": new_count
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

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
            return Response(status_code=400, content="Unable to synthesize audio")
        return Response(content=audio_bytes, media_type="audio/mpeg")
    except Exception as e:
        return Response(status_code=500, content=str(e))

@app.get("/api/threads")
async def list_threads():
    """List all saved conversation threads."""
    return {"threads": thread_store.list_threads()}

@app.get("/api/threads/{thread_id}")
async def get_thread(
    thread_id: str,
    request: Request,
    dev_mode: bool = False
):
    """Get complete conversation thread with messages."""
    thread = thread_store.get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
        
    header_key = request.headers.get("x-dev-key")
    if not header_key and request.headers.get("authorization"):
        auth_hdr = request.headers.get("authorization", "")
        if auth_hdr.startswith("Bearer "):
            header_key = auth_hdr[7:].strip()

    effective_dev_mode = is_dev_authorized(dev_mode, header_key=header_key)

    if not effective_dev_mode and thread.get("messages"):
        sanitized_messages = []
        for msg in thread["messages"]:
            if msg.get("role") == "assistant" and (msg.get("isAbstained") or not msg.get("answerGrounded")):
                sanitized_msg = dict(msg)
                sanitized_msg["ollamaStream"] = "[Worker output redacted: grounding validation failed closed.]"
                sanitized_msg["groqStream"] = "[Worker output redacted: grounding validation failed closed.]"
                sanitized_messages.append(sanitized_msg)
            else:
                sanitized_messages.append(msg)
        thread = dict(thread)
        thread["messages"] = sanitized_messages
    return thread

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
    success = thread_store.update_thread_title(thread_id, req.title)
    if not success:
        raise HTTPException(status_code=404, detail="Thread not found")
    return {"status": "updated", "id": thread_id, "title": req.title}

@app.delete("/api/threads/{thread_id}")
async def delete_thread(thread_id: str):
    """Delete a conversation thread."""
    success = thread_store.delete_thread(thread_id)
    if not success:
        raise HTTPException(status_code=404, detail="Thread not found")
    return {"status": "deleted", "id": thread_id}

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
      - 'ollama': Tokens from Ollama local worker
      - 'groq': Tokens from Groq cloud worker
      - 'consensus': Final synthesis from Arbiter
      - 'diagnostics': Token count and trim alert
    """
    thread_id = payload.thread_id or session_state["thread_id"]
    query = payload.query.strip()
    
    header_dev_key = request.headers.get("x-dev-key")
    effective_dev_mode = is_dev_authorized(payload.dev_mode, payload.dev_key, header_dev_key)

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
                yield f"event: evidence_status\ndata: {json.dumps(data)}\n\n"
            elif event_type == "validation_status":
                yield f"event: validation_status\ndata: {json.dumps(data)}\n\n"
            elif event_type == "graph_result":
                if not graph_retrieved_docs and data.get("retrieved_docs"):
                    graph_retrieved_docs = data.get("retrieved_docs", [])
                    yield f"event: docs\ndata: {json.dumps(graph_retrieved_docs)}\n\n"
                token_count = data.get("token_count", 0)
                was_trimmed = data.get("was_trimmed", False)
                session_state["token_count"] = token_count
                session_state["was_trimmed"] = was_trimmed
                
                is_grounded = bool(data.get("answer_grounded", False)) and not bool(data.get("is_abstained", False))

                if not is_grounded and not effective_dev_mode:
                    safe_ollama_output = "[Worker output redacted: grounding validation failed closed.]"
                    safe_groq_output = "[Worker output redacted: grounding validation failed closed.]"
                else:
                    safe_ollama_output = data.get("ollama_output", "")
                    safe_groq_output = data.get("groq_output", "")

                # Persist turn to thread_store
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
                
                payload_out = {
                    "final_response": data.get("final_response", ""),
                    "arbiter_debate": data.get("arbiter_debate", ""),
                    "ollama_output": safe_ollama_output,
                    "groq_output": safe_groq_output,
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
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
