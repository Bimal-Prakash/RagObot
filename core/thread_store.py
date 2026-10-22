import os
import json
import uuid
import time
from pathlib import Path
from typing import Any, Optional
import config

THREADS_DIR = config.BASE_DIR / "data" / "threads"
THREADS_DIR.mkdir(parents=True, exist_ok=True)

def generate_thread_title(query: str, max_length: int = 40) -> str:
    """Generates a concise, clean title from the initial user query."""
    clean = " ".join(query.strip().split())
    if not clean:
        return "New Chat"
    # Remove leading question words or markdown if needed
    clean = clean.strip("?\"'#* ")
    if len(clean) <= max_length:
        return clean
    # Truncate at word boundary
    truncated = clean[:max_length].rsplit(" ", 1)[0]
    return (truncated or clean[:max_length]).strip() + "..."

class ThreadStore:
    """Persistent JSON-based conversation thread manager."""

    def __init__(self, storage_dir: Path = THREADS_DIR):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, thread_id: str) -> Path:
        # Sanitize thread_id to prevent directory traversal
        safe_id = "".join(c for c in thread_id if c.isalnum() or c in ("-", "_"))
        return self.storage_dir / f"{safe_id}.json"

    def list_threads(self) -> list[dict[str, Any]]:
        """Returns all saved threads sorted by updated_at descending."""
        threads = []
        for file in self.storage_dir.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    threads.append({
                        "id": data.get("id", file.stem),
                        "title": data.get("title", "Untitled Chat"),
                        "created_at": data.get("created_at", 0),
                        "updated_at": data.get("updated_at", 0),
                        "message_count": len(data.get("messages", [])),
                        "token_count": data.get("token_count", 0),
                        "was_trimmed": data.get("was_trimmed", False),
                        "last_uploaded_doc": data.get("last_uploaded_doc")
                    })
            except Exception:
                continue

        threads.sort(key=lambda t: t.get("updated_at", 0), reverse=True)
        return threads

    def get_thread(self, thread_id: str) -> Optional[dict[str, Any]]:
        """Loads complete thread with full message history."""
        path = self._get_path(thread_id)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def create_thread(self, thread_id: Optional[str] = None, title: str = "New Chat") -> dict[str, Any]:
        """Creates a new empty thread and persists it."""
        t_id = thread_id or str(uuid.uuid4())
        now = int(time.time() * 1000)
        thread_data = {
            "id": t_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
            "token_count": 0,
            "was_trimmed": False,
            "last_uploaded_doc": None,
            "messages": []
        }
        self.save_thread(thread_data)
        return thread_data

    def set_thread_uploaded_doc(self, thread_id: str, doc_name: str) -> bool:
        """Associates the latest uploaded document with a thread."""
        thread = self.get_thread(thread_id)
        if not thread:
            thread = self.create_thread(thread_id=thread_id)
        thread["last_uploaded_doc"] = doc_name
        thread["updated_at"] = int(time.time() * 1000)
        return self.save_thread(thread)

    def get_thread_uploaded_doc(self, thread_id: str) -> Optional[str]:
        """Retrieves the latest uploaded document for a thread."""
        thread = self.get_thread(thread_id)
        if thread:
            return thread.get("last_uploaded_doc")
        return None

    def save_thread(self, thread_data: dict[str, Any]) -> bool:
        """Saves thread data atomically to disk."""
        thread_id = thread_data.get("id")
        if not thread_id:
            return False
        path = self._get_path(thread_id)
        temp_path = path.with_suffix(".tmp")
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(thread_data, f, indent=2, ensure_ascii=False)
            temp_path.replace(path)
            return True
        except Exception:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            return False

    def update_thread_title(self, thread_id: str, title: str) -> bool:
        """Updates the title of an existing thread."""
        thread = self.get_thread(thread_id)
        if not thread:
            return False
        thread["title"] = title.strip() or "Untitled Chat"
        thread["updated_at"] = int(time.time() * 1000)
        return self.save_thread(thread)

    def delete_thread(self, thread_id: str) -> bool:
        """Deletes a thread file from disk."""
        path = self._get_path(thread_id)
        if path.exists():
            try:
                path.unlink()
                return True
            except Exception:
                return False
        return False

    def append_turn(
        self,
        thread_id: str,
        user_msg: dict[str, Any],
        assistant_msg: dict[str, Any],
        token_count: int = 0,
        was_trimmed: bool = False
    ) -> dict[str, Any]:
        """Appends a user/assistant turn to thread and auto-updates title if first query."""
        thread = self.get_thread(thread_id)
        if not thread:
            thread = self.create_thread(thread_id=thread_id)

        # Auto-generate title from first user message if current title is default
        if (not thread.get("messages") or thread.get("title") in ("New Chat", "Untitled Chat")) and user_msg.get("content"):
            thread["title"] = generate_thread_title(user_msg["content"])

        thread["messages"].append(user_msg)
        thread["messages"].append(assistant_msg)
        thread["token_count"] = token_count
        thread["was_trimmed"] = was_trimmed
        thread["updated_at"] = int(time.time() * 1000)
        self.save_thread(thread)
        return thread

thread_store = ThreadStore()
