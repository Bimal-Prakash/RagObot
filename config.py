"""
RagObot Production Configuration.
Structured schema validation powered by Pydantic v2 BaseSettings.
Loads environment variables from .env or system environment with strict typing.
"""
from pathlib import Path
from typing import Any
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Core Directories & Environment
    ENVIRONMENT: str = Field(default="development", description="Deployment environment ('development', 'staging', 'production')")
    HOST: str = Field(default="127.0.0.1", description="Server host address to bind")
    PORT: int = Field(default=8000, description="Server port to bind")
    ALLOWED_ORIGINS: list[str] = Field(
        default=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000", "http://127.0.0.1:3000"],
        description="Allowed CORS origins list"
    )
    BASE_DIR: Path = Path(__file__).resolve().parent
    DATA_DIR: Path = Path("./data")
    CHROMA_DIR: Path = Path("./data/chroma_db")

    # Cloud LLM Settings (Groq)
    GROQ_API_KEY: str = Field(default="", description="Optional Groq API key for cloud LPU inference")
    GROQ_MODEL: str = Field(default="llama-3.3-70b-versatile", description="Groq cloud reasoning model")

    # Local / Cloud LLM Settings (Ollama)
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434", description="Ollama server endpoint")
    OLLAMA_MODEL: str = Field(default="llama3.2", description="Default model executed via Ollama")
    EMBEDDING_MODEL: str = Field(default="nomic-embed-text", description="ChromaDB dense embedding model")

    # Multi-Agent Coordination & Retrieval
    ARBITER_ENGINE: str = Field(default="groq", description="Consensus arbiter engine ('groq' or 'ollama')")
    MAX_CONTEXT_TOKENS: int = Field(default=3500, description="Rolling context window token limit")
    TOP_K_CHUNKS: int = Field(default=5, description="Semantic chunks retrieved per user query")
    CHUNK_SIZE: int = Field(default=1000, description="Character size of text chunks for dense retrieval")
    CHUNK_OVERLAP: int = Field(default=150, description="Character overlap between consecutive chunks")

    # Security & Developer Access Control
    DEV_MODE_KEY: str = Field(default="", description="Secret key for developer diagnostic access")
    ALLOW_UNAUTHENTICATED_DEV_MODE: bool = Field(default=False, description="Allow dev_mode without key for local development")

    # Native Speech Pipeline (Neural TTS with SAPI fallback)
    TTS_ENABLED: bool = Field(default=True, description="Enable local speech synthesis")
    TTS_VOICE: str = Field(default="en-US-JennyNeural", description="Neural Voice (en-US-JennyNeural, en-US-GuyNeural, en-US-AriaNeural)")
    TTS_RATE: int = Field(default=0, description="Speech rate offset (-10 to +10)")
    TTS_VOLUME: int = Field(default=100, description="Speech volume (0 to 100)")

    @model_validator(mode="before")
    @classmethod
    def parse_allowed_origins(cls, data: Any) -> Any:
        """Parses comma-separated string format for ALLOWED_ORIGINS into list of URLs."""
        if isinstance(data, dict):
            origins = data.get("ALLOWED_ORIGINS")
            if isinstance(origins, str):
                data["ALLOWED_ORIGINS"] = [o.strip() for o in origins.split(",") if o.strip()]
        return data

    @model_validator(mode="after")
    def validate_production_security(self) -> 'AppConfig':
        """Enforces that unauthenticated developer mode cannot be enabled in production deployments."""
        if str(self.ENVIRONMENT).strip().lower() == "production":
            if self.ALLOW_UNAUTHENTICATED_DEV_MODE:
                raise ValueError(
                    "CRITICAL SECURITY CONFIGURATION ERROR: ALLOW_UNAUTHENTICATED_DEV_MODE cannot be True in production!"
                )
        return self

# Global Singleton Instance
settings = AppConfig()

# Clean module-level exports for backward compatibility across all imports
ENVIRONMENT = settings.ENVIRONMENT
HOST = settings.HOST
PORT = settings.PORT
ALLOWED_ORIGINS = settings.ALLOWED_ORIGINS
BASE_DIR = settings.BASE_DIR
DATA_DIR = settings.DATA_DIR
CHROMA_DIR = settings.CHROMA_DIR
GROQ_API_KEY = settings.GROQ_API_KEY
GROQ_MODEL = settings.GROQ_MODEL
OLLAMA_BASE_URL = settings.OLLAMA_BASE_URL
OLLAMA_MODEL = settings.OLLAMA_MODEL
EMBEDDING_MODEL = settings.EMBEDDING_MODEL
ARBITER_ENGINE = settings.ARBITER_ENGINE
MAX_CONTEXT_TOKENS = settings.MAX_CONTEXT_TOKENS
TOP_K_CHUNKS = settings.TOP_K_CHUNKS
CHUNK_SIZE = settings.CHUNK_SIZE
CHUNK_OVERLAP = settings.CHUNK_OVERLAP
DEV_MODE_KEY = settings.DEV_MODE_KEY
ALLOW_UNAUTHENTICATED_DEV_MODE = settings.ALLOW_UNAUTHENTICATED_DEV_MODE
TTS_ENABLED = settings.TTS_ENABLED
TTS_VOICE = settings.TTS_VOICE
TTS_RATE = settings.TTS_RATE
TTS_VOLUME = settings.TTS_VOLUME
