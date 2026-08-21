"""
Application Settings & Environment Configuration.

Uses pydantic-settings to parse environment variables from a .env file or
the system environment, providing typed settings across the application.
"""

from typing import List, Union
import sys
from pydantic import field_validator, model_validator, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central application configuration model.
    Loads and validates runtime environment variables.
    """
    PROJECT_NAME: str = "DocMind AI API"
    VERSION: str = "1.0.0"
    BACKEND_VERSION: str = "1.0.0"
    ENV: str = "development"
    HOST: str = "127.0.0.1"
    PORT: int = 8000

    # Environment variable configurations
    GROQ_API_KEY: str
    GROQ_MODEL: str = "openai/gpt-oss-20b"
    GROQ_TPM_LIMIT: int = 8000  # Groq rolling TPM quota ceiling
    GROQ_PREFLIGHT_PROMPT_TOKEN_BUDGET: int = 2500  # Max estimated prompt tokens before calling Groq
    GROQ_QUERY_REWRITE_RESERVE_TOKENS: int = 300  # Input/output allowance for query rewriting
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    UPLOAD_DIRECTORY: str = "uploads"
    MAX_UPLOAD_SIZE: int = 20971520
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K: int = 25
    LOG_LEVEL: str = "INFO"

    # Internal mapped config settings
    LLM_MODEL: str = "openai/gpt-oss-20b"
    DEFAULT_TOP_K: int = 25

    # PDF Upload Configuration Settings
    UPLOAD_STREAM_CHUNK_BYTES: int = 1048576  # 1 MB buffer for streaming chunk writes

    # Text Chunking Engine Configuration Settings
    CHUNK_VERSION: str = "1.0"  # Version identifier for chunking engine schema
    TOKEN_ESTIMATION_RATIO: int = 4  # Characters per estimated token ratio

    # Embedding Engine Configuration Settings
    EMBEDDING_DIMENSION: int = 384  # Embedding vector output dimension
    EMBEDDING_NORMALIZE: bool = True  # Enable L2 normalization for cosine similarity compatibility
    EMBEDDING_BATCH_SIZE: int = 32  # Number of texts processed concurrently during inference
    EMBEDDING_VERSION: str = "1.0"  # Version of the embedding schema

    # Vector Indexing Engine Configuration Settings
    INDEX_TYPE: str = "IndexFlatIP"  # FAISS index type. IP stands for Inner Product (Cosine similarity when normalized)
    INDEX_VERSION: str = "1.0"  # Version of the indexing schema
    VECTOR_DISTANCE: str = "cosine"  # Metric distance measurement algorithm

    # Semantic Retrieval Engine Configuration Settings
    DEFAULT_TOP_K: int = 30  # Default candidate chunks to retrieve per search query
    INITIAL_RETRIEVAL_TOP_K: int = 35  # Generous candidate pool pulled prior to reranking & token-budget trimming
    MAX_TOP_K: int = 50  # Maximum allowed chunks to retrieve in a single query
    MIN_SIMILARITY_SCORE: float = 0.40  # Minimum vector similarity threshold score
    ADAPTIVE_SCORE_DROP_LIMIT: float = 0.45  # Limit cutoff difference for adaptive top-k pruning (preserves diverse multi-match candidates)
    MAX_QUERY_LENGTH: int = 2000  # Maximum length of user query query string to prevent resource exhaustion
    ENABLE_NEIGHBOR_MERGING: bool = True  # Enable adjacent chunk merging
    MAX_MERGED_CHUNKS: int = 2  # Max consecutive chunks allowed in a merged block
    MAX_MERGED_CHUNK_CHARS: int = 1500  # Max total characters allowed in a merged block
    ENABLE_HYBRID_SEARCH: bool = True  # Enable BM25 + Vector Hybrid Retrieval
    BM25_K1: float = 1.5  # Term frequency saturation parameter
    BM25_B: float = 0.75  # Length normalization parameter
    BM25_DEFAULT_SIMILARITY_SCORE: float = 0.40  # Default vector similarity score for BM25-only matches
    RRF_K: int = 60  # Reciprocal Rank Fusion constant
    RRF_FUSED_SCORE_FLOOR: float = 0.45  # Score floor for top hybrid RRF matches
    ENABLE_RERANKER: bool = True  # Enable cross-scoring reranking pass
    RERANKER_TOP_K: int = 35  # Maximum candidate chunks to retain after reranking
    ENABLE_SELECTIVE_QUERY_REWRITING: bool = False  # Enable selective query rewriting filter
    ENABLE_BM25_SEARCH: bool = True  # Enable BM25 lexical keyword search alongside vector search
    REWRITE_MIN_WORD_COUNT: int = 5  # Minimum word count required to trigger query rewriter LLM
    QUERY_REWRITE_TIMEOUT_SECONDS: int = 10  # Timeout in seconds for query rewrite LLM call
    QUERY_REWRITE_MAX_TOKENS: int = 100  # Max completion tokens for query rewriter
    QUERY_REWRITE_TEMPERATURE: float = 0.3  # Temperature for query expansion LLM
    RETRIEVAL_VERSION: str = "1.0"  # Version of the retrieval engine schema

    # AI Chat (RAG) Engine Configuration Settings
    LLM_PROVIDER: str = "groq"  # Default large language model API provider
    LLM_TEMPERATURE: float = 0.2  # Control generation determinism
    LLM_MAX_TOKENS: int = 1024  # Maximum completion token length (supports reasoning tokens + full response)
    GROQ_PREFLIGHT_HEADROOM_FLOOR: int = 1000  # Minimum token headroom floor for prompt assembly
    GROQ_COMPLETION_RESERVE_TOKENS: int = 1024  # Reserve tokens allocated for completion generation
    GROQ_MAX_QUEUE_WAIT_SECONDS: float = 25.0  # Maximum seconds to queue in-memory before returning 429
    GROQ_SOFT_CAP_RATIO: float = 0.85  # Soft cap threshold (85% of token window) to return graceful busy message
    GROQ_BUSY_MESSAGE: str = "This demo is currently busy with other visitors — please try again in about a minute."
    ENABLE_RESPONSE_CACHE: bool = True  # Enable in-memory LRU response caching for identical/repeated queries
    RESPONSE_CACHE_MAX_ENTRIES: int = 200  # Maximum cached query entries per server process
    CONTEXT_TRIM_DECAY_RATE: float = 0.9  # Truncation ratio applied during iterative context trimming
    MAX_CONTEXT_CHUNKS: int = 30  # Maximum number of document chunks allowed in prompt context (supports comprehensive multi-matches)
    MAX_CONTEXT_CHARACTERS: int = 24000  # Upper context size characters limit
    MAX_CONTEXT_LENGTH: int = 24000  # Configurable upper context character limit
    LLM_TIMEOUT: int = 30  # Timeout threshold in seconds for API completions call
    CHAT_VERSION: str = "1.0"  # Chat engine version identifier
    SYSTEM_PROMPT_VERSION: str = "1.0"  # Core system prompt version
    ENABLE_TOKEN_ESTIMATION: bool = True  # Enable estimated token metrics tracking
    ENABLE_REQUEST_LOGGING: bool = True  # Observability logger flag

    # Context & Memory Configuration Settings
    MODEL_CONTEXT_WINDOW: int = 131072  # Model context window in tokens
    CONVERSATION_MAX_TURNS: int = 2  # Max conversation turns to keep in memory per session (capped to prevent compounding token growth)
    CONVERSATION_MAX_TOKENS: int = 350  # Token budget reserved for conversation history

    # API Rate Limiting Configuration Settings
    GENERAL_REQUEST_LIMIT: int = 100  # Default general request limit per client IP
    GENERAL_REQUEST_WINDOW_SECONDS: int = 60  # Default time window in seconds
    EXPENSIVE_REQUEST_LIMIT: int = 30  # Stricter request limit for expensive/LLM endpoints (aligned with Groq 30 RPM)
    EXPENSIVE_REQUEST_WINDOW_SECONDS: int = 60  # Time window for expensive/LLM endpoints

    # Resource Safety Limit Settings
    MAX_PDF_PAGES: int = 100
    MAX_EXTRACTED_TEXT_SIZE: int = 1000000
    MAX_CHUNKS: int = 2000

    # Debug Log Telemetry Configuration Settings
    DEBUG_LOG_MAX_ENTRIES: int = 100  # Default maximum entries per document debug log
    DEBUG_LOG_MAX_SIZE_MB: float = 1.0  # Default maximum size in MB per document debug log

    # CORS configuration - default allowed origins for local frontend development
    CORS_ORIGINS: Union[str, List[str]] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

    @field_validator("GROQ_API_KEY", mode="after")
    @classmethod
    def validate_groq_api_key(cls, v: str) -> str:
        """
        Validates that GROQ_API_KEY is not empty or equal to the default placeholder.
        """
        if not v or v.strip() == "" or v.strip() == "your_groq_api_key_here":
            raise ValueError("GROQ_API_KEY is missing or contains placeholder values.")
        return v.strip()

    @field_validator("CORS_ORIGINS", mode="before")
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        """
        Parses comma-separated string into a list of origins if passed via .env.
        """
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",") if i.strip()]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    @model_validator(mode="before")
    @classmethod
    def map_legacy_settings(cls, data: dict) -> dict:
        """
        Maps env-provided variables into internal config fields.
        """
        if "GROQ_MODEL" in data:
            data["LLM_MODEL"] = data["GROQ_MODEL"]
        if "TOP_K" in data:
            data["DEFAULT_TOP_K"] = int(data["TOP_K"])
        if "MAX_CONTEXT_LENGTH" in data:
            data["MAX_CONTEXT_CHARACTERS"] = int(data["MAX_CONTEXT_LENGTH"])
            data["MAX_CONTEXT_LENGTH"] = int(data["MAX_CONTEXT_LENGTH"])
        if "BACKEND_VERSION" in data:
            data["VERSION"] = data["BACKEND_VERSION"]
        return data

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )


# Instantiated settings object to be imported across the application
try:
    settings = Settings()
except ValidationError as exc:
    print("\n" + "="*80, file=sys.stderr)
    print("CRITICAL CONFIGURATION ERROR: Failed to validate environment settings.", file=sys.stderr)
    print("="*80, file=sys.stderr)
    for error in exc.errors():
        loc = ".".join(str(l) for l in error.get("loc", []))
        msg = error.get("msg")
        print(f" - {loc}: {msg}", file=sys.stderr)
    print("\nHelp: Please verify that you have copied backend/.env.example to backend/.env", file=sys.stderr)
    print("and set a valid GROQ_API_KEY variable value.\n" + "="*80 + "\n", file=sys.stderr)
    sys.exit(1)
