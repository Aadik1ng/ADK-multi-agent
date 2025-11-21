"""
Configuration for the multi-agent context understanding system.
"""
import os

# =========================
# LLM Configuration
# =========================
DEFAULT_MODEL = "groq/llama-3.3-70b-versatile"
ENTITY_EXTRACTION_MODEL = os.getenv("ENTITY_MODEL", DEFAULT_MODEL)
JUDGE_MODEL = os.getenv("JUDGE_MODEL", DEFAULT_MODEL)
KNOWLEDGE_GRAPH_MODEL = os.getenv("KNOWLEDGE_GRAPH_MODEL", DEFAULT_MODEL)

# LLM Parameters
DEFAULT_TEMPERATURE = 0.3
ENTITY_TEMPERATURE = 0.2  # Lower for more consistent entity extraction
JUDGE_TEMPERATURE = 0.3
KNOWLEDGE_GRAPH_TEMPERATURE = 0.3

# =========================
# Session State Keys
# =========================
class SessionKeys:
    """Constants for session state keys to avoid magic strings."""
    USER_NAME = "user_name"
    USER_QUERY = "user_query"
    INTERACTION_HISTORY = "interaction_history"
    ENTITIES = "entities"
    FETCHED_CONTEXT = "fetched_context"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    JUDGE_RESULT = "judge_result"
    FINAL_SUMMARY = "final_summary"

# =========================
# Application Settings
# =========================
APP_NAME = "AgreeGraph"
DEFAULT_USER_ID = "Aaditya"

# =========================
# Agent Settings
# =========================
MAX_ENTITIES_PER_QUERY = 20
MAX_NEWS_ARTICLES_PER_ENTITY = 3
WEB_FETCH_TIMEOUT = 30  # seconds

# =========================
# Neo4j Settings (if needed)
# =========================
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# =========================
# Logging Configuration
# =========================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("LOG_FORMAT", "console")  # "json" or "console"
LOG_FILE = os.getenv("LOG_FILE", None)  # Optional file output

# =========================
# Cache Configuration
# =========================
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
CACHE_BACKEND = os.getenv("CACHE_BACKEND", "memory")  # "memory" or "redis"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Cache TTL (time-to-live) in seconds
ENTITY_CACHE_TTL = int(os.getenv("ENTITY_CACHE_TTL", "3600"))  # 1 hour
WEB_FETCH_CACHE_TTL = int(os.getenv("WEB_FETCH_CACHE_TTL", "1800"))  # 30 minutes
LLM_CACHE_TTL = int(os.getenv("LLM_CACHE_TTL", "7200"))  # 2 hours
KNOWLEDGE_GRAPH_CACHE_TTL = int(os.getenv("KNOWLEDGE_GRAPH_CACHE_TTL", "3600"))  # 1 hour

# Cache size limits (for in-memory cache)
MAX_CACHE_SIZE = int(os.getenv("MAX_CACHE_SIZE", "1000"))  # Max items per cache

