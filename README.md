# Multi-Agent Knowledge Graph Builder

A production-ready multi-agent pipeline that extracts entities from user input, fetches real-time web context, constructs a knowledge graph, and produces an executive analysis summary with agreement status.

**✨ Features**: Dynamic entity types, comprehensive logging, intelligent caching, structured error handling, and performance monitoring.

---

## Overview

The system runs four agents in sequence:

**EntityAgent → FetchAgent → KnowledgeDBAgent → JudgeAgent**

- **EntityAgent** - Extracts named entities with dynamically determined types (not limited to predefined categories)
- **FetchAgent** - Retrieves Wikipedia summaries and Google News items with intelligent caching
- **KnowledgeDBAgent** - Enriches entities and builds a knowledge graph with relationships, saved to Neo4j
- **JudgeAgent** - Evaluates agreement across sources and returns an executive summary with search suggestions

### Key Features

- 🎯 **Dynamic Entity Extraction**: Automatically determines entity types based on context (Organization, Person, Technology, Event, Concept, etc.)
- 💾 **Intelligent Caching**: 5-10x faster responses for repeated queries with in-memory or Redis caching
- 🔍 **Comprehensive Logging**: Structured JSON/console logging with performance metrics and cache statistics
- ⚡ **Performance Monitoring**: Automatic timing for all operations with detailed metrics
- 🛡️ **Production Ready**: Configurable via environment variables, graceful error handling, distributed caching support

---

## Prerequisites

- Python 3.8+ installed
- `uv` package manager installed
- A running Neo4j instance (local or container)
- API keys: Google API key and Groq API key

---

## Installation

### 1. Install Dependencies

Install `uv` (follow your OS-specific steps), then create and activate a virtual environment:

```bash
# Create and activate virtual environment with uv
uv venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

# Install required packages
uv pip install google-adk litellm wikipediaapi aiohttp certifi beautifulsoup4 neo4j pydantic cachetools
```

**Optional**: For Redis caching support:
```bash
uv pip install redis
```

### 2. Environment Setup

Create a `.env` file in the project root with the required keys:

```env
# Required API Keys
GOOGLE_GENAI_USE_VERTEXAI=FALSE
GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# Logging Configuration (Optional)
LOG_LEVEL=INFO                    # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT=console                # console or json
# LOG_FILE=/var/log/agent.log    # Optional file output

# Caching Configuration (Optional)
CACHE_ENABLED=true                # Enable/disable caching
CACHE_BACKEND=memory              # memory or redis
# REDIS_URL=redis://localhost:6379/0  # If using Redis
ENTITY_CACHE_TTL=3600             # 1 hour
WEB_FETCH_CACHE_TTL=1800          # 30 minutes
KNOWLEDGE_GRAPH_CACHE_TTL=3600    # 1 hour
MAX_CACHE_SIZE=1000               # Max items for in-memory cache

# Neo4j Configuration (Optional)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

> **Note:** The JudgeAgent and KnowledgeDBAgent expect `GROQ_API_KEY` and will raise an error if it is missing.

### 3. Neo4j Setup

- Use a local Neo4j install or a container and configure credentials to match your environment variables
- Ensure the Neo4j port is available (default 7687) for driver connections

---

## Project Structure

```
context_agent_app/
├── agent.py                      # Main orchestrator
├── config.py                     # Centralized configuration
├── utils.py                      # Shared utilities
├── logging_config.py             # Logging infrastructure
├── cache.py                      # Caching layer
└── subagents/
    ├── entity_agent/
    │   └── agent.py              # Dynamic entity extraction
    ├── fetch_agent/
    │   ├── agent.py              # Web fetching with caching
    │   └── tools/
    │       └── web_fetch_tool.py
    ├── knowledgeDB_agent/
    │   ├── agent.py              # Knowledge graph construction
    │   └── tools/
    │       └── neo4j.py
    └── judge_agent/
        └── agent.py              # Analysis and summarization
```

---

## How It Works

### Architecture

1. **EntityAgent** uses LLM to extract entities with context-appropriate types (not limited to 4 predefined types)
2. **FetchAgent** checks cache first, then fetches from Wikipedia API and Google News RSS
3. **KnowledgeDBAgent** re-invokes EntityAgent on combined context, generates relationships with LLM (cached), and persists to Neo4j
4. **JudgeAgent** aggregates all data to return agreement status, summary, and search suggestions

### Performance

- **First query**: 10-20s (fetches from web, generates relationships)
- **Cached query**: 1-2s (**5-10x faster**)
- **Cache hit rate**: Typically 70-90% for repeated entities

### Logging

All operations are logged with:
- Operation timing (duration_ms)
- Cache hit/miss status
- Entity counts and metrics
- Error context and stack traces
- Session ID for request tracing

**Console format** (development):
```
[23:04:15] INFO [agent.FetchAgent] [FetchAgent] Starting web fetch operation
[23:04:15] DEBUG [agent.FetchAgent] [FetchAgent] 💾 Cache hit for entity: Google
[23:04:16] INFO [agent.FetchAgent] [FetchAgent] (1234.56ms) Fetch operation completed
```

**JSON format** (production):
```json
{
  "timestamp": "2025-11-20T17:34:15Z",
  "level": "INFO",
  "agent_name": "FetchAgent",
  "message": "Fetch operation completed successfully",
  "duration_ms": 1234.56,
  "cache_hits": 1,
  "cache_misses": 1
}
```

---

## Usage

### Web Interface

Activate the environment, ensure `.env` is set, and run:

```bash
adk web --port 1111 --reload
```

Then open your browser to the provided URL and submit queries.

### Programmatic Usage

```python
import asyncio
from context_agent_app.agent import runner, USER_ID, SESSION_ID, set_user_query
from google.genai import types

statements = [
    "Apple is developing new AI chips",
    "Google DeepMind created Gemini",
    "Microsoft invested in OpenAI",
]
set_user_query(statements)

content = types.Content(
    role="user",
    parts=[types.Part(text="\\n".join(statements))]
)

async def run():
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=content
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(f"[{event.author}] {part.text}")

asyncio.run(run())
```

### Cache Management

```python
from context_agent_app.cache import get_cache_manager

# Get cache statistics
manager = get_cache_manager()
stats = manager.get_all_stats()
print(stats)

# Clear all caches
manager.clear_all()
```

---

## Output Format

The session state contains entities, fetched context, knowledge graph, and judge result:

```json
{
  "entities": {
    "entities": [
      {
        "name": "Apple",
        "type": "Organization"
      },
      {
        "name": "WWDC",
        "type": "Event"
      },
      {
        "name": "Neural Engine",
        "type": "Technology"
      }
    ]
  },
  "fetched_context": [
    {
      "entity": "Apple",
      "wikipedia": {
        "summary": "...",
        "url": "..."
      },
      "news": [
        {
          "title": "...",
          "link": "...",
          "published": "..."
        }
      ]
    }
  ],
  "knowledge_graph": {
    "nodes": [
      {
        "name": "Apple",
        "type": "Organization",
        "summary": "A multinational technology company..."
      }
    ],
    "relationships": [
      {
        "from_node": "Apple",
        "to_node": "Neural Engine",
        "type": "develops"
      }
    ]
  },
  "judge_result": {
    "agreement_status": "Agree",
    "summary": "...",
    "search_suggestions": ["Apple AI chip roadmap", "Neural Engine benchmarks"]
  }
}
```

---

## Configuration

All configuration is centralized in `config.py` and can be overridden via environment variables:

### LLM Models
```python
ENTITY_EXTRACTION_MODEL = "groq/llama-3.3-70b-versatile"
JUDGE_MODEL = "groq/llama-3.3-70b-versatile"
KNOWLEDGE_GRAPH_MODEL = "groq/llama-3.3-70b-versatile"
```

### Cache TTLs
- Entity extraction: 1 hour
- Web fetch: 30 minutes
- Knowledge graph relationships: 1 hour

### Logging Levels
- DEBUG: Detailed cache hits/misses, LLM calls
- INFO: Operation timing, cache statistics (default)
- WARNING: Missing data, fallbacks
- ERROR: Failures with context

---

## Troubleshooting

### Common Issues

**Missing `GROQ_API_KEY`**
- Error: `ValueError: GROQ_API_KEY environment variable not set`
- Solution: Add `GROQ_API_KEY=your_key` to `.env`

**No entities found**
- FetchAgent reports and stores empty context to keep the run stable
- Check EntityAgent logs for extraction issues

**SSL or parsing issues**
- Handled with `certifi` and BeautifulSoup's HTML parser
- Check logs for specific error messages

**Neo4j connection failures**
- Ensure Neo4j is running: `docker run -p 7687:7687 -p 7474:7474 neo4j`
- Verify credentials match `NEO4J_*` environment variables

**Cache not working**
- Check `CACHE_ENABLED=true` in `.env`
- For Redis: Ensure Redis is running and `REDIS_URL` is correct
- Check logs for cache initialization messages

### Debug Mode

Enable detailed logging:
```bash
export LOG_LEVEL=DEBUG
export LOG_FORMAT=console
adk web --port 1111 --reload
```

---

## Dependencies

### Core
- **google-adk** - Agent orchestration and sessions
- **litellm** - LLM calls to Groq models
- **pydantic** - Data models and validation

### Web Fetching
- **wikipediaapi** - Wikipedia content retrieval
- **aiohttp** - Async HTTP requests
- **beautifulsoup4** - HTML parsing
- **certifi** - SSL certificate handling

### Storage
- **neo4j** - Graph database driver

### Caching & Logging
- **cachetools** - In-memory caching with TTL
- **redis** (optional) - Distributed caching

---

## Performance Benchmarks

### Without Caching
- Entity extraction: ~500-1000ms
- Web fetch (3 entities): ~6-15s
- Knowledge graph: ~2-3s
- **Total**: ~10-20s

### With Caching (Warm Cache)
- Entity extraction: ~500-1000ms
- Web fetch (3 entities): ~30-150ms (cached)
- Knowledge graph: ~50-100ms (cached)
- **Total**: ~1-2s (**5-10x faster**)

### Cache Statistics Example
```json
{
  "web_fetch": {
    "hits": 15,
    "misses": 5,
    "hit_rate": 0.75,
    "total_requests": 20
  },
  "knowledge_graph": {
    "hits": 8,
    "misses": 2,
    "hit_rate": 0.80,
    "total_requests": 10
  }
}
```

---

## Advanced Features

### Dynamic Entity Types

Unlike traditional NER systems limited to predefined types, this system automatically determines appropriate entity types:

- **Standard**: Organization, Person, Technology, Location
- **Extended**: Event, Concept, Product, Date/Time, Methodology, etc.
- **Custom**: Any contextually appropriate type

Example:
```json
{
  "name": "WWDC",
  "type": "Event"
},
{
  "name": "Transformer Architecture",
  "type": "Concept"
}
```

### Distributed Caching with Redis

For multi-instance deployments:

```bash
# Start Redis
docker run -d -p 6379:6379 redis

# Configure app
export CACHE_BACKEND=redis
export REDIS_URL=redis://localhost:6379/0
```

### Structured Logging for Production

```bash
# Enable JSON logging for log aggregation
export LOG_FORMAT=json
export LOG_FILE=/var/log/agent.log

# Send to ELK/Splunk/CloudWatch
tail -f /var/log/agent.log | your-log-shipper
```

---

## Contributing

When adding new features:

1. Add configuration to `config.py`
2. Use `AgentLogger` for logging
3. Use `SessionKeys` constants for session state
4. Add caching for expensive operations
5. Include performance metrics in logs

---

## License

[Your License Here]

---

## Notes

- The system is designed to handle failures gracefully and continue processing even when some data sources are unavailable
- All agents use structured error handling with full context logging
- Cache can be disabled entirely by setting `CACHE_ENABLED=false`
- Logging format can be switched between console and JSON without code changes
- All configuration is environment-variable driven for easy deployment