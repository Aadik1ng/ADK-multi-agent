# Multi-Agent Knowledge Graph Builder

A multi-agent pipeline that extracts entities from user input, fetches real-time web context, constructs a knowledge graph, and produces an executive analysis summary with agreement status.

---

## Overview

The system runs four agents in sequence:

**EntityAgent → FetchAgent → KnowledgeDBAgent → JudgeAgent**

- **EntityAgent** extracts named entities such as people, organizations, technologies, and locations
- **FetchAgent** retrieves Wikipedia summaries and Google News items for the extracted entities
- **KnowledgeDBAgent** enriches and saves a knowledge graph (nodes and relationships) to Neo4j
- **JudgeAgent** evaluates agreement across sources and returns a concise executive summary with search suggestions

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
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install required packages
uv pip install google-adk litellm wikipediaapi aiohttp certifi beautifulsoup4 neo4j pydantic
```

### 2. Environment Setup

Create a `.env` file in the project root with the required keys:

```env
GOOGLE_GENAI_USE_VERTEXAI=FALSE
GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

> **Note:** The JudgeAgent and KnowledgeDBAgent expect `GROQ_API_KEY` and will raise an error if it is missing.

### 3. Neo4j Setup

- Use a local Neo4j install or a container and configure credentials to match your environment variables or tool defaults
- Ensure the Neo4j port is available (default 7687) for driver connections

---

## Project Structure

```
project/
├── agent.py
└── subagents/
    ├── entity_agent/
    │   └── agent.py
    ├── fetch_agent/
    │   ├── agent.py
    │   └── tools/
    │       └── web_fetch_tool.py
    ├── knowledgeDB_agent/
    │   ├── agent.py
    │   └── tools/
    │       └── neo4j.py
    └── judge_agent/
        └── agent.py
```

---

## How It Works

1. `agent.py` wires a `SequentialAgent` that runs EntityAgent → FetchAgent → KnowledgeDBAgent → JudgeAgent with a stateful session
2. Fetching uses Wikipedia API and Google News RSS via `aiohttp`, `BeautifulSoup`, and `wikipediaapi`
3. `KnowledgeDBAgent` re-invokes `EntityAgent` on combined fetched context, then builds relationships with an LLM and persists to Neo4j
4. `JudgeAgent` aggregates entities, fetched context, and the graph to return agreement status, summary, and suggested searches

---

## Usage

### Web Interface

Activate the environment, ensure `.env` is set, and run the orchestrator script:

```bash
adk web
```

### Programmatic Usage

```python
import asyncio
from agent import runner, USER_ID, SESSION_ID, set_user_query
from google.genai import types

statements = [
    "Apple is developing new AI chips",
    "Google DeepMind created Gemini",
    "Microsoft invested in OpenAI",
]
set_user_query(statements)

content = types.Content(
    role="user",
    parts=[types.Part(text="\n".join(statements))]
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

---

## Output Format

The session state contains entities, fetched context, knowledge graph, and judge result:

```json
{
  "entities": [
    {
      "name": "Apple",
      "type": "Organization"
    }
  ],
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
          "link": "..."
        }
      ]
    }
  ],
  "knowledge_graph": {
    "nodes": [
      {
        "name": "Apple",
        "type": "Organization",
        "summary": "..."
      }
    ],
    "relationships": [
      {
        "from_node": "Apple",
        "to_node": "AI chips",
        "type": "develops"
      }
    ]
  },
  "judge_result": {
    "agreement_status": "Agree/Disagree/Partial",
    "summary": "...",
    "search_suggestions": ["...", "..."]
  }
}
```

---

## Troubleshooting

- **Missing `GROQ_API_KEY`** will raise a `ValueError` in JudgeAgent and KnowledgeDBAgent
- **No entities found**: FetchAgent reports and stores empty context to keep the run stable
- **SSL or parsing issues** are handled with `certifi` and BeautifulSoup's HTML parser in the fetch tool
- **Neo4j connection failures**: Ensure Neo4j credentials match your environment or the tool defaults

---

## Dependencies

- **google-adk** - Agent orchestration and sessions
- **litellm** - LLM calls to Groq models
- **wikipediaapi** - Wikipedia content retrieval
- **aiohttp** - Async HTTP requests
- **beautifulsoup4** - HTML parsing
- **certifi** - SSL certificate handling
- **neo4j** - Graph database driver
- **pydantic** - Data models and validation

---

## Notes

- Adjust model names in subagents if you want to swap LLMs or configurations
- Ensure Neo4j credentials match your environment or the tool defaults to avoid connection failures
- The system is designed to handle failures gracefully and continue processing even when some data sources are unavailable