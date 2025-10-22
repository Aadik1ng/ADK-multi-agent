import uuid
from google.adk.agents import SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Import all subagents
from .subagents.entity_agent.agent import entity_agent
from .subagents.fetch_agent.agent import fetch_agent
from .subagents.knowledgeDB_agent.agent import KnowledgeDBAgent
from .subagents.judge_agent.agent import judge_agent


# =========================
# 1️⃣ Session Setup
# =========================
session_service_stateful = InMemorySessionService()
KnowledgeDBAgent.temp_session_service = InMemorySessionService()

# 2️⃣ Create the agent instance
knowledgeDB_agent = KnowledgeDBAgent()

initial_state = {
    "user_name": "Brandon Hancock",
    "user_query": "",
    "interaction_history": [],
    "entities": [],
    "fetched_context": [],
    "knowledge_graph": {},
    "final_summary": "",
}

APP_NAME = "Brandon MultiAgent Bot"
USER_ID = "brandon_hancock"
SESSION_ID = str(uuid.uuid4())

# ⚠️ ADK session creation is async — do this inside an async runner if needed.
# Example:
# stateful_session = await session_service_stateful.create_session(
#     app_name=APP_NAME,
#     user_id=USER_ID,
#     session_id=SESSION_ID,
#     state=initial_state,
# )
stateful_session = session_service_stateful.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
        state=initial_state,
    )
print(f"[Session] Created new session: {SESSION_ID}")

# =========================
# 3️⃣ Multi-Agent Pipeline
# =========================
root_agent = SequentialAgent(
    name="context_understanding_root_agent",
    sub_agents=[
        entity_agent,
        fetch_agent,
        knowledgeDB_agent,  # ✅ pass the *instance*, not () call
        judge_agent,
    ],
)


runner = Runner(
    agent=root_agent,
    app_name=APP_NAME,
    session_service=session_service_stateful,
)

# =========================
# 3️⃣ Utility Functions
# =========================
def set_user_query(statements: list[str]):
    session = session_service_stateful.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )
    joined_query = "\n".join(statements).strip()
    session.state["user_query"] = joined_query  # store as a string
    session.state["interaction_history"].append({
        "action": "user_query",
        "statements": statements
    })
    session_service_stateful.update_session(session)
    print(f"[Session] Updated user_query with {len(statements)} statements")


def add_agent_response(agent_name: str, response_text: str):
    session = session_service_stateful.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )
    session.state["interaction_history"].append({
        "action": "agent_response",
        "agent": agent_name,
        "response": response_text
    })
    session_service_stateful.update_session(session)


async def process_query_from_web_gui(ctx):
    """
    Triggered automatically by ADK Web when user submits a query.
    """
    # Retrieve query from session state (automatically managed by ADK Web)
    statements = ctx.session.state.get("user_query", [])

    if not statements or not isinstance(statements, list) or not any(s.strip() for s in statements):
        print("[EntityAgent] ⚠️ No valid query found in session state. Skipping processing.")
        return

    # Filter and store
    statements = [s.strip() for s in statements if s.strip()]
    set_user_query(statements)

    joined = "\n".join(statements)
    prompt = f"""Analyze the following list of statements:\n\n{joined}\n\n
Extract important entities, fetch relevant context, construct a knowledge graph, 
and provide an executive summary indicating agreements, disagreements, and suggested follow-up searches.
"""

    content = types.Content(role="user", parts=[types.Part(text=prompt)])
    print(f"[Runner] Starting sequential multi-agent pipeline for query:\n{joined}")

    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=content,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(f"[{event.author}] {part.text}")
                    add_agent_response(event.author, part.text)

    final_session = session_service_stateful.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )
    print("[Runner] Multi-agent analysis complete.")
    print(f"  - Entities: {len(final_session.state.get('entities', []))}")
    print(f"  - Contexts: {len(final_session.state.get('fetched_context', []))}")
    print(f"  - Knowledge Graph Nodes: {len(final_session.state.get('knowledge_graph', {}))}")
    print(f"  - Summary: {final_session.state.get('final_summary', 'N/A')[:120]}...")
    return final_session.state


def get_session_state():
    return session_service_stateful.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    ).state


def reset_session():
    session = session_service_stateful.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )
    session.state = initial_state.copy()
    session_service_stateful.update_session(session)
    print("[Session] Reset complete")
