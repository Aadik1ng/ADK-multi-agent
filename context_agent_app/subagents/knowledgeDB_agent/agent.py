from typing import ClassVar
from google.adk.agents import BaseAgent
from google.adk.events import Event
from google.genai import types
from .tools.neo4j import Neo4jTool
from pydantic import PrivateAttr
import json
import logging
from google.adk.runners import Runner
from litellm import completion
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from context_agent_app.subagents.entity_agent.agent import entity_agent


class KnowledgeDBAgent(BaseAgent):
    """Agent that builds a knowledge graph from fetched context using EntityAgent."""

    # ClassVar must be assigned outside the class after import
    temp_session_service: ClassVar = None

    _neo_tool: Neo4jTool = PrivateAttr()

    def __init__(self):
        super().__init__(name="KnowledgeDBAgent")
        self._neo_tool = Neo4jTool()

    async def _run_async_impl(self, ctx):
        try:
            # Step 1: Get fetched context
            fetch_output = ctx.session.state.get("fetched_context", [])
            if not fetch_output:
                yield Event(
                    author=self.name,
                    content=types.Content(
                        role=self.name,
                        parts=[types.Part(text="⚠️ No fetched context available.")]
                    )
                )
                ctx.session.state["knowledge_graph"] = {"nodes": [], "relationships": []}
                return

            # Step 2: Combine fetched context
            combined_text = "\n".join(
                f"{item.get('entity', 'Unknown')}:\n"
                + (item.get('wikipedia', {}).get('summary', '') or "")
                + " "
                + " ".join(article.get("title", "") for article in item.get("news", [])[:3])
                for item in fetch_output
            )

            # Step 3: Call EntityAgent programmatically
            if not self.temp_session_service:
                raise ValueError("temp_session_service is not set. Set it via KnowledgeDBAgent.temp_session_service = InMemorySessionService()")

            temp_session = await self.temp_session_service.create_session(
                app_name="KnowledgeDB_EntityExtraction",
                user_id=ctx.session.user_id,
                session_id=f"{ctx.session.id}_entity_extraction",
                state={"user_query": combined_text}
            )

            # Import EntityAgent

            runner = Runner(
                agent=entity_agent,
                app_name="KnowledgeDB_EntityExtraction",
                session_service=self.temp_session_service
            )

            content = types.Content(role="user", parts=[types.Part(text=combined_text)])
            enriched_entities = []

            async for event in runner.run_async(
                user_id=ctx.session.user_id,
                session_id=temp_session.id,
                new_message=content
            ):
                if event.author == "EntityAgent":
                    updated_session = await self.temp_session_service.get_session(
                        app_name="KnowledgeDB_EntityExtraction",
                        user_id=ctx.session.user_id,
                        session_id=temp_session.id
                    )
                    entities_data = updated_session.state.get("entities", {})
                    if isinstance(entities_data, str):
                        entities_data = json.loads(entities_data)
                    enriched_entities = entities_data.get("entities", [])

            if not enriched_entities:
                yield Event(
                    author=self.name,
                    content=types.Content(
                        role=self.name,
                        parts=[types.Part(text="⚠️ No entities extracted by EntityAgent.")]
                    )
                )
                ctx.session.state["knowledge_graph"] = {"nodes": [], "relationships": []}
                return

            # Step 4: Enrich nodes with descriptions
            nodes = []
            for ent in enriched_entities:
                name = ent.get("name", "Unknown")
                type_ = ent.get("type", "Unknown")
                description = ""
                for item in fetch_output:
                    if item.get("entity") == name:
                        wiki_summary = item.get("wikipedia", {}).get("summary", "")
                        description = wiki_summary.split('.')[0] + '.' if wiki_summary else ""
                        break
                if not description:
                    description = f"A {type_.lower()} entity mentioned in the context."
                nodes.append({"name": name, "type": type_, "summary": description})

            # Step 5: Generate relationships with one LLM call
            relationship_prompt = f"""
You are a Knowledge Graph generator.

Given these entities with descriptions:
{json.dumps(nodes, indent=2)}

Suggest relationships between entities based on their descriptions and types.
Return ONLY a JSON array with objects:
[{{"from_node": "...", "to_node": "...", "type": "..."}}]
Only include relationships between existing nodes. Return [] if none.
"""

            llm_response = completion(
                model="groq/llama-3.1-8b-instant",
                messages=[{"role": "user", "content": relationship_prompt}],
                temperature=0.3
            )

            rels_text = llm_response.choices[0].message.content
            try:
                relationships = json.loads(rels_text)
            except json.JSONDecodeError:
                relationships = []

            # Step 6: Save to Neo4j and session
            knowledge_graph = {"nodes": nodes, "relationships": relationships}
            self._neo_tool.save_knowledge_graph(knowledge_graph)
            ctx.session.state["knowledge_graph"] = knowledge_graph

            # Step 7: Yield success event
            yield Event(
                author=self.name,
                content=types.Content(
                    role=self.name,
                    parts=[types.Part(
                        text=f"✅ Knowledge graph built successfully with {len(nodes)} nodes "
                             f"and {len(relationships)} relationships."
                    )]
                )
            )

        except Exception as e:
            logging.exception("Failed to build knowledge graph")
            ctx.session.state["knowledge_graph"] = {"nodes": [], "relationships": []}
            yield Event(
                author=self.name,
                content=types.Content(
                    role=self.name,
                    parts=[types.Part(text=f"❌ Failed to build knowledge graph: {e}")]
                )
            )

    def _determine_relationship(self, from_type, to_type):
        """Fallback heuristic if needed."""
        if from_type == "Organization" and to_type == "Person":
            return "employs"
        if from_type == "Organization" and to_type == "Technology":
            return "develops"
        if from_type == "Person" and to_type == "Technology":
            return "works_with"
        if from_type == "Location" and to_type == "Organization":
            return "hosts"
        if from_type == "Location" and to_type == "Technology":
            return "located_in"
        return None
