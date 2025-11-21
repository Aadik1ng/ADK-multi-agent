from typing import ClassVar
from google.adk.agents import BaseAgent
from google.adk.events import Event
from google.genai import types
from .tools.neo4j import Neo4jTool
from pydantic import PrivateAttr
import json
import logging
import time
from google.adk.runners import Runner
from litellm import completion
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from context_agent_app.config import (
    KNOWLEDGE_GRAPH_MODEL, 
    KNOWLEDGE_GRAPH_TEMPERATURE, 
    SessionKeys, 
    MAX_NEWS_ARTICLES_PER_ENTITY,
    KNOWLEDGE_GRAPH_CACHE_TTL,
    CACHE_ENABLED
)
from context_agent_app.utils import parse_json_safely
from context_agent_app.subagents.entity_agent.agent import entity_agent
from context_agent_app.logging_config import AgentLogger
from context_agent_app.cache import get_cache_manager, generate_cache_key, hash_text


class KnowledgeDBAgent(BaseAgent):
    """Agent that builds a knowledge graph from fetched context using EntityAgent."""

    # ClassVar must be assigned outside the class after import
    temp_session_service: ClassVar = None

    _neo_tool: Neo4jTool = PrivateAttr()
    _logger: AgentLogger = PrivateAttr()
    _cache_manager = PrivateAttr()
    _kg_cache = PrivateAttr()

    def __init__(self):
        super().__init__(name="KnowledgeDBAgent")
        self._neo_tool = Neo4jTool()
        self._logger = AgentLogger("KnowledgeDBAgent")
        self._cache_manager = get_cache_manager()
        self._kg_cache = self._cache_manager.get_cache("knowledge_graph", ttl=KNOWLEDGE_GRAPH_CACHE_TTL)
        self._logger.info(f"KnowledgeDBAgent initialized with caching (enabled: {CACHE_ENABLED}, TTL: {KNOWLEDGE_GRAPH_CACHE_TTL}s)")

    async def _run_async_impl(self, ctx):
        start_time = time.time()
        self._logger.info("Starting knowledge graph construction", extra={"session_id": ctx.session.id})
        
        try:
            # Step 1: Get fetched context
            fetch_output = ctx.session.state.get(SessionKeys.FETCHED_CONTEXT, [])
            if not fetch_output:
                self._logger.warning("No fetched context available for knowledge graph construction")
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
                + " ".join(article.get("title", "") for article in item.get("news", [])[:MAX_NEWS_ARTICLES_PER_ENTITY])
                for item in fetch_output
            )

            # Step 3: Call EntityAgent programmatically
            if not self.temp_session_service:
                raise ValueError("temp_session_service is not set. Set it via KnowledgeDBAgent.temp_session_service = InMemorySessionService()")

            temp_session = await self.temp_session_service.create_session(
                app_name="KnowledgeDB_EntityExtraction",
                user_id=ctx.session.user_id,
                session_id=f"{ctx.session.id}_entity_extraction",
                state={SessionKeys.USER_QUERY: combined_text}
            )

            # Use EntityAgent from imports

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
                    entities_data = updated_session.state.get(SessionKeys.ENTITIES, {})
                    if isinstance(entities_data, str):
                        entities_data = json.loads(entities_data)
                    enriched_entities = entities_data.get("entities", [])

            if not enriched_entities:
                self._logger.warning("No entities extracted by EntityAgent for knowledge graph")
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
            self._logger.debug(f"Enriching {len(enriched_entities)} entities with descriptions")
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
            
            self._logger.info(f"Enriched {len(nodes)} nodes for knowledge graph")
            
            # Step 5: Generate relationships with LLM (check cache first)
            cache_key = generate_cache_key("kg_relationships", hash_text(json.dumps(nodes, sort_keys=True)))
            relationships = self._kg_cache.get(cache_key) if CACHE_ENABLED else None
            
            if relationships:
                self._logger.info(f"Retrieved {len(relationships)} relationships from cache", extra={"cache_hit": True})
            else:
                self._logger.debug("Generating relationships with LLM", extra={"cache_hit": False})
                relationship_prompt = f"""
You are a Knowledge Graph generator.

Given these entities with descriptions:
{json.dumps(nodes, indent=2)}

Suggest relationships between entities based on their descriptions and types.
Return ONLY a JSON array with objects:
[{{"from_node": "...", "to_node": "...", "type": "..."}}]
Only include relationships between existing nodes. Return [] if none.
"""

                llm_start = time.time()
                llm_response = completion(
                    model=KNOWLEDGE_GRAPH_MODEL,
                    messages=[{"role": "user", "content": relationship_prompt}],
                    temperature=KNOWLEDGE_GRAPH_TEMPERATURE
                )
                llm_duration = (time.time() - llm_start) * 1000
                self._logger.info(f"LLM relationship generation completed", extra={"duration_ms": llm_duration})

                rels_text = llm_response.choices[0].message.content
                relationships = parse_json_safely(rels_text, default=[])
                
                if not isinstance(relationships, list):
                    self._logger.warning(f"Expected list of relationships, got {type(relationships)}")
                    relationships = []
                else:
                    self._logger.info(f"Generated {len(relationships)} relationships")
                    
                # Cache the relationships
                if CACHE_ENABLED and relationships:
                    self._kg_cache.set(cache_key, relationships)
                    self._logger.debug("Cached relationship generation results")

            # Step 6: Save to Neo4j and session
            self._logger.debug("Saving knowledge graph to Neo4j")
            knowledge_graph = {"nodes": nodes, "relationships": relationships}
            self._neo_tool.save_knowledge_graph(knowledge_graph)
            ctx.session.state[SessionKeys.KNOWLEDGE_GRAPH] = knowledge_graph
            
            total_duration = (time.time() - start_time) * 1000
            self._logger.info(
                "Knowledge graph construction completed",
                extra={
                    "duration_ms": total_duration,
                    "node_count": len(nodes),
                    "relationship_count": len(relationships)
                }
            )

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
            error_duration = (time.time() - start_time) * 1000
            self._logger.error(
                "Knowledge graph construction failed",
                extra={"duration_ms": error_duration, "error": str(e)},
                exc_info=True
            )
            ctx.session.state[SessionKeys.KNOWLEDGE_GRAPH] = {"nodes": [], "relationships": []}
            yield Event(
                author=self.name,
                content=types.Content(
                    role=self.name,
                    parts=[types.Part(text=f"❌ Failed to build knowledge graph: {e}")]
                )
            )
