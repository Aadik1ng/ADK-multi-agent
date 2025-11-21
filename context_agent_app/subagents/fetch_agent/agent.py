# subagents/fetch_agent/agent.py

import os
import sys
from pathlib import Path
import time
from pydantic import PrivateAttr

# Add tools directory to path
tools_path = Path(__file__).parent.parent.parent / "tools"
sys.path.insert(0, str(tools_path))

from google.adk.agents import BaseAgent
from google.adk.events import Event
from google.genai import types
from .tools.web_fetch_tool import web_fetch_tool
from context_agent_app.config import SessionKeys, MAX_NEWS_ARTICLES_PER_ENTITY, WEB_FETCH_CACHE_TTL, CACHE_ENABLED
from context_agent_app.utils import extract_entity_names
from context_agent_app.logging_config import AgentLogger
from context_agent_app.cache import get_cache_manager, generate_cache_key


class FetchAgent(BaseAgent):
    """Custom agent that fetches real web context and news for entities."""
    
    _logger: AgentLogger = PrivateAttr()
    _cache_manager = PrivateAttr()
    _web_cache = PrivateAttr()
    
    def __init__(self):
        super().__init__(name="FetchAgent")
        self._logger = AgentLogger("FetchAgent")
        self._cache_manager = get_cache_manager()
        self._web_cache = self._cache_manager.get_cache("web_fetch", ttl=WEB_FETCH_CACHE_TTL)
        self._logger.info(f"FetchAgent initialized with caching (enabled: {CACHE_ENABLED}, TTL: {WEB_FETCH_CACHE_TTL}s)")
    
    async def _run_async_impl(self, ctx):
        """Main execution logic for the agent."""
        start_time = time.time()
        self._logger.info("Starting web fetch operation", extra={"session_id": ctx.session.id})
        
        # Get entities from session state
        entities_data = ctx.session.state.get(SessionKeys.ENTITIES, {})
        
        # Handle both dict structure and direct list
        if isinstance(entities_data, dict):
            entities = entities_data.get("entities", [])
        elif isinstance(entities_data, list):
            entities = entities_data
        else:
            entities = []
        
        if not entities:
            ctx.session.state[SessionKeys.FETCHED_CONTEXT] = []
            self._logger.warning("No entities found to fetch context for")
            
            yield Event(
                author=self.name,
                content=types.Content(
                    role=self.name,
                    parts=[types.Part(text="⚠️ No entities found to fetch context for. EntityAgent may not have run yet.")]
                )
            )
            return
        
        # Extract entity names using shared utility
        entity_names = extract_entity_names(entities)
        
        if not entity_names:
            ctx.session.state[SessionKeys.FETCHED_CONTEXT] = []
            self._logger.warning("No valid entity names extracted")
            yield Event(
                author=self.name,
                content=types.Content(
                    role=self.name,
                    parts=[types.Part(text="⚠️ No valid entity names extracted.")]
                )
            )
            return
        
        self._logger.info(f"Fetching context for {len(entity_names)} entities", extra={"entity_count": len(entity_names)})
        
        # Notify user we're starting the fetch
        yield Event(
            author=self.name,
            content=types.Content(
                role=self.name,
                parts=[types.Part(text=f"🔍 Fetching real-time context and news for: {', '.join(entity_names)}...")]
            )
        )
        
        try:
            # Try to get from cache first
            context_data = []
            cache_hits = 0
            cache_misses = 0
            
            for entity_name in entity_names:
                cache_key = generate_cache_key("web_fetch", entity_name)
                cached_result = self._web_cache.get(cache_key) if CACHE_ENABLED else None
                
                if cached_result:
                    context_data.append(cached_result)
                    cache_hits += 1
                    self._logger.debug(f"Cache hit for entity: {entity_name}", extra={"cache_hit": True})
                else:
                    cache_misses += 1
                    self._logger.debug(f"Cache miss for entity: {entity_name}", extra={"cache_hit": False})
            
            # Fetch missing entities
            if cache_misses > 0:
                entities_to_fetch = [name for name in entity_names if not any(item.get("entity") == name for item in context_data)]
                self._logger.info(f"Fetching {len(entities_to_fetch)} entities from web (cache hits: {cache_hits})")
                
                fetch_start = time.time()
                fetched_data = await web_fetch_tool.fetch_multiple_entities(
                    entities_to_fetch, 
                    include_news=True
                )
                fetch_duration = (time.time() - fetch_start) * 1000
                self._logger.info(f"Web fetch completed", extra={"duration_ms": fetch_duration, "entity_count": len(entities_to_fetch)})
                
                # Cache the fetched results
                if CACHE_ENABLED:
                    for item in fetched_data:
                        entity_name = item.get("entity")
                        if entity_name:
                            cache_key = generate_cache_key("web_fetch", entity_name)
                            self._web_cache.set(cache_key, item)
                            self._logger.debug(f"Cached result for entity: {entity_name}")
                
                context_data.extend(fetched_data)
            else:
                self._logger.info(f"All {len(entity_names)} entities served from cache", extra={"cache_hit": True})
            
            # Store raw context data in session state
            ctx.session.state[SessionKeys.FETCHED_CONTEXT] = context_data
            
            # Format for display
            formatted_context = web_fetch_tool.format_context_for_llm(context_data)
            
            # Count what we found
            total_sources = 0
            total_news = 0
            for item in context_data:
                if item.get("wikipedia"):
                    total_sources += 1
                if item.get("duckduckgo"):
                    total_sources += 1
                total_news += len(item.get("news", []))
            
            total_duration = (time.time() - start_time) * 1000
            self._logger.info(
                f"Fetch operation completed successfully",
                extra={
                    "duration_ms": total_duration,
                    "entity_count": len(entity_names),
                    "cache_hits": cache_hits,
                    "cache_misses": cache_misses,
                    "total_sources": total_sources,
                    "total_news": total_news
                }
            )
            
            cache_info = f" (💾 {cache_hits} cached)" if cache_hits > 0 else ""
            summary = f"""✅ Successfully fetched context for {len(entity_names)} entities{cache_info}:
- {total_sources} encyclopedia/reference sources
- {total_news} recent news articles

{formatted_context}
"""
            
            yield Event(
                author=self.name,
                content=types.Content(
                    role=self.name,
                    parts=[types.Part(text=summary)]
                )
            )
            
        except Exception as e:
            error_duration = (time.time() - start_time) * 1000
            error_msg = f"❌ Error fetching context: {str(e)}"
            self._logger.error(
                "Fetch operation failed",
                extra={"duration_ms": error_duration, "error": str(e)},
                exc_info=True
            )
            
            # Set empty context on error
            ctx.session.state[SessionKeys.FETCHED_CONTEXT] = []
            
            yield Event(
                author=self.name,
                content=types.Content(
                    role=self.name,
                    parts=[types.Part(text=error_msg)]
                )
            )


# Create instance
fetch_agent = FetchAgent()
