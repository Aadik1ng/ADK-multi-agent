# subagents/fetch_agent/agent.py

import os
import sys
from pathlib import Path

# Add tools directory to path
tools_path = Path(__file__).parent.parent.parent / "tools"
sys.path.insert(0, str(tools_path))

from google.adk.agents import BaseAgent
from google.adk.events import Event
from google.genai import types
from .tools.web_fetch_tool import web_fetch_tool


class FetchAgent(BaseAgent):
    """Custom agent that fetches real web context and news for entities."""
    
    def __init__(self):
        super().__init__(name="FetchAgent")
    
    async def _run_async_impl(self, ctx):
        """Main execution logic for the agent."""
        
        # Get entities from session state (EntityAgent stores it under 'entities' key)
        # Based on your output: stateDelta.entities.entities[]
        entities_data = ctx.session.state.get("entities", {})
        
        # Handle both dict structure and direct list
        if isinstance(entities_data, dict):
            entities = entities_data.get("entities", [])
        elif isinstance(entities_data, list):
            entities = entities_data
        else:
            entities = []
        
        if not entities:
            ctx.session.state["fetched_context"] = []
            
            yield Event(
                author=self.name,
                content=types.Content(
                    role=self.name,
                    parts=[types.Part(text="⚠️ No entities found to fetch context for. EntityAgent may not have run yet.")]
                )
            )
            return
        
        # Extract entity names from the structured entity objects
        entity_names = []
        for entity in entities:
            if isinstance(entity, dict) and "name" in entity:
                entity_names.append(entity["name"])
            elif isinstance(entity, str):
                entity_names.append(entity)
            else:
                print(f"[FetchAgent] Warning: Unexpected entity format: {entity}")
        
        if not entity_names:
            ctx.session.state["fetched_context"] = []
            yield Event(
                author=self.name,
                content=types.Content(
                    role=self.name,
                    parts=[types.Part(text="⚠️ No valid entity names extracted.")]
                )
            )
            return
        
        # Notify user we're starting the fetch
        yield Event(
            author=self.name,
            content=types.Content(
                role=self.name,
                parts=[types.Part(text=f"🔍 Fetching real-time context and news for: {', '.join(entity_names)}...")]
            )
        )
        
        try:
            # Fetch context from multiple web sources
            print(f"[FetchAgent] Calling web_fetch_tool for entities: {entity_names}")
            context_data = await web_fetch_tool.fetch_multiple_entities(
                entity_names, 
                include_news=True
            )
            
            print(f"[FetchAgent] Fetched context data: {len(context_data)} items")
            
            # Store raw context data in session state
            ctx.session.state["fetched_context"] = context_data
            
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
            
            summary = f"""✅ Successfully fetched context for {len(entity_names)} entities:
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
            error_msg = f"❌ Error fetching context: {str(e)}"
            print(f"[FetchAgent] {error_msg}")
            import traceback
            traceback.print_exc()
            
            # Set empty context on error
            ctx.session.state["fetched_context"] = []
            
            yield Event(
                author=self.name,
                content=types.Content(
                    role=self.name,
                    parts=[types.Part(text=error_msg)]
                )
            )


# Create instance
fetch_agent = FetchAgent()
