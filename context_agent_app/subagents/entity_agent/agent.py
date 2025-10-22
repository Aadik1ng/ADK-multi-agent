"""
Entity Agent - Extracts entities from user queries
Uses the simpler Agent class with tools pattern
"""

import os
import json
import re
from typing import List, Literal
from pydantic import BaseModel
from litellm import completion
from google.adk.agents import Agent
from google.adk.tools.tool_context import ToolContext
from pydantic import BaseModel, Field

class EntityAgentInputs(BaseModel):
    entities: list = Field(default_factory=list)   # default empty list
    user_query: str = ""                           # default empty string

# Set Groq API key

# -----------------------------
# Pydantic Models
# -----------------------------
class Entity(BaseModel):
    name: str
    type: Literal["Organization", "Person", "Technology", "Location"]

class EntityOutput(BaseModel):
    entities: List[Entity] = []

# -----------------------------
# Entity Agent
# -----------------------------
entity_agent = Agent(
    name="EntityAgent",
    model="gemini-2.5-pro",
    description="Extracts named entities from user queries",
    instruction="""
    You are an Entity Extraction Agent that identifies important named entities in text.
    

    <user_query>
    {user_query?}
    </user_query>
    
    Your task:
    1. Analyze the user's query to identify named entities
    3. Identify:
       - Organizations (companies, institutions)
       - People (names of individuals)
       - Technologies (products, platforms, systems)
       - Locations (places, regions, countries)
    
    4. Report the extracted entities clearly
    """,
    input_schema=EntityAgentInputs ,
    output_schema=EntityOutput,
    output_key="entities"
)
