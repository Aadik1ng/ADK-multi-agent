"""
Entity Agent - Extracts entities from user queries with dynamic entity types.
Uses the ADK Agent class with declarative schema pattern.
Includes comprehensive logging and caching for performance.
"""

from typing import List
from pydantic import BaseModel, Field
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from context_agent_app.config import (
    ENTITY_EXTRACTION_MODEL, 
    ENTITY_TEMPERATURE, 
    SessionKeys,
    ENTITY_CACHE_TTL,
    CACHE_ENABLED
)
from context_agent_app.logging_config import AgentLogger
from context_agent_app.cache import get_cache_manager, generate_cache_key, hash_text

# Initialize logger
logger = AgentLogger("EntityAgent")

# Initialize cache
cache_manager = get_cache_manager()
entity_cache = cache_manager.get_cache("entity_extraction", ttl=ENTITY_CACHE_TTL)

# -----------------------------
# Pydantic Models
# -----------------------------
class EntityAgentInputs(BaseModel):
    """Input schema for EntityAgent."""
    entities: List[dict] = Field(default_factory=list)
    user_query: str = Field(default="")

class Entity(BaseModel):
    """Dynamic entity model - type is inferred from context, not hardcoded."""
    name: str = Field(description="The name of the entity")
    type: str = Field(description="The type/category of the entity (e.g., Organization, Person, Technology, Location, Event, Concept, etc.)")

class EntityOutput(BaseModel):
    """Output schema for EntityAgent."""
    entities: List[Entity] = Field(default_factory=list)

# -----------------------------
# Entity Agent
# -----------------------------
logger.info("Initializing EntityAgent with dynamic entity type extraction")
logger.info(f"Cache enabled: {CACHE_ENABLED}, TTL: {ENTITY_CACHE_TTL}s")

entity_agent = Agent(
    name="EntityAgent",
    model=LiteLlm(model=ENTITY_EXTRACTION_MODEL, temperature=ENTITY_TEMPERATURE),
    description="Extracts named entities from user queries with dynamically determined entity types",
    instruction="""
You are an advanced Entity Extraction Agent that identifies entities from complex queries including events, concepts, metaphors, and historical/biblical references.

<user_query>
{user_query?}
</user_query>

## STEP 1: ANALYZE THE QUERY TYPE

First, understand what the query is asking about:
- **Event/Historical Reference**: Is it about something that happened? (e.g., "turning water into wine", "Battle of Waterloo", "moon landing")
- **Concept/Idea**: Is it about an abstract concept or theory? (e.g., "quantum entanglement", "supply and demand")
- **Metaphor/Idiom**: Is it a saying or metaphorical phrase? (e.g., "the apple doesn't fall far from the tree")
- **Question about Entity**: Is it asking about a specific person, place, or thing? (e.g., "who is Elon Musk", "what is the Eiffel Tower")
- **Literal Entities**: Is it a straightforward list of entities? (e.g., "Google and Microsoft")

## STEP 2: EXTRACT COMPREHENSIVE ENTITIES

Based on the query type, extract 3-8 relevant entities including:

### For Events/Miracles/Historical References:
- The event itself as an entity (with type: Historical Event, Biblical Event, Miracle, etc.)
- Key participants/figures involved
- Locations where it occurred
- Time periods or dates
- Related texts, documents, or sources
- Broader context (e.g., wars, movements, religious texts)

### For Concepts/Ideas:
- The main concept
- Related theories or frameworks
- Key figures who developed/studied it
- Fields of study
- Practical applications

### For Metaphors/Idioms:
- The metaphorical concept being expressed
- Literal elements if relevant
- Cultural or linguistic origin
- Related concepts

### For Questions:
- The subject being asked about
- Related entities that provide context
- Alternative names or related terms

## STEP 3: INCLUDE IMPLICIT CONTEXT

Don't just extract what's explicitly mentioned - include entities that would help understand the query:
- For biblical references → extract religious figures, sacred texts, locations, denominations
- For historical events → extract key people, places, time periods, related events
- For scientific concepts → extract scientists, theories, related fields, applications
- For cultural references → extract origins, key figures, related works

## EXAMPLES

**Query: "turning water into wine"**
Entities to extract:
- "Wedding at Cana" (type: Biblical Event)
- "Jesus Christ" (type: Religious Figure)
- "Bible" (type: Religious Text)
- "New Testament" (type: Religious Text)
- "Gospel of John" (type: Religious Text)
- "Miracle" (type: Concept)
- "Christianity" (type: Religion)

**Query: "who shot JFK"**
Entities to extract:
- "John F. Kennedy" (type: Person)
- "JFK Assassination" (type: Historical Event)
- "Lee Harvey Oswald" (type: Person)
- "Dallas" (type: Location)
- "1963" (type: Date)
- "Warren Commission" (type: Organization)

**Query: "the apple doesn't fall far from the tree"**
Entities to extract:
- "Proverb" (type: Concept)
- "Heredity" (type: Concept)
- "Family resemblance" (type: Concept)
- "Genetics" (type: Concept)

**Query: "what is quantum entanglement"**
Entities to extract:
- "Quantum entanglement" (type: Scientific Concept)
- "Quantum mechanics" (type: Scientific Field)
- "Albert Einstein" (type: Person)
- "Erwin Schrödinger" (type: Person)
- "Bell's theorem" (type: Scientific Concept)

## ENTITY TYPE GUIDELINES

Use descriptive, specific types:
- Biblical Event, Historical Event, Scientific Discovery, Cultural Event
- Religious Figure, Historical Figure, Scientist, Artist, Political Leader
- Religious Text, Historical Document, Scientific Theory
- Miracle, Concept, Theory, Principle, Law
- Religion, Philosophy, Ideology, Movement
- Sacred Site, Historical Location, Landmark

## OUTPUT REQUIREMENTS

1. Extract 3-8 entities per query (not just 1-2)
2. Include both explicitly mentioned AND implicit contextual entities
3. Prioritize entities that would help fetch comprehensive information
4. Use specific, descriptive entity types
5. Think about what information would best answer or explain the query

Remember: Your goal is to extract entities that will enable fetching the most relevant and comprehensive context to understand and answer the user's query.
""",
    input_schema=EntityAgentInputs,
    output_schema=EntityOutput,
    output_key=SessionKeys.ENTITIES
)

logger.info("EntityAgent initialized successfully")
