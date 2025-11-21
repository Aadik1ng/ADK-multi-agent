# subagents/judge_agent/agent.py

from pydantic import BaseModel, PrivateAttr
from typing import List, Optional
from litellm import completion
import os
import json
import time
from google.adk.agents import BaseAgent
from google.adk.events import Event
from google.genai import types
from context_agent_app.config import JUDGE_MODEL, JUDGE_TEMPERATURE, SessionKeys
from context_agent_app.utils import extract_json_from_response, parse_json_safely
from context_agent_app.logging_config import AgentLogger

# Set Groq API key
if "GROQ_API_KEY" not in os.environ:
    raise ValueError("GROQ_API_KEY environment variable not set")

# -----------------------------
# Pydantic Models
# -----------------------------
class JudgeResult(BaseModel):
    agreement_status: str  # "Agree", "Disagree", "Partial"
    direct_answer: str  # Direct yes/no or explanation answering the query
    summary: str
    search_suggestions: Optional[List[str]] = []

# Helper function removed - now using shared utility from context_agent_app.utils

# -----------------------------
# Judge Agent
# -----------------------------
class JudgeAgent(BaseAgent):
    """Custom agent that analyzes all previous outputs and generates final summary."""
    
    _logger: AgentLogger = PrivateAttr()
    
    def __init__(self):
        super().__init__(name="JudgeAgent")
        self._logger = AgentLogger("JudgeAgent")
        self._logger.info("JudgeAgent initialized")
    
    async def _run_async_impl(self, ctx):
        """Main execution logic for the agent."""
        start_time = time.time()
        self._logger.info("Starting judge analysis", extra={"session_id": ctx.session.id})
        
        # Get all data from session state using SessionKeys
        entities_output = ctx.session.state.get(SessionKeys.ENTITIES, [])
        fetch_output = ctx.session.state.get(SessionKeys.FETCHED_CONTEXT, [])
        knowledge_graph = ctx.session.state.get(SessionKeys.KNOWLEDGE_GRAPH, {"nodes": [], "relationships": []})
        user_query = ctx.session.state.get(SessionKeys.USER_QUERY, "")
        
        self._logger.debug(
            "Retrieved session data",
            extra={
                "entity_count": len(entities_output) if isinstance(entities_output, list) else 0,
                "fetch_count": len(fetch_output) if isinstance(fetch_output, list) else 0,
                "kg_nodes": len(knowledge_graph.get("nodes", [])) if isinstance(knowledge_graph, dict) else 0
            }
        )
        
        # Prepare prompt for LiteLLM
        prompt = f"""You are an AI Judge Agent that provides clear, direct answers to user queries based on fetched information.

Original query: {user_query}

Entities extracted:
{json.dumps(entities_output, indent=2)}

Fetched summaries:
{json.dumps(fetch_output, indent=2)}

Knowledge graph:
{json.dumps(knowledge_graph, indent=2)}

## YOUR TASK: Answer the user's query clearly and directly

### STEP 1: Provide a Direct Answer

First, determine what type of query this is and provide an appropriate direct answer:

- **Yes/No Question** (e.g., "is X true?", "did Y happen?"): Answer "Yes" or "No" with a brief explanation
- **What/Who/Where Question** (e.g., "what is X?", "who did Y?"): Provide a clear definition or identification
- **Event/Historical Query** (e.g., "turning water into wine", "Battle of Waterloo"): Confirm what it refers to and provide context
- **Comparison Query** (e.g., "X vs Y"): Briefly state what's being compared
- **General Query**: Provide the most relevant direct answer based on the information

Examples:
- Query: "turning water into wine" → Direct Answer: "This refers to a biblical miracle performed by Jesus Christ at the Wedding at Cana, as described in the Gospel of John."
- Query: "who shot JFK" → Direct Answer: "Lee Harvey Oswald is officially credited with assassinating President John F. Kennedy on November 22, 1963."
- Query: "is the earth flat" → Direct Answer: "No, the Earth is an oblate spheroid (nearly spherical), as confirmed by extensive scientific evidence."

### STEP 2: Evaluate Source Agreement (Be Less Strict)

Determine if the sources provide consistent information:

**Default to "Agree" unless there are clear contradictions:**
- "Agree": Sources are consistent and provide coherent, complementary information (even if covering different aspects)
- "Partial": Sources have minor inconsistencies but generally align on the main facts
- "Disagree": Sources have major contradictions about the same topic

**Important Rules:**
- If sources complement each other (different aspects of same topic) → "Agree"
- If news mentions modern entities with same names as historical figures → "Agree" (not a contradiction)
- Only mark "Partial" or "Disagree" if there are ACTUAL factual contradictions
- When in doubt, choose "Agree"

### STEP 3: Create Comprehensive Summary

Provide a comprehensive answer that:
1. Directly addresses what the user is asking about
2. Synthesizes information from all sources
3. Focuses on the PRIMARY topic (usually from Wikipedia/authoritative sources)
4. Includes relevant context and details
5. Is clear, informative, and well-organized

### STEP 4: Suggest Related Searches

Suggest 2-3 specific follow-up searches about the PRIMARY topic that would help the user learn more.

## OUTPUT FORMAT

Return ONLY a valid JSON object with this exact structure (no markdown, no extra text):
{{
    "agreement_status": "Agree/Disagree/Partial",
    "direct_answer": "Clear, direct answer to the user's query",
    "summary": "Comprehensive summary addressing the query",
    "search_suggestions": ["Specific search query 1", "Specific search query 2"]
}}
"""

        try:
            # Call LiteLLM/Groq with config values
            self._logger.debug(f"Calling LLM for judge analysis (model: {JUDGE_MODEL})")
            llm_start = time.time()
            
            response = completion(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=JUDGE_TEMPERATURE,
            )
            
            llm_duration = (time.time() - llm_start) * 1000
            self._logger.info(f"LLM call completed", extra={"duration_ms": llm_duration})

            # Parse response using shared utility
            response_content = response.choices[0].message.content
            result_json = parse_json_safely(response_content, default={})
            
            if not result_json:
                raise ValueError("Failed to parse JSON from LLM response")
            
            judge_result = JudgeResult(**result_json)
            
            # Store in session state BEFORE yielding event
            ctx.session.state[SessionKeys.JUDGE_RESULT] = judge_result.dict()
            
            total_duration = (time.time() - start_time) * 1000
            self._logger.info(
                "Judge analysis completed successfully",
                extra={
                    "duration_ms": total_duration,
                    "agreement_status": judge_result.agreement_status,
                    "search_suggestions_count": len(judge_result.search_suggestions)
                }
            )
            
            # Create user-friendly output message
            output_message = f"""**Analysis Complete**

**Direct Answer:** {judge_result.direct_answer}

**Status:** {judge_result.agreement_status}

**Summary:**
{judge_result.summary}

**Suggested searches:**
{chr(10).join(f"• {s}" for s in judge_result.search_suggestions)}
"""
            
            # Send output to user
            yield Event(
                author=self.name,
                content=types.Content(
                    role=self.name,
                    parts=[types.Part(text=output_message)]
                )
            )
            
        except (json.JSONDecodeError, ValueError) as e:
            error_duration = (time.time() - start_time) * 1000
            error_msg = f"JSON parsing error: {e}"
            self._logger.error(
                "Judge analysis failed - JSON parsing error",
                extra={"duration_ms": error_duration, "error": str(e)}
            )
            print(f"[JudgeAgent] {error_msg}")
            print(f"[JudgeAgent] Raw response: {response_content[:500]}...")
            
            judge_result = JudgeResult(
                agreement_status="Unknown",
                direct_answer="Failed to parse analysis results. Please try again.",
                summary="An error occurred while analyzing the information.",
                search_suggestions=[]
            )
            
            # Store error result in session state
            ctx.session.state[SessionKeys.JUDGE_RESULT] = judge_result.dict()
            
            yield Event(
                author=self.name,
                content=types.Content(
                    role=self.name,
                    parts=[types.Part(text=f"❌ Error: {judge_result.summary}\n\nDebug info: {error_msg}")]
                )
            )
            
        except Exception as e:
            error_duration = (time.time() - start_time) * 1000
            error_msg = f"Error in Judge Agent: {e}"
            self._logger.error(
                "Judge analysis failed - unexpected error",
                extra={"duration_ms": error_duration, "error": str(e)},
                exc_info=True
            )
            
            judge_result = JudgeResult(
                agreement_status="Error",
                direct_answer=f"An error occurred during analysis: {str(e)}",
                summary="Unable to complete analysis due to an error.",
                search_suggestions=[]
            )
            
            # Store error result in session state
            ctx.session.state[SessionKeys.JUDGE_RESULT] = judge_result.dict()
            
            yield Event(
                author=self.name,
                content=types.Content(
                    role=self.name,
                    parts=[types.Part(text=f"❌ Error: {judge_result.summary}")]
                )
            )

# Create instance
judge_agent = JudgeAgent()