# subagents/judge_agent/agent.py

from pydantic import BaseModel
from typing import List, Optional
from litellm import completion
import os
import json
import re
from google.adk.agents import BaseAgent
from google.adk.events import Event
from google.genai import types

# Set Groq API key
if "GROQ_API_KEY" not in os.environ:
    raise ValueError("GROQ_API_KEY environment variable not set")

# -----------------------------
# Pydantic Models
# -----------------------------
class JudgeResult(BaseModel):
    agreement_status: str  # "Agree", "Disagree", "Partial"
    summary: str
    search_suggestions: Optional[List[str]] = []

# -----------------------------
# Helper Function
# -----------------------------
def extract_json_from_response(text: str) -> str:
    """Extract JSON from markdown code blocks or raw text."""
    # Try to find JSON in markdown code blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        return json_match.group(1)
    
    # Try to find JSON object directly
    json_match = re.search(r'\{.*?\}', text, re.DOTALL)
    if json_match:
        return json_match.group(0)
    
    return text

# -----------------------------
# Judge Agent
# -----------------------------
class JudgeAgent(BaseAgent):
    """Custom agent that analyzes all previous outputs and generates final summary."""
    
    def __init__(self):
        super().__init__(name="JudgeAgent")
    
    async def _run_async_impl(self, ctx):
        """Main execution logic for the agent."""
        # Get all data from session state
        entities_output = ctx.session.state.get("entities", [])
        fetch_output = ctx.session.state.get("fetched_context", [])
        knowledge_graph = ctx.session.state.get("knowledge_graph", {"nodes": [], "relationships": []})
        user_query = ctx.session.state.get("user_query", "")
        
        # Prepare prompt for LiteLLM
        prompt = f"""You are an AI Judge Agent analyzing information about a user's query.

Original query: {user_query}

Entities extracted:
{json.dumps(entities_output, indent=2)}

Fetched summaries:
{json.dumps(fetch_output, indent=2)}

Knowledge graph:
{json.dumps(knowledge_graph, indent=2)}

Your tasks:
1. Determine if the information from different sources agrees, disagrees, or partially agrees.
2. Provide a concise executive summary that answers the user's original query.
3. Suggest 2-3 specific Google searches to learn more about the most important entities.

Return ONLY a valid JSON object with this exact structure (no markdown, no extra text):
{{
    "agreement_status": "Agree/Disagree/Partial",
    "summary": "Executive summary here...",
    "search_suggestions": ["Specific search query 1", "Specific search query 2"]
}}
"""

        try:
            # Call LiteLLM/Groq
            response = completion(
                model="groq/llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )

            # Parse response
            response_content = response.choices[0].message.content
            cleaned_json = extract_json_from_response(response_content)
            result_json = json.loads(cleaned_json)
            judge_result = JudgeResult(**result_json)
            
            # Store in session state BEFORE yielding event
            ctx.session.state["judge_result"] = judge_result.dict()
            
            # Create user-friendly output message
            output_message = f"""**Analysis Complete**

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
            
        except json.JSONDecodeError as e:
            error_msg = f"JSON parsing error: {e}"
            print(error_msg)
            print(f"Raw response content: {cleaned_json if 'cleaned_json' in locals() else response_content}")
            
            judge_result = JudgeResult(
                agreement_status="Unknown",
                summary="Failed to parse analysis results. Please try again.",
                search_suggestions=[]
            )
            
            # Store error result in session state
            ctx.session.state["judge_result"] = judge_result.dict()
            
            yield Event(
                author=self.name,
                content=types.Content(
                    role=self.name,
                    parts=[types.Part(text=f"❌ Error: {judge_result.summary}\n\nDebug info: {error_msg}")]
                )
            )
            
        except Exception as e:
            error_msg = f"Error in Judge Agent: {e}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            
            judge_result = JudgeResult(
                agreement_status="Error",
                summary=f"An error occurred during analysis: {str(e)}",
                search_suggestions=[]
            )
            
            # Store error result in session state
            ctx.session.state["judge_result"] = judge_result.dict()
            
            yield Event(
                author=self.name,
                content=types.Content(
                    role=self.name,
                    parts=[types.Part(text=f"❌ Error: {judge_result.summary}")]
                )
            )

# Create instance
judge_agent = JudgeAgent()