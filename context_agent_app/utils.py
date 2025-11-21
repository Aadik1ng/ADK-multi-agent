"""
Shared utility functions for the multi-agent system.
"""
import json
import re
from typing import Any, Dict, List, Optional


def extract_json_from_response(text: str) -> str:
    """
    Extract JSON from markdown code blocks or raw text.
    
    Args:
        text: Response text that may contain JSON
        
    Returns:
        Extracted JSON string
        
    Raises:
        ValueError: If no JSON found in the text
    """
    # Try to find JSON in markdown code blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        return json_match.group(1)
    
    # Try to find JSON array in markdown code blocks
    json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', text, re.DOTALL)
    if json_match:
        return json_match.group(1)
    
    # Try to find JSON object directly
    json_match = re.search(r'\{.*?\}', text, re.DOTALL)
    if json_match:
        return json_match.group(0)
    
    # Try to find JSON array directly
    json_match = re.search(r'\[.*?\]', text, re.DOTALL)
    if json_match:
        return json_match.group(0)
    
    raise ValueError(f"No JSON found in response: {text[:200]}...")


def parse_json_safely(text: str, default: Any = None) -> Any:
    """
    Safely parse JSON with fallback to default value.
    
    Args:
        text: JSON string to parse
        default: Default value if parsing fails
        
    Returns:
        Parsed JSON or default value
    """
    try:
        cleaned = extract_json_from_response(text)
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"[Utils] JSON parsing failed: {e}")
        print(f"[Utils] Raw text: {text[:200]}...")
        return default


def extract_entity_names(entities: Any) -> List[str]:
    """
    Extract entity names from various entity data structures.
    
    Args:
        entities: Can be a list of dicts, list of strings, dict with 'entities' key, etc.
        
    Returns:
        List of entity name strings
    """
    entity_names = []
    
    # Handle dict with 'entities' key
    if isinstance(entities, dict):
        entities = entities.get("entities", [])
    
    # Handle list of entities
    if isinstance(entities, list):
        for entity in entities:
            if isinstance(entity, dict) and "name" in entity:
                entity_names.append(entity["name"])
            elif isinstance(entity, str):
                entity_names.append(entity)
            else:
                print(f"[Utils] Warning: Unexpected entity format: {entity}")
    
    return entity_names


def format_entities_for_display(entities: List[Dict[str, Any]]) -> str:
    """
    Format entities for human-readable display.
    
    Args:
        entities: List of entity dictionaries with 'name' and 'type' keys
        
    Returns:
        Formatted string for display
    """
    if not entities:
        return "No entities found"
    
    lines = []
    for entity in entities:
        name = entity.get("name", "Unknown")
        entity_type = entity.get("type", "Unknown")
        lines.append(f"• {name} ({entity_type})")
    
    return "\n".join(lines)


def validate_session_state(state: Dict[str, Any], required_keys: List[str]) -> bool:
    """
    Validate that session state contains required keys.
    
    Args:
        state: Session state dictionary
        required_keys: List of required key names
        
    Returns:
        True if all required keys present, False otherwise
    """
    missing_keys = [key for key in required_keys if key not in state]
    if missing_keys:
        print(f"[Utils] Missing required session keys: {missing_keys}")
        return False
    return True


def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """
    Truncate text to maximum length with suffix.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix
