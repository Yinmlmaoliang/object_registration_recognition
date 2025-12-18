"""
Function Calling tool definitions for DeepSeek API.

Defines the JSON schemas for tools that the LLM can call:
- register_object: Register a new object from handheld images
- recognize_object: Recognize and find objects in query scenes
- list_registered_objects: List all registered objects
"""

from typing import List, Dict, Any

# Tool schema for registering objects
REGISTER_OBJECT_TOOL = {
    "type": "function",
    "function": {
        "name": "register_object",
        "description": (
            "Register a new object from handheld images. Use this when the user wants to "
            "add, register, remember, or learn a new object. Extract the image path, "
            "a unique instance ID, and descriptive attributes from the user's message. "
            "The attributes are crucial for later text-based search and retrieval."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": (
                        "Path to the image file or directory containing handheld images "
                        "of the object. Can be a single image path (e.g., 'examples/mug.jpg') "
                        "or a directory path (e.g., 'examples/handheld/mug1/')."
                    )
                },
                "instance_id": {
                    "type": "string",
                    "description": (
                        "A unique identifier for this object instance. Should be descriptive "
                        "and concise. Generate based on object type and distinguishing features. "
                        "Examples: 'mug1', 'my_blue_cup', 'mickey_mug', 'cellphone1'."
                    )
                },
                "attributes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of text descriptions about the object's characteristics. "
                        "IMPORTANT: Always write attributes in English for better semantic matching. "
                        "Include ALL details mentioned by the user: color, pattern, brand, "
                        "ownership ('my', 'favorite'), distinguishing features, etc. "
                        "These are used for semantic text-based retrieval later. "
                        "Examples: ['my favorite blue mug', 'Mickey Mouse pattern', "
                        "'ceramic cup for morning coffee']."
                    )
                },
                "templates_path": {
                    "type": "string",
                    "description": (
                        "Optional. Path to save templates. If not specified, "
                        "defaults to 'templates/objects'."
                    )
                }
            },
            "required": ["image_path", "instance_id", "attributes"]
        }
    }
}

# Tool schema for recognizing objects
RECOGNIZE_OBJECT_TOOL = {
    "type": "function",
    "function": {
        "name": "recognize_object",
        "description": (
            "Recognize and find registered objects in query scene images. Use this when "
            "the user wants to find, locate, identify, or search for an object they "
            "previously registered. Supports text-based semantic search using the "
            "object's attributes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query_image_path": {
                    "type": "string",
                    "description": (
                        "Path to the query image file or directory containing scene images "
                        "to search in. Examples: 'examples/scence/', 'scene.jpg'."
                    )
                },
                "text_query": {
                    "type": "string",
                    "description": (
                        "Natural language description of the object to find. "
                        "IMPORTANT: Always write the query in English for better semantic matching. "
                        "Extract key characteristics from the user's request. This is used for semantic "
                        "matching against registered object attributes. "
                        "Examples: 'my Mickey Mouse mug', 'the blue cup', 'favorite cellphone'."
                    )
                },
                "templates_path": {
                    "type": "string",
                    "description": (
                        "Optional. Path to load templates from. If not specified, "
                        "defaults to 'templates/objects'."
                    )
                },
                "confidence_threshold": {
                    "type": "number",
                    "description": (
                        "Optional. Minimum visual confidence score for recognition (0.0-1.0). "
                        "Higher values mean stricter matching. Defaults to 0.5."
                    )
                },
                "text_threshold": {
                    "type": "number",
                    "description": (
                        "Optional. Minimum text similarity score for retrieval (0.0-1.0). "
                        "Higher values require closer text match. Defaults to 0.3."
                    )
                }
            },
            "required": ["query_image_path", "text_query"]
        }
    }
}

# Tool schema for listing registered objects
LIST_REGISTERED_OBJECTS_TOOL = {
    "type": "function",
    "function": {
        "name": "list_registered_objects",
        "description": (
            "List all registered objects in the template library. Use when the user asks "
            "what objects are registered, what objects you know about, or wants to see "
            "available objects."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "templates_path": {
                    "type": "string",
                    "description": (
                        "Optional. Path to templates. If not specified, "
                        "defaults to 'templates/objects'."
                    )
                }
            },
            "required": []
        }
    }
}

# All available tools
AVAILABLE_TOOLS: List[Dict[str, Any]] = [
    REGISTER_OBJECT_TOOL,
    RECOGNIZE_OBJECT_TOOL,
    LIST_REGISTERED_OBJECTS_TOOL
]


def get_tool_by_name(name: str) -> Dict[str, Any]:
    """
    Get tool definition by name.

    Args:
        name: Tool function name

    Returns:
        Tool definition dict, or None if not found
    """
    for tool in AVAILABLE_TOOLS:
        if tool["function"]["name"] == name:
            return tool
    return None


def get_tool_names() -> List[str]:
    """Get list of all available tool names."""
    return [tool["function"]["name"] for tool in AVAILABLE_TOOLS]
