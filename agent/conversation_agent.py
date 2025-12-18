"""
Conversation agent using DeepSeek API with function calling.

Handles the conversation loop and tool execution for object
registration and recognition.
"""

import json
from typing import List, Dict, Any, Optional
from openai import OpenAI

from .config import AgentConfig
from .tools import AVAILABLE_TOOLS
from .agent_service import AgentService


class ConversationAgent:
    """
    Conversation agent using DeepSeek API with function calling.

    This class manages the conversation with the DeepSeek API, handles
    function calling for object registration and recognition, and
    maintains conversation history.

    Example:
        >>> config = AgentConfig.from_env()
        >>> service = AgentService()
        >>> agent = ConversationAgent(config, service)
        >>> response = agent.chat("Register my blue mug from examples/handheld/mug1/")
        >>> print(response)
    """

    SYSTEM_PROMPT = """You are an intelligent assistant for an object registration and recognition system. You help users manage and find their objects.

Your capabilities:
1. **Register objects**: When users want to add/register/remember a new object, extract the image path, generate an instance_id, and list ALL descriptive attributes they mention.
2. **Find objects**: When users want to find/locate/search for an object, use their description as a text query for semantic matching.
3. **List objects**: Show what objects are registered in the system.

Important guidelines:
- **CRITICAL: Always use English for attributes and text_query parameters.** The semantic matching model works best with English text. Translate user descriptions to English when calling tools.
- For registration: Extract ALL descriptive details as attributes (color, pattern, ownership like "my favorite", brand, material, etc.). Write them in English.
- For recognition: Formulate the text_query in English that captures what the user is looking for.
- Always respond to the user in their language (e.g., Chinese), but use English for tool parameters.
- After executing a tool, summarize the results clearly and concisely.
- If an operation fails, explain the error helpfully.

When generating instance_id:
- Keep it short and descriptive (e.g., 'mug1', 'my_blue_cup', 'mickey_mug')
- Use underscores for multi-word IDs
- Include distinguishing features if mentioned"""

    def __init__(
        self,
        config: AgentConfig,
        service: AgentService
    ):
        """
        Initialize the conversation agent.

        Args:
            config: Agent configuration
            service: AgentService instance for business logic
        """
        self.config = config
        self.service = service

        # Initialize OpenAI client for DeepSeek API
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.api_base_url
        )

        # Conversation history
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.SYSTEM_PROMPT}
        ]

    def chat(self, user_message: str) -> str:
        """
        Process a user message and return the response.

        This method:
        1. Adds the user message to conversation history
        2. Calls DeepSeek API with function calling enabled
        3. If the model requests tool calls, executes them
        4. Returns the final response

        Args:
            user_message: User's input message

        Returns:
            Assistant's response string
        """
        # Add user message to history
        self.messages.append({"role": "user", "content": user_message})

        try:
            # Call DeepSeek API with function calling
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=self.messages,
                tools=AVAILABLE_TOOLS,
                tool_choice="auto"
            )

            assistant_message = response.choices[0].message

            # Check if the model wants to call a function
            if assistant_message.tool_calls:
                # Add assistant message with tool calls to history
                self.messages.append(assistant_message.model_dump())

                # Execute each tool call
                for tool_call in assistant_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    # Execute the function
                    result = self._execute_tool(function_name, function_args)

                    # Add tool result to messages
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, ensure_ascii=False)
                    })

                # Get final response after tool execution
                final_response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=self.messages
                )

                final_message = final_response.choices[0].message.content
                self.messages.append({"role": "assistant", "content": final_message})

                return final_message
            else:
                # No tool call, just return the response
                content = assistant_message.content or ""
                self.messages.append({"role": "assistant", "content": content})
                return content

        except Exception as e:
            error_msg = f"Error communicating with API: {str(e)}"
            # Don't add error to conversation history
            return error_msg

    def _execute_tool(
        self,
        function_name: str,
        function_args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a tool function and return results.

        Args:
            function_name: Name of the function to call
            function_args: Arguments for the function

        Returns:
            Dictionary with execution results
        """
        if function_name == "register_object":
            response = self.service.register_object(
                image_path=function_args["image_path"],
                instance_id=function_args["instance_id"],
                attributes=function_args.get("attributes", []),
                templates_path=function_args.get("templates_path")
            )
            return {
                "success": response.success,
                "instance_id": response.instance_id,
                "num_images_processed": response.num_images_processed,
                "num_successful": response.num_successful,
                "templates_path": response.templates_path,
                "has_text_embeddings": response.has_text_embeddings,
                "error": response.error_message
            }

        elif function_name == "recognize_object":
            response = self.service.recognize_object(
                query_image_path=function_args["query_image_path"],
                text_query=function_args["text_query"],
                templates_path=function_args.get("templates_path"),
                confidence_threshold=function_args.get(
                    "confidence_threshold",
                    self.config.default_confidence_threshold
                ),
                text_threshold=function_args.get(
                    "text_threshold",
                    self.config.default_text_threshold
                )
            )
            return {
                "success": response.success,
                "query": response.query,
                "num_images_searched": response.num_images_searched,
                "num_matches": len(response.matches),
                "matches": response.matches,
                "visualization_path": response.visualization_path,
                "error": response.error_message
            }

        elif function_name == "list_registered_objects":
            return self.service.list_registered_objects(
                templates_path=function_args.get("templates_path")
            )

        else:
            return {"error": f"Unknown function: {function_name}"}

    def reset_conversation(self):
        """Reset conversation history, keeping only system prompt."""
        self.messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT}
        ]

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the current conversation history."""
        return self.messages.copy()
