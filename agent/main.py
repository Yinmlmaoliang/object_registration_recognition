#!/usr/bin/env python3
"""
Terminal conversation agent for object registration and recognition.

Usage:
    # Set API key first
    export DEEPSEEK_API_KEY='your_api_key'

    # Run the agent
    python -m agent.main

    # Or run directly
    python agent/main.py

Commands during conversation:
    quit, exit, q  - Exit the agent
    reset          - Reset conversation history
    help           - Show help message
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.config import AgentConfig
from agent.agent_service import AgentService
from agent.conversation_agent import ConversationAgent


def print_welcome():
    """Print welcome message."""
    print()
    print("=" * 65)
    print("  Object Registration & Recognition Agent")
    print("=" * 65)
    print()
    print("I can help you:")
    print("  - Register objects from handheld images (remember new objects)")
    print("  - Find/recognize objects in scene images (locate objects)")
    print("  - List registered objects (see what I know)")
    print()
    print("Example commands:")
    print('  "I took photos of my blue mug at examples/handheld/mug1/"')
    print('  "Find my Mickey Mouse mug in examples/scence/"')
    print('  "What objects do you know about?"')
    print()
    print("Commands: 'quit' to exit, 'reset' to clear history, 'help' for help")
    print("=" * 65)
    print()


def print_help():
    """Print help message."""
    print()
    print("-" * 50)
    print("Help")
    print("-" * 50)
    print()
    print("Registration examples:")
    print('  "I just took a photo of my cellphone at examples/handheld/cellphone1/"')
    print('  "Remember my blue mug with Mickey Mouse, photos at examples/handheld/mug1/"')
    print()
    print("Recognition examples:")
    print('  "Find my Mickey Mouse mug in examples/scence/"')
    print('  "Where is my cellphone? Check examples/scence/"')
    print()
    print("Other:")
    print('  "What objects are registered?"')
    print('  "List all objects"')
    print()
    print("Commands:")
    print("  quit, exit, q  - Exit the agent")
    print("  reset          - Reset conversation history")
    print("  help           - Show this help message")
    print("-" * 50)
    print()


def main():
    """Main entry point."""
    print_welcome()

    # Load configuration
    try:
        config = AgentConfig.from_env()
    except ValueError as e:
        print(f"Configuration error: {e}")
        print()
        print("Please set the DEEPSEEK_API_KEY environment variable:")
        print("  export DEEPSEEK_API_KEY='your_api_key'")
        print()
        return 1

    # Initialize service
    service = AgentService(
        device=config.device,
        default_templates_path=config.default_templates_path,
        default_output_path=config.default_output_path,
        verbose=config.verbose
    )

    # Preload models at startup to avoid delays during conversation
    print()
    service.preload_models()
    print()

    # Initialize conversation agent
    agent = ConversationAgent(config=config, service=service)

    print("Agent ready! Type your message below.")
    print()

    # Conversation loop
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Handle special commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print()
                print("Goodbye!")
                break

            if user_input.lower() == 'reset':
                agent.reset_conversation()
                print("Conversation reset.")
                print()
                continue

            if user_input.lower() == 'help':
                print_help()
                continue

            # Get response from agent
            print()
            print("Assistant: ", end="", flush=True)

            response = agent.chat(user_input)

            # Print response (already printed "Assistant: " prefix)
            print(response)
            print()

        except KeyboardInterrupt:
            print()
            print()
            print("Goodbye!")
            break

        except Exception as e:
            print()
            print(f"Error: {e}")
            print("Please try again.")
            print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
