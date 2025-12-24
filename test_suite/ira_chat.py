import os
from pathlib import Path
from typing import NamedTuple

from dotenv import load_dotenv
from mem0 import MemoryClient
from openai import OpenAI

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")


class ChatResponse(NamedTuple):
    """Response from the chat method containing the message and relevant memories."""
    response: str
    relevant_memories: list[dict]

FIXED_MEMORIES = [
    {"memory": "User enjoys playing badminton on weekends."},
    {"memory": "User finds work less enjoyable lately."},
    {"memory": "User values spending quality time with family."},
    {"memory": "User prefers relaxing by watching videos online."},
    {"memory": "User tries to stay calm during stressful situations."},
]

class IraChat:
    """A chatbot class that uses OpenAI with a customized system prompt and Mem0 memories."""

    def __init__(
        self,
        name: str,
        model: str = "gpt-4o",
        user_id: str | None = None,
        use_fixed_memories: bool = False
    ):
        """
        Initialize the IraChat instance.

        Args:
            name: The user's preferred name to personalize the conversation.
            model: The OpenAI model to use (default: gpt-4o).
            user_id: The user ID for Mem0 memory lookup (default: lowercase name).
            use_fixed_memories: If True, use FIXED_MEMORIES instead of fetching from Mem0.
        """
        self.client = OpenAI()
        self.model = model
        self.name = name
        self.user_id = user_id or name.lower()
        self.use_fixed_memories = use_fixed_memories
        self.messages: list[dict[str, str]] = []
        
        # Initialize Mem0 client (only if not using fixed memories)
        if not use_fixed_memories:
            mem0_api_key = os.getenv("MEM0_API_KEY")
            self.memory_client = MemoryClient(api_key=mem0_api_key) if mem0_api_key else None
            
            if not self.memory_client:
                print("Warning: MEM0_API_KEY not set. Memory features will be disabled.")
        else:
            self.memory_client = None
            print("Using fixed memories instead of Mem0 API.")

        # Load and populate the system prompt
        prompt_path = Path(__file__).parent / "system_prompt.txt"
        with open(prompt_path, "r", encoding="utf-8") as f:
            system_prompt_template = f.read()

        # Fetch user memories from Mem0
        user_memories = self._fetch_all_memories()
        
        # Populate placeholders in the system prompt
        self.system_prompt = (
            system_prompt_template
            .replace("{user_name}", name)
            .replace("{user_memories}", user_memories)
            .replace("{history_summary}", "No previous conversation summary available.")
        )

        # Initialize messages with system prompt
        self.messages.append({"role": "system", "content": self.system_prompt})

    def _fetch_all_memories(self) -> str:
        """
        Fetch all memories for the user from Mem0 or use fixed memories.
        
        Returns:
            A formatted string of all user memories.
        """
        # Use fixed memories if flag is set
        if self.use_fixed_memories:
            memory_list = []
            for memory in FIXED_MEMORIES:
                memory_text = memory.get('memory', '')
                if memory_text:
                    memory_list.append(f"- {memory_text}")
            return "\n".join(memory_list) if memory_list else "No memories available."
        
        # Otherwise fetch from Mem0
        if not self.memory_client:
            return "No memories available."
        
        try:
            memories = self.memory_client.get_all(
                filters={
                    "AND": [
                        {"user_id": self.user_id}
                    ]
                }
            )
            
            if not memories or not memories.get('results'):
                return "No memories available."
            
            # Format memories as a list
            memory_list = []
            for memory in memories['results']:
                memory_text = memory.get('memory', '')
                if memory_text:
                    memory_list.append(f"- {memory_text}")
            
            if not memory_list:
                return "No memories available."
            
            return "\n".join(memory_list)
            
        except Exception as e:
            print(f"Warning: Failed to fetch memories from Mem0: {e}")
            return "No memories available."

    def _search_relevant_memories(self, query: str, limit: int = 5) -> list[dict]:
        """
        Search for relevant memories based on the query.
        
        Args:
            query: The search query (usually the user's message).
            limit: Maximum number of memories to return.
            
        Returns:
            A list of relevant memory dictionaries.
        """
        # Return fixed memories if flag is set
        if self.use_fixed_memories:
            return FIXED_MEMORIES[:limit]
        
        if not self.memory_client:
            return []
        
        try:
            search_results = self.memory_client.search(
                query=query,
                user_id=self.user_id,
                limit=limit
            )
            
            if not search_results or not search_results.get('results'):
                return []
            
            return search_results['results']
            
        except Exception as e:
            print(f"Warning: Failed to search memories from Mem0: {e}")
            return []

    def chat(self, message: str) -> ChatResponse:
        """
        Send a message and get a response from OpenAI.

        Args:
            message: The user's message string.

        Returns:
            A ChatResponse containing:
            - response: The assistant's response as a string.
            - relevant_memories: List of relevant memories from Mem0.
        """
        # Search for relevant memories based on the user's message
        relevant_memories = self._search_relevant_memories(message)
        
        # Add user message to history
        self.messages.append({"role": "user", "content": message})

        # Call OpenAI API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
        )

        # Extract assistant response
        assistant_message = response.choices[0].message.content or ""

        # Add assistant response to history
        self.messages.append({"role": "assistant", "content": assistant_message})

        return ChatResponse(
            response=assistant_message,
            relevant_memories=relevant_memories
        )

    def reset_conversation(self) -> None:
        """Reset the conversation history, keeping only the system prompt."""
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def get_conversation_history(self) -> list[dict[str, str]]:
        """Get the full conversation history including system prompt."""
        return self.messages.copy()
    
    def get_all_memories(self) -> list[dict]:
        """
        Get all memories for the current user.
        
        Returns:
            A list of all memory dictionaries.
        """
        if not self.memory_client:
            return []
        
        try:
            memories = self.memory_client.get_all(
                filters={
                    "AND": [
                        {"user_id": self.user_id}
                    ]
                }
            )
            return memories.get('results', []) if memories else []
        except Exception as e:
            print(f"Warning: Failed to get all memories: {e}")
            return []
