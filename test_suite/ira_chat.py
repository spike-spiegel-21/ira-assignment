from pathlib import Path
from openai import OpenAI


class IraChat:
    """A chatbot class that uses OpenAI with a customized system prompt."""

    def __init__(self, name: str, model: str = "gpt-4o"):
        """
        Initialize the IraChat instance.

        Args:
            name: The user's preferred name to personalize the conversation.
            model: The OpenAI model to use (default: gpt-4o).
        """
        self.client = OpenAI()
        self.model = model
        self.messages: list[dict[str, str]] = []

        # Load and populate the system prompt
        prompt_path = Path(__file__).parent / "system_prompt.txt"
        with open(prompt_path, "r", encoding="utf-8") as f:
            system_prompt_template = f.read()

        # Populate the name placeholder in the system prompt
        self.system_prompt = system_prompt_template.replace("{user_name}", name)

        # Initialize messages with system prompt
        self.messages.append({"role": "system", "content": self.system_prompt})

    def chat(self, message: str) -> str:
        """
        Send a message and get a response from OpenAI.

        Args:
            message: The user's message string.

        Returns:
            The assistant's response as a string.
        """
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

        return assistant_message

    def reset_conversation(self) -> None:
        """Reset the conversation history, keeping only the system prompt."""
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def get_conversation_history(self) -> list[dict[str, str]]:
        """Get the full conversation history including system prompt."""
        return self.messages.copy()

