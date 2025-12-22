"""
Async Conversation Memory Manager

A wrapper class that manages conversation state and generates summaries asynchronously.
When messages are added, it automatically summarizes older messages beyond the token limit.
"""

import os
import asyncio
from typing import List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

import tiktoken


class MessageRole(str, Enum):
    """Role of a chat message."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class ChatMessage:
    """A single chat message."""
    role: MessageRole
    content: str
    
    def to_dict(self) -> dict:
        return {"role": self.role.value, "content": self.content}


@dataclass
class ConversationState:
    """Current state of the conversation memory."""
    summary: Optional[str] = None
    recent_messages: List[ChatMessage] = field(default_factory=list)
    total_messages_processed: int = 0
    messages_summarized: int = 0


class AsyncConversationMemory:
    """
    Async conversation memory manager that automatically summarizes older messages.
    
    When messages are added via `put()`, it checks if the token limit is exceeded.
    If so, it asynchronously summarizes the older messages and stores the summary.
    
    When `get()` is called, it returns the summary (if any) plus recent messages
    that fit within the token limit.
    
    Usage:
        memory = AsyncConversationMemory(
            api_key=os.getenv("OPENAI_API_KEY"),
            token_limit=500
        )
        
        await memory.put(ChatMessage(role=MessageRole.USER, content="Hello!"))
        await memory.put(ChatMessage(role=MessageRole.ASSISTANT, content="Hi there!"))
        
        # Get recent messages + summary
        messages = memory.get()
        
        # Get just the summary
        summary = memory.get_summary()
    """
    
    DEFAULT_SUMMARIZE_PROMPT = """You are a conversation summarizer. 
Summarize the key points from the following conversation between user and assistant.
Keep the summary concise but preserve important details, names, numbers, and decisions made.
Focus on actionable information and context needed to continue the conversation."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        token_limit: int = 500,
        model: str = "gpt-4o-mini",
        summarize_prompt: Optional[str] = None,
        tokenizer_model: str = "gpt-4o",
        auto_summarize: bool = True,
    ):
        """
        Initialize the async conversation memory.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            token_limit: Maximum tokens for recent messages before summarization
            model: Model to use for summarization
            summarize_prompt: Custom prompt for summarization
            tokenizer_model: Model to use for token counting
            auto_summarize: If True, automatically summarize on put() when limit exceeded
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key.")
        
        self.token_limit = token_limit
        self.model = model
        self.summarize_prompt = summarize_prompt or self.DEFAULT_SUMMARIZE_PROMPT
        self.auto_summarize = auto_summarize
        
        # Initialize tokenizer
        try:
            self._tokenizer = tiktoken.encoding_for_model(tokenizer_model)
        except KeyError:
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # State
        self._messages: List[ChatMessage] = []
        self._summary: Optional[str] = None
        self._summary_lock = asyncio.Lock()
        self._is_summarizing = False
        self._total_messages_processed = 0
        self._messages_summarized = 0
        
        # OpenAI client (lazy loaded)
        self._client = None
    
    def _get_client(self):
        """Lazy load the OpenAI async client."""
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self.api_key)
        return self._client
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in a string."""
        return len(self._tokenizer.encode(text))
    
    def _count_message_tokens(self, messages: List[ChatMessage]) -> int:
        """Count total tokens in a list of messages."""
        if not messages:
            return 0
        combined = " ".join(f"{m.role.value}: {m.content}" for m in messages)
        return self._count_tokens(combined)
    
    def _get_current_token_count(self) -> int:
        """Get current token count of all messages."""
        return self._count_message_tokens(self._messages)
    
    def _split_messages_by_token_limit(self) -> tuple[List[ChatMessage], List[ChatMessage]]:
        """
        Split messages into recent (within token limit) and older (to be summarized).
        
        Returns:
            Tuple of (recent_messages, older_messages_to_summarize)
        """
        if not self._messages:
            return [], []
        
        recent_messages: List[ChatMessage] = []
        current_tokens = 0
        
        # Traverse from newest to oldest
        for msg in reversed(self._messages):
            msg_tokens = self._count_message_tokens([msg])
            if current_tokens + msg_tokens <= self.token_limit:
                recent_messages.insert(0, msg)
                current_tokens += msg_tokens
            else:
                break
        
        # Messages to summarize are those not in recent
        older_count = len(self._messages) - len(recent_messages)
        older_messages = self._messages[:older_count]
        
        return recent_messages, older_messages
    
    def _format_messages_for_summary(self, messages: List[ChatMessage]) -> str:
        """Format messages into a transcript for summarization."""
        transcript = "Transcript:\n"
        for msg in messages:
            transcript += f"{msg.role.value}: {msg.content}\n\n"
        return transcript
    
    async def _generate_summary_async(self, messages: List[ChatMessage]) -> str:
        """Generate a summary of messages asynchronously using OpenAI."""
        client = self._get_client()
        
        transcript = self._format_messages_for_summary(messages)
        
        # Include existing summary in the prompt if available
        if self._summary:
            user_content = f"Previous conversation summary:\n{self._summary}\n\nNew messages to incorporate:\n{transcript}\n\nGenerate an updated comprehensive summary."
        else:
            user_content = transcript
        
        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.summarize_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    async def put(self, message: ChatMessage) -> None:
        """
        Add a message to the conversation memory.
        
        If auto_summarize is True and the token limit is exceeded,
        this will automatically trigger async summarization of older messages.
        
        Args:
            message: The chat message to add
        """
        self._messages.append(message)
        self._total_messages_processed += 1
        
        # Check if we need to summarize
        if self.auto_summarize:
            current_tokens = self._get_current_token_count()
            if current_tokens > self.token_limit:
                await self._summarize_if_needed()
    
    async def _summarize_if_needed(self) -> None:
        """Summarize older messages if token limit is exceeded."""
        async with self._summary_lock:
            if self._is_summarizing:
                return
            
            recent_messages, older_messages = self._split_messages_by_token_limit()
            
            if not older_messages:
                return
            
            self._is_summarizing = True
            
            try:
                # Generate summary of older messages
                new_summary = await self._generate_summary_async(older_messages)
                
                # Update state
                self._summary = new_summary
                self._messages_summarized += len(older_messages)
                
                # Keep only recent messages
                self._messages = recent_messages
                
            finally:
                self._is_summarizing = False
    
    async def force_summarize(self) -> Optional[str]:
        """
        Force summarization of older messages, regardless of token limit.
        
        Returns:
            The generated summary, or None if no messages to summarize
        """
        recent_messages, older_messages = self._split_messages_by_token_limit()
        
        if not older_messages:
            return self._summary
        
        async with self._summary_lock:
            self._is_summarizing = True
            try:
                new_summary = await self._generate_summary_async(older_messages)
                self._summary = new_summary
                self._messages_summarized += len(older_messages)
                self._messages = recent_messages
                return new_summary
            finally:
                self._is_summarizing = False
    
    def get(self) -> List[ChatMessage]:
        """
        Get all messages for the LLM context.
        
        Returns a list with:
        1. Summary as a SYSTEM message (if exists)
        2. All recent messages within token limit
        
        Returns:
            List of ChatMessage objects ready for LLM context
        """
        result: List[ChatMessage] = []
        
        # Add summary as system message if it exists
        if self._summary:
            result.append(ChatMessage(
                role=MessageRole.SYSTEM,
                content=f"Previous conversation summary:\n{self._summary}"
            ))
        
        # Add recent messages
        result.extend(self._messages)
        
        return result
    
    def get_as_dicts(self) -> List[dict]:
        """Get messages as list of dicts for OpenAI API."""
        return [msg.to_dict() for msg in self.get()]
    
    def get_summary(self) -> Optional[str]:
        """Get just the current summary."""
        return self._summary
    
    def get_recent_messages(self) -> List[ChatMessage]:
        """Get only the recent messages (without summary)."""
        return self._messages.copy()
    
    def get_state(self) -> ConversationState:
        """Get the current state of the conversation memory."""
        return ConversationState(
            summary=self._summary,
            recent_messages=self._messages.copy(),
            total_messages_processed=self._total_messages_processed,
            messages_summarized=self._messages_summarized
        )
    
    def get_token_count(self) -> int:
        """Get current token count of recent messages."""
        return self._count_message_tokens(self._messages)
    
    def reset(self) -> None:
        """Reset all memory state."""
        self._messages = []
        self._summary = None
        self._total_messages_processed = 0
        self._messages_summarized = 0
    
    def clear_summary(self) -> None:
        """Clear only the summary, keeping recent messages."""
        self._summary = None
    
    @property
    def is_summarizing(self) -> bool:
        """Check if a summarization is currently in progress."""
        return self._is_summarizing
    
    @property
    def has_summary(self) -> bool:
        """Check if a summary exists."""
        return self._summary is not None
    
    def put_nowait(self, message: ChatMessage) -> asyncio.Task:
        """
        Add a message to memory in a non-blocking way (fire and forget).
        
        This method schedules the put() coroutine as a background task
        and returns immediately. Useful for pipeline integration where
        you don't want to block the frame processing.
        
        Args:
            message: The chat message to add
            
        Returns:
            The asyncio Task object (can be ignored or awaited later)
        """
        return asyncio.create_task(self.put(message))
    
    def get_context_for_llm(self, system_prompt: Optional[str] = None) -> List[dict]:
        """
        Get context formatted for LLM, optionally with a system prompt prepended.
        
        This returns messages in the format expected by LLMContext:
        1. System prompt (if provided)
        2. Summary as system message (if exists)
        3. Recent messages within token limit
        
        Args:
            system_prompt: Optional system prompt to prepend
            
        Returns:
            List of message dicts ready for LLMContext.set_messages()
        """
        result: List[dict] = []
        
        # Add system prompt first if provided
        if system_prompt:
            result.append({"role": "system", "content": system_prompt})
        
        # Add summary as system message if it exists
        if self._summary:
            result.append({
                "role": "system", 
                "content": f"Previous conversation summary:\n{self._summary}"
            })
        
        # Add recent messages
        for msg in self._messages:
            result.append(msg.to_dict())
        
        return result


# ============================================================================
# Test Script
# ============================================================================

async def test_async_memory():
    """Test the AsyncConversationMemory class."""
    from dotenv import load_dotenv
    load_dotenv()
    
    print("=" * 60)
    print("AsyncConversationMemory Test")
    print("=" * 60)
    
    # Create memory with small token limit to force summarization
    memory = AsyncConversationMemory(
        token_limit=150,  # Small limit to force summarization
        model="gpt-4o-mini",
        summarize_prompt="""Summarize this conversation concisely. 
Keep names, numbers, and key decisions. Be brief."""
    )
    
    print(f"Token Limit: {memory.token_limit}")
    print(f"Auto Summarize: {memory.auto_summarize}")
    print()
    
    # Simulate a conversation
    conversation = [
        (MessageRole.USER, "Hi! My name is Alex and I need help planning a trip."),
        (MessageRole.ASSISTANT, "Hello Alex! I'd love to help you plan your trip. Where are you thinking of going?"),
        (MessageRole.USER, "I want to visit Japan for 2 weeks in April."),
        (MessageRole.ASSISTANT, "April is perfect for cherry blossoms! What cities do you want to visit?"),
        (MessageRole.USER, "Tokyo and Kyoto for sure. Maybe Osaka too."),
        (MessageRole.ASSISTANT, "Great choices! I'd suggest 5 days in Tokyo, 4 in Kyoto, and 3 in Osaka."),
        (MessageRole.USER, "What about my budget? I have around $5000."),
        (MessageRole.ASSISTANT, "For 2 weeks with $5000, you can stay at mid-range hotels and enjoy local food."),
        (MessageRole.USER, "Should I get a JR Pass?"),
        (MessageRole.ASSISTANT, "Absolutely! The 14-day JR Pass costs about $450 and covers most train travel."),
        (MessageRole.USER, "What about flights?"),
        (MessageRole.ASSISTANT, "Flights from the US average $800-1200. Book early for better prices."),
    ]
    
    print("📝 Adding messages to memory...\n")
    
    for i, (role, content) in enumerate(conversation):
        msg = ChatMessage(role=role, content=content)
        await memory.put(msg)
        
        state = memory.get_state()
        token_count = memory.get_token_count()
        
        print(f"  [{i+1:02d}] {role.value}: {content[:40]}...")
        print(f"       Tokens: {token_count} | Has Summary: {memory.has_summary} | Summarized: {state.messages_summarized}")
        
        if memory.has_summary and i > 0:
            # Show summary was just generated
            if state.messages_summarized > 0 and i == len([m for m in conversation[:i+1]]):
                print(f"       ✅ Summary generated!")
        print()
    
    print("=" * 60)
    print("FINAL STATE")
    print("=" * 60)
    
    state = memory.get_state()
    print(f"\nTotal messages processed: {state.total_messages_processed}")
    print(f"Messages summarized: {state.messages_summarized}")
    print(f"Recent messages kept: {len(state.recent_messages)}")
    print(f"Current token count: {memory.get_token_count()}")
    
    print("\n" + "-" * 40)
    print("📋 SUMMARY:")
    print("-" * 40)
    if state.summary:
        print(state.summary)
    else:
        print("(No summary yet)")
    
    print("\n" + "-" * 40)
    print("💬 RECENT MESSAGES:")
    print("-" * 40)
    for msg in state.recent_messages:
        icon = "👤" if msg.role == MessageRole.USER else "🤖"
        print(f"  {icon} {msg.role.value}: {msg.content}")
    
    print("\n" + "-" * 40)
    print("📦 FULL CONTEXT FOR LLM (get()):")
    print("-" * 40)
    context = memory.get()
    for msg in context:
        icon = "📋" if msg.role == MessageRole.SYSTEM else "👤" if msg.role == MessageRole.USER else "🤖"
        print(f"  {icon} [{msg.role.value}]:")
        print(f"     {msg.content[:100]}{'...' if len(msg.content) > 100 else ''}")
        print()
    
    print("=" * 60)
    print("✅ Test completed!")
    print("=" * 60)
    
    return memory


async def test_manual_summarize():
    """Test manual/forced summarization."""
    from dotenv import load_dotenv
    load_dotenv()
    
    print("\n" + "=" * 60)
    print("Manual Summarization Test (auto_summarize=False)")
    print("=" * 60)
    
    # Create memory without auto-summarization, with very small token limit
    memory = AsyncConversationMemory(
        token_limit=40,  # Very small to force summarization
        auto_summarize=False  # Disable auto summarization
    )
    
    # Add several messages
    messages = [
        (MessageRole.USER, "What's the capital of France?"),
        (MessageRole.ASSISTANT, "The capital of France is Paris."),
        (MessageRole.USER, "What's the population?"),
        (MessageRole.ASSISTANT, "Paris has about 2.1 million people in the city proper."),
        (MessageRole.USER, "What about the metro area?"),
        (MessageRole.ASSISTANT, "The Paris metropolitan area has about 12 million people."),
    ]
    
    for role, content in messages:
        await memory.put(ChatMessage(role=role, content=content))
    
    print(f"\nMessages added: {len(messages)}")
    print(f"Token count: {memory.get_token_count()}")
    print(f"Has summary: {memory.has_summary}")
    print(f"Token limit: {memory.token_limit}")
    
    print("\n🔄 Calling force_summarize()...")
    summary = await memory.force_summarize()
    
    print(f"\n📋 Generated Summary:")
    print("-" * 40)
    print(summary)
    print("-" * 40)
    
    state = memory.get_state()
    print(f"\nAfter summarization:")
    print(f"  Recent messages: {len(state.recent_messages)}")
    print(f"  Messages summarized: {state.messages_summarized}")
    print(f"  New token count: {memory.get_token_count()}")


if __name__ == "__main__":
    import asyncio
    
    async def main():
        await test_async_memory()
        await test_manual_summarize()
    
    asyncio.run(main())

