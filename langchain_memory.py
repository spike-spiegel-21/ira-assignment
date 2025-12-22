"""
Test script for ChatSummaryMemoryBuffer from LlamaIndex.
Demonstrates how the buffer summarizes older messages when token limit is exceeded.
"""

import os
from dotenv import load_dotenv
from llama_index.core.memory import ChatSummaryMemoryBuffer
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.llms.openai import OpenAI

# Load environment variables
load_dotenv()

# Custom summarization prompt
CUSTOM_SUMMARIZE_PROMPT = """You are a conversation summarizer. 
Summarize the key points from the following conversation between user and assistant.
Keep the summary concise but preserve important details, names, numbers, and decisions made.
Focus on actionable information and context that would be needed to continue the conversation."""

# Token limit for ~10 small messages (each small message is roughly 15-20 tokens)
# Setting to 150 tokens to force summarization after adding several messages
TOKEN_LIMIT = 150


def create_dummy_messages():
    """Create a list of dummy chat messages to test summarization."""
    messages = [
        ChatMessage(role=MessageRole.USER, content="Hi! My name is Alex and I'm looking for a new laptop."),
        ChatMessage(role=MessageRole.ASSISTANT, content="Hello Alex! I'd be happy to help you find a laptop. What's your budget?"),
        ChatMessage(role=MessageRole.USER, content="I have around $1500 to spend."),
        ChatMessage(role=MessageRole.ASSISTANT, content="Great budget! Are you looking for gaming, work, or general use?"),
        ChatMessage(role=MessageRole.USER, content="Mainly for software development and some light gaming."),
        ChatMessage(role=MessageRole.ASSISTANT, content="Perfect! I'd recommend looking at the MacBook Pro or Dell XPS 15."),
        ChatMessage(role=MessageRole.USER, content="I prefer Windows. What about the Dell XPS?"),
        ChatMessage(role=MessageRole.ASSISTANT, content="The Dell XPS 15 with i7 and 16GB RAM is excellent for development."),
        ChatMessage(role=MessageRole.USER, content="Does it have good battery life?"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Yes, it offers around 10-12 hours of battery life."),
        ChatMessage(role=MessageRole.USER, content="That sounds perfect. What about the display?"),
        ChatMessage(role=MessageRole.ASSISTANT, content="It has a stunning 15.6 inch OLED display with 3.5K resolution."),
        ChatMessage(role=MessageRole.USER, content="Great! I'll consider the Dell XPS 15 then."),
        ChatMessage(role=MessageRole.ASSISTANT, content="Excellent choice, Alex! Would you like me to find the best deals?"),
    ]
    return messages


def test_chat_summary_memory_buffer():
    """Test the ChatSummaryMemoryBuffer with dummy messages."""
    
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")
    
    # Initialize OpenAI LLM for summarization
    llm = OpenAI(
        model="gpt-4o-mini",
        api_key=api_key,
        temperature=0.3  # Lower temperature for more consistent summaries
    )
    
    # Create the ChatSummaryMemoryBuffer
    memory = ChatSummaryMemoryBuffer.from_defaults(
        llm=llm,
        token_limit=TOKEN_LIMIT,
        summarize_prompt=CUSTOM_SUMMARIZE_PROMPT,
        count_initial_tokens=False
    )
    
    print("=" * 60)
    print("ChatSummaryMemoryBuffer Test")
    print("=" * 60)
    print(f"Token Limit: {TOKEN_LIMIT}")
    print(f"Summarize Prompt: {CUSTOM_SUMMARIZE_PROMPT[:50]}...")
    print("=" * 60)
    
    # Add dummy messages one by one
    dummy_messages = create_dummy_messages()
    
    print(f"\n📝 Adding {len(dummy_messages)} dummy messages to memory...\n")
    
    for i, msg in enumerate(dummy_messages):
        memory.put(msg)
        print(f"  [{i+1}] {msg.role.value}: {msg.content[:50]}...")
    
    print("\n" + "=" * 60)
    print("BEFORE SUMMARIZATION - All messages in buffer:")
    print("=" * 60)
    all_messages = memory.get_all()
    print(f"Total messages in buffer: {len(all_messages)}")
    for i, msg in enumerate(all_messages):
        print(f"  [{i+1}] {msg.role.value}: {msg.content}")
    
    print("\n" + "=" * 60)
    print("AFTER CALLING get() - Triggering summarization:")
    print("=" * 60)
    
    # The get() method triggers summarization when messages exceed token limit
    processed_messages = memory.get()
    
    print(f"Messages after processing: {len(processed_messages)}")
    print(f"Token count after processing: {memory.get_token_count()}")
    print()
    
    for i, msg in enumerate(processed_messages):
        role_icon = "🤖" if msg.role == MessageRole.ASSISTANT else "👤" if msg.role == MessageRole.USER else "📋"
        print(f"  {role_icon} [{msg.role.value}]:")
        print(f"     {msg.content}")
        print()
    
    # Check if summarization occurred
    print("=" * 60)
    print("ANALYSIS:")
    print("=" * 60)
    
    if len(processed_messages) < len(dummy_messages):
        print("✅ SUMMARIZATION OCCURRED!")
        print(f"   Original message count: {len(dummy_messages)}")
        print(f"   Processed message count: {len(processed_messages)}")
        print(f"   Messages summarized: {len(dummy_messages) - len(processed_messages) + 1}")
        
        # The first message should be the summary (system message)
        if processed_messages and processed_messages[0].role == MessageRole.SYSTEM:
            print("\n📋 SUMMARIZED MESSAGE (System):")
            print("-" * 40)
            print(processed_messages[0].content)
            print("-" * 40)
    else:
        print("⚠️  No summarization occurred - all messages fit within token limit.")
        print("   Try reducing TOKEN_LIMIT or adding more messages.")
    
    return memory, processed_messages


def test_incremental_summarization():
    """Test how summarization works as messages are added incrementally."""
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
    
    llm = OpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0.3)
    
    # Use a very small token limit to force frequent summarization
    memory = ChatSummaryMemoryBuffer.from_defaults(
        llm=llm,
        token_limit=100,  # Very small limit
        summarize_prompt=CUSTOM_SUMMARIZE_PROMPT
    )
    
    print("\n" + "=" * 60)
    print("INCREMENTAL SUMMARIZATION TEST (token_limit=100)")
    print("=" * 60)
    
    test_exchanges = [
        ("What's the weather like?", "It's sunny and 72°F today."),
        ("Should I bring an umbrella?", "No need, no rain expected."),
        ("What about tomorrow?", "Tomorrow looks cloudy with a chance of rain."),
        ("Thanks for the info!", "You're welcome! Have a great day!"),
    ]
    
    for i, (user_msg, assistant_msg) in enumerate(test_exchanges):
        print(f"\n--- Exchange {i+1} ---")
        memory.put(ChatMessage(role=MessageRole.USER, content=user_msg))
        memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=assistant_msg))
        
        # Get processed messages after each exchange
        processed = memory.get()
        print(f"Messages after get(): {len(processed)}")
        
        if processed and processed[0].role == MessageRole.SYSTEM:
            print(f"📋 Current summary: {processed[0].content[:100]}...")
    
    print("\n--- Final State ---")
    final_messages = memory.get_all()
    print(f"Total messages: {len(final_messages)}")
    for msg in final_messages:
        print(f"  [{msg.role.value}]: {msg.content[:80]}...")


if __name__ == "__main__":
    print("\n🚀 Starting ChatSummaryMemoryBuffer Tests\n")
    
    try:
        # Run main test
        memory, processed = test_chat_summary_memory_buffer()
        
        # Run incremental test
        test_incremental_summarization()
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
