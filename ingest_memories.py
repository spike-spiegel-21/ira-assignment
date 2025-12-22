"""
Script to ingest messages from a conversation JSON file into Mem0.

This script reads messages from the conversation.json file and adds them to Mem0
in batches of 6 messages, with a random delay between batches.
"""

import json
import os
import random
import time
from pathlib import Path
from dotenv import load_dotenv
from mem0 import MemoryClient

# Load environment variables
load_dotenv()

# Path to the conversation JSON file
CONVERSATION_JSON_PATH = Path(__file__).parent / "recordings" / "20251223_001931" / "conversation.json"

# Batch configuration
BATCH_SIZE = 6
MIN_DELAY_SECONDS = 10
MAX_DELAY_SECONDS = 20


def main():
    """Load all messages from conversation.json and add them to Mem0 in batches of 6."""
    
    # Get Mem0 API key from environment
    mem0_api_key = os.getenv("MEM0_API_KEY")
    if not mem0_api_key:
        raise ValueError(
            "MEM0_API_KEY not found in environment variables. "
            "Please set it in your .env file or export it."
        )
    
    # Initialize Mem0 client
    client = MemoryClient(api_key=mem0_api_key)
    
    # Load the conversation JSON file
    print(f"Loading conversation from: {CONVERSATION_JSON_PATH}")
    with open(CONVERSATION_JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get all messages
    all_messages = data['messages']
    total_messages = len(all_messages)
    total_batches = (total_messages + BATCH_SIZE - 1) // BATCH_SIZE  # Ceiling division
    
    print(f"\n📊 Total messages: {total_messages}")
    print(f"📦 Batch size: {BATCH_SIZE}")
    print(f"🔄 Total batches: {total_batches}")
    print(f"⏱️  Delay between batches: {MIN_DELAY_SECONDS}-{MAX_DELAY_SECONDS} seconds")
    print("=" * 60)
    
    # Process messages in batches
    for batch_num in range(total_batches):
        start_idx = batch_num * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, total_messages)
        batch_messages = all_messages[start_idx:end_idx]
        
        print(f"\n🔄 Processing batch {batch_num + 1}/{total_batches} (messages {start_idx + 1}-{end_idx})")
        
        # Show batch contents
        for i, msg in enumerate(batch_messages):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')[:50]  # Preview first 50 chars
            print(f"   {start_idx + i + 1}. [{role}]: {content}...")
        
        # Format messages for Mem0
        formatted_messages = []
        for msg in batch_messages:
            formatted_messages.append({
                "role": msg.get("role"),
                "content": msg.get("content")
            })
        
        # Add batch to Mem0
        try:
            result = client.add(
                messages=formatted_messages,
                user_id="mayank",
            )
            
            print(f"   ✅ Batch {batch_num + 1} added successfully!")
            
            # Show progress bar
            progress = (batch_num + 1) / total_batches
            bar_length = 40
            filled = int(bar_length * progress)
            bar = "█" * filled + "░" * (bar_length - filled)
            print(f"   Progress: [{bar}] {progress * 100:.1f}%")
            
        except Exception as e:
            print(f"   ❌ Error adding batch {batch_num + 1}: {e}")
            raise
        
        # Add random delay between batches (except after the last batch)
        if batch_num < total_batches - 1:
            delay = random.uniform(MIN_DELAY_SECONDS, MAX_DELAY_SECONDS)
            print(f"   ⏳ Waiting {delay:.1f} seconds before next batch...")
            time.sleep(delay)
    
    print("\n" + "=" * 60)
    print(f"🎉 Successfully ingested all {total_messages} messages in {total_batches} batches!")
    print("=" * 60)


if __name__ == "__main__":
    main()
