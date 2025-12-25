#!/usr/bin/env python3
"""
Test script for generating and running conversational test cases.

Usage:
    Generate test case:
        python test.py --generate_test=role_adherance --name=ananya
        python test.py --generate_test=topic_adherance --name=arjun
        python test.py --generate_test=turn_context_relavancy --name=kaavya --with-fixed-memories

    Run evaluation:
        python test.py --run_test=role_adherance
        python test.py --run_test=topic_adherance
        python test.py --run_test=turn_context_relavancy
        python test.py --run_test=all  # Run all test types

Requirements:
    pip install deepeval openai python-dotenv

Environment:
    Set OPENAI_API_KEY in .env file or as environment variable
"""

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

# Try to load dotenv, but don't fail if not installed
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    print("Note: python-dotenv not installed. Using environment variables directly.")

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
from deepeval.dataset import ConversationalGolden
from deepeval.metrics import TurnContextualRelevancyMetric, RoleAdherenceMetric, TopicAdherenceMetric
from deepeval.simulator import ConversationSimulator
from deepeval.test_case import ConversationalTestCase, Turn

from ira_chat import IraChat, FIXED_MEMORIES


# Available test types
TEST_TYPES = ["role_adherance", "topic_adherance", "turn_context_relavancy"]

# All valid choices for --run_test (includes "all")
RUN_TEST_CHOICES = TEST_TYPES + ["all"]

# Verify OpenAI API key is set
if not os.environ.get("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY environment variable is not set.")
    print("Please set it in .env file or export it: export OPENAI_API_KEY=your_key")
    exit(1)


def load_var_config(test_type_dir: Path) -> dict[str, Any]:
    """Load the configuration from var.json."""
    var_path = test_type_dir / "var.json"
    with open(var_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_chatbot_role(test_type_dir: Path) -> str:
    """Load the chatbot role from var.json."""
    config = load_var_config(test_type_dir)
    return config["chatbot_role"]


def load_topics(test_type_dir: Path) -> list[str]:
    """Load the topics from var.json for TopicAdherenceMetric."""
    config = load_var_config(test_type_dir)
    return config.get("topics", [])


def load_scenario_for_character(character_dir: Path) -> dict[str, Any] | None:
    """
    Load the scenario JSON file from a character's directory.
    
    Returns the scenario data or None if no JSON file found.
    """
    for json_file in character_dir.glob("*.json"):
        # Skip files that look like test case outputs
        if "test_case" in json_file.name:
            continue
        with open(json_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def create_conversational_golden(scenario_data: dict[str, Any]) -> ConversationalGolden:
    """Create a ConversationalGolden object from scenario data."""
    turns = []
    for turn_data in scenario_data.get("turns", []):
        turns.append(Turn(
            role=turn_data["role"],
            content=turn_data["content"]
        ))
    
    return ConversationalGolden(
        scenario=scenario_data.get("scenario", ""),
        user_description=scenario_data.get("user_description", ""),
        expected_outcome=scenario_data.get("expected_outcome", ""),
        turns=turns
    )


def create_chatbot_callback(character_name: str, use_fixed_memories: bool = False):
    """Create a chatbot callback function for the given character."""
    chatbot = IraChat(name=character_name, use_fixed_memories=use_fixed_memories)
    
    def callback(input: str) -> Turn:
        chat_response = chatbot.chat(input)
        # Extract just the response text from ChatResponse
        return Turn(
            role="assistant",
            content=chat_response.response,
        )
    
    return callback


def create_chatbot_callback_with_context(character_name: str, use_fixed_memories: bool = False):
    """
    Create a chatbot callback that also captures retrieval_context.
    Used for turn_context_relavancy test type.
    """
    chatbot = IraChat(name=character_name, use_fixed_memories=use_fixed_memories)
    
    def callback(input: str) -> Turn:
        chat_response = chatbot.chat(input)
        # Extract memory texts for retrieval_context
        retrieval_context = [
            mem.get("memory", "") for mem in chat_response.relevant_memories
        ]
        return Turn(
            role="assistant",
            content=chat_response.response,
            retrieval_context=retrieval_context if retrieval_context else None
        )
    
    return callback


def load_all_test_cases(test_type: str) -> list[ConversationalTestCase]:
    """
    Load all test cases from character directories.
    
    Returns a list of ConversationalTestCase objects.
    """
    test_type_dir = Path(__file__).parent / test_type
    test_cases = []
    
    for character_dir in test_type_dir.iterdir():
        if not character_dir.is_dir():
            continue
            
        test_case_file = character_dir / "test_case.json"
        if not test_case_file.exists():
            print(f"  Warning: No test_case.json found in {character_dir.name}")
            continue
        
        with open(test_case_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        character_name = data.get("character_name", character_dir.name.capitalize())
        test_case_data = data.get("test_case", {})
        
        # Create Turn objects from the test case data
        turns = []
        for turn_data in test_case_data.get("turns", []):
            # Include retrieval_context if present (for turn_context_relavancy)
            retrieval_context = turn_data.get("retrieval_context")
            turns.append(Turn(
                role=turn_data["role"],
                content=turn_data["content"],
                retrieval_context=retrieval_context
            ))
        
        # Create ConversationalTestCase
        test_case = ConversationalTestCase(
            turns=turns,
            chatbot_role=test_case_data.get("chatbot_role", data.get("chatbot_role", ""))
        )
        
        test_cases.append(test_case)
        print(f"  Loaded test case for: {character_name} ({len(turns)} turns)")
    
    return test_cases


def run_test(test_type: str, threshold: float = 0.5) -> None:
    """
    Run evaluation on all test cases for a given test type.
    """
    test_type_dir = Path(__file__).parent / test_type
    
    if not test_type_dir.exists():
        print(f"Error: Test type directory not found: {test_type_dir}")
        exit(1)
    
    print(f"\nLoading test cases from: {test_type_dir}")
    test_cases = load_all_test_cases(test_type)
    
    if not test_cases:
        print("Error: No test cases found. Generate test cases first using --generate_test")
        exit(1)
    
    print(f"\nLoaded {len(test_cases)} test case(s)")
    print("-" * 50)
    
    # Create the metric based on test type
    if test_type == "role_adherance":
        metric = RoleAdherenceMetric(threshold=threshold)
        print(f"Using RoleAdherenceMetric with threshold: {threshold}")
    elif test_type == "topic_adherance":
        topics = load_topics(test_type_dir)
        if not topics:
            print("Error: No topics defined in var.json for topic_adherance")
            exit(1)
        metric = TopicAdherenceMetric(relevant_topics=topics, threshold=threshold, model="gpt-4o-mini")
        print(f"Using TopicAdherenceMetric with threshold: {threshold}")
        print(f"Topics: {topics}")
    elif test_type == "turn_context_relavancy":
        metric = TurnContextualRelevancyMetric(threshold=threshold, model="gpt-4o-mini")
        print(f"Using TurnContextualRelevancyMetric with threshold: {threshold}")
    else:
        print(f"Error: Unknown test type: {test_type}")
        exit(1)
    
    # Run evaluation
    print("\nRunning evaluation...")
    evaluate(test_cases=test_cases, metrics=[metric], async_config=AsyncConfig(run_async=False))
    
    print("\n" + "=" * 50)
    print("Evaluation complete! Opening results viewer...")
    print("=" * 50)
    
    # Run deepeval view command
    subprocess.run(["deepeval", "view"], check=False)


def run_all_tests(threshold: float = 0.5) -> None:
    """
    Run evaluation on all test types sequentially.
    """
    print("=" * 60)
    print("RUNNING ALL TEST TYPES")
    print("=" * 60)
    
    all_test_cases = []
    all_metrics = []
    
    for test_type in TEST_TYPES:
        test_type_dir = Path(__file__).parent / test_type
        
        if not test_type_dir.exists():
            print(f"\nWarning: Test type directory not found: {test_type_dir}")
            continue
        
        print(f"\n{'='*50}")
        print(f"Loading test cases for: {test_type}")
        print(f"{'='*50}")
        
        test_cases = load_all_test_cases(test_type)
        
        if not test_cases:
            print(f"Warning: No test cases found for {test_type}")
            continue
        
        print(f"Loaded {len(test_cases)} test case(s)")
        
        # Create the metric based on test type
        if test_type == "role_adherance":
            metric = RoleAdherenceMetric(threshold=threshold)
            print(f"Using RoleAdherenceMetric with threshold: {threshold}")
        elif test_type == "topic_adherance":
            topics = load_topics(test_type_dir)
            if not topics:
                print(f"Warning: No topics defined in var.json for {test_type}")
                continue
            metric = TopicAdherenceMetric(relevant_topics=topics, threshold=threshold, model="gpt-4o-mini")
            print(f"Using TopicAdherenceMetric with threshold: {threshold}")
        elif test_type == "turn_context_relavancy":
            metric = TurnContextualRelevancyMetric(threshold=threshold, model="gpt-4o-mini")
            print(f"Using TurnContextualRelevancyMetric with threshold: {threshold}")
        else:
            print(f"Warning: Unknown test type: {test_type}")
            continue
        
        all_test_cases.extend(test_cases)
        all_metrics.append(metric)
    
    if not all_test_cases:
        print("\nError: No test cases found in any test type directory.")
        print("Generate test cases first using --generate_test")
        exit(1)
    
    print(f"\n{'='*60}")
    print(f"RUNNING EVALUATION ON {len(all_test_cases)} TOTAL TEST CASES")
    print(f"Using {len(all_metrics)} metrics: {[type(m).__name__ for m in all_metrics]}")
    print(f"{'='*60}")
    
    # Run evaluation with all metrics
    print("\nRunning evaluation...")
    evaluate(test_cases=all_test_cases, metrics=all_metrics, async_config=AsyncConfig(run_async=False))
    
    print("\n" + "=" * 50)
    print("All evaluations complete! Opening results viewer...")
    print("=" * 50)
    
    # Run deepeval view command
    subprocess.run(["deepeval", "view"], check=False)


def generate_test_case(
    test_type: str,
    character_name: str,
    max_simulations: int = 10,
    use_fixed_memories: bool = False
) -> None:
    """
    Generate a conversational test case for a specific character.
    
    Saves the result inside the character's directory.
    """
    test_type_dir = Path(__file__).parent / test_type
    character_dir = test_type_dir / character_name.lower()
    
    # Validate character directory exists
    if not character_dir.exists():
        print(f"Error: Character directory not found: {character_dir}")
        available = [d.name for d in test_type_dir.iterdir() if d.is_dir()]
        print(f"Available characters: {', '.join(available)}")
        exit(1)
    
    # Load chatbot role
    chatbot_role = load_chatbot_role(test_type_dir)
    print(f"Loaded chatbot role: {chatbot_role[:50]}...")
    
    # Load scenario for this character
    scenario_data = load_scenario_for_character(character_dir)
    if scenario_data is None:
        print(f"Error: No scenario JSON file found in {character_dir}")
        exit(1)
    
    print(f"\nProcessing character: {character_name.capitalize()}")
    print(f"  Scenario: {scenario_data.get('scenario', 'N/A')[:60]}...")
    
    # Create ConversationalGolden object
    golden = create_conversational_golden(scenario_data)
    
    # Create chatbot callback for this character
    # Use callback with context for turn_context_relavancy to capture retrieval_context
    if test_type == "turn_context_relavancy":
        chatbot_callback = create_chatbot_callback_with_context(
            character_name.capitalize(),
            use_fixed_memories=use_fixed_memories
        )
    else:
        chatbot_callback = create_chatbot_callback(
            character_name.capitalize(),
            use_fixed_memories=use_fixed_memories
        )
    
    # Create simulator and run simulation
    simulator = ConversationSimulator(model_callback=chatbot_callback, simulator_model="gpt-4o-mini")
    
    print(f"  Running simulation with max {max_simulations} user simulations...")
    test_cases = simulator.simulate(
        conversational_goldens=[golden],
        max_user_simulations=max_simulations
    )
    
    if not test_cases:
        print(f"Error: No test cases generated for {character_name}")
        exit(1)
    
    # Get the first test case and add chatbot_role
    test_case = test_cases[0]
    test_case.chatbot_role = chatbot_role
    
    # Prepare output data
    output_data = {
        "character_name": character_name.capitalize(),
        "scenario": scenario_data.get("scenario", ""),
        "chatbot_role": chatbot_role,
        "test_case": json.loads(test_case.model_dump_json())
    }
    
    # Save to character's directory
    output_path = character_dir / "test_case.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n  Generated {len(test_case.turns)} turns")
    print(f"  Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate and run conversational test cases."
    )
    
    # Mutually exclusive group for generate vs run
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--generate_test",
        type=str,
        choices=TEST_TYPES,
        help="Generate test case for a specific character (e.g., role_adherance, topic_adherance)"
    )
    action_group.add_argument(
        "--run_test",
        type=str,
        choices=RUN_TEST_CHOICES,
        help="Run evaluation on test cases (e.g., role_adherance, topic_adherance, turn_context_relavancy, all)"
    )
    
    # Arguments for generate_test
    parser.add_argument(
        "--name",
        type=str,
        help="Character name for test case generation (e.g., ananya, arjun, kaavya, ishaan)"
    )
    parser.add_argument(
        "--max-simulations",
        type=int,
        default=10,
        help="Maximum number of user simulations per conversation (default: 10)"
    )
    
    # Arguments for run_test
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for evaluation metrics (default: 0.5)"
    )
    
    # Memory options
    parser.add_argument(
        "--with-fixed-memories",
        action="store_true",
        help="Use fixed memories instead of fetching from Mem0 API"
    )
    
    args = parser.parse_args()
    
    if args.generate_test:
        # Generate test case mode
        if not args.name:
            print("Error: --name is required when using --generate_test")
            exit(1)
        
        use_fixed_memories = getattr(args, 'with_fixed_memories', False)
        
        print(f"Generating test case for: {args.name}")
        print(f"Test type: {args.generate_test}")
        print(f"Max simulations: {args.max_simulations}")
        print(f"Using fixed memories: {use_fixed_memories}")
        print("-" * 50)
        
        generate_test_case(
            test_type=args.generate_test,
            character_name=args.name,
            max_simulations=args.max_simulations,
            use_fixed_memories=use_fixed_memories
        )
        
        print("\nDone!")
    
    elif args.run_test:
        # Run evaluation mode
        print(f"Running evaluation for: {args.run_test}")
        print(f"Threshold: {args.threshold}")
        print("-" * 50)
        
        if args.run_test == "all":
            run_all_tests(threshold=args.threshold)
        else:
            run_test(
                test_type=args.run_test,
                threshold=args.threshold
            )
    
    return 0


if __name__ == "__main__":
    exit(main())
