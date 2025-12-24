#!/usr/bin/env python3
"""
Test script for generating and running conversational test cases.

Usage:
    Generate test case:
        python test.py --generate_test=role_adherance --name=ananya

    Run evaluation:
        python test.py --run_test=role_adherance

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
from deepeval.dataset import ConversationalGolden
from deepeval.metrics import RoleAdherenceMetric
from deepeval.simulator import ConversationSimulator
from deepeval.test_case import ConversationalTestCase, Turn

from ira_chat import IraChat


# Verify OpenAI API key is set
if not os.environ.get("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY environment variable is not set.")
    print("Please set it in .env file or export it: export OPENAI_API_KEY=your_key")
    exit(1)


def load_chatbot_role(test_type_dir: Path) -> str:
    """Load the chatbot role from var.json."""
    var_path = test_type_dir / "var.json"
    with open(var_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["chatbot_role"]


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


def create_chatbot_callback(character_name: str):
    """Create a chatbot callback function for the given character."""
    chatbot = IraChat(name=character_name)
    
    def callback(input: str) -> Turn:
        response = chatbot.chat(input)
        return Turn(
            role="assistant",
            content=response,
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
            turns.append(Turn(
                role=turn_data["role"],
                content=turn_data["content"]
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
    else:
        print(f"Error: Unknown test type: {test_type}")
        exit(1)
    
    # Run evaluation
    print("\nRunning evaluation...")
    evaluate(test_cases=test_cases, metrics=[metric])
    
    print("\n" + "=" * 50)
    print("Evaluation complete! Opening results viewer...")
    print("=" * 50)
    
    # Run deepeval view command
    subprocess.run(["deepeval", "view"], check=False)


def generate_test_case(
    test_type: str,
    character_name: str,
    max_simulations: int = 10
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
    chatbot_callback = create_chatbot_callback(character_name.capitalize())
    
    # Create simulator and run simulation
    simulator = ConversationSimulator(model_callback=chatbot_callback)
    
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
        choices=["role_adherance"],
        help="Generate test case for a specific character (e.g., role_adherance)"
    )
    action_group.add_argument(
        "--run_test",
        type=str,
        choices=["role_adherance"],
        help="Run evaluation on all test cases (e.g., role_adherance)"
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
    
    args = parser.parse_args()
    
    if args.generate_test:
        # Generate test case mode
        if not args.name:
            print("Error: --name is required when using --generate_test")
            exit(1)
        
        print(f"Generating test case for: {args.name}")
        print(f"Test type: {args.generate_test}")
        print(f"Max simulations: {args.max_simulations}")
        print("-" * 50)
        
        generate_test_case(
            test_type=args.generate_test,
            character_name=args.name,
            max_simulations=args.max_simulations
        )
        
        print("\nDone!")
    
    elif args.run_test:
        # Run evaluation mode
        print(f"Running evaluation for: {args.run_test}")
        print(f"Threshold: {args.threshold}")
        print("-" * 50)
        
        run_test(
            test_type=args.run_test,
            threshold=args.threshold
        )
    
    return 0


if __name__ == "__main__":
    exit(main())
