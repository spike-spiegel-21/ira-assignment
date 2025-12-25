# Ira Evaluation Test Suite

A comprehensive test suite for evaluating the Ira conversational AI using [DeepEval](https://github.com/confident-ai/deepeval) metrics.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Test Types](#test-types)
- [Commands](#commands)
  - [Generate Test Cases](#generate-test-cases)
  - [Run Evaluations](#run-evaluations)
- [Command Options](#command-options)
- [Directory Structure](#directory-structure)
- [Configuration Files](#configuration-files)
- [Examples](#examples)

---

## Prerequisites

### Environment Variables

Create a `.env` file in the project root with:

```bash
OPENAI_API_KEY=your_openai_api_key
MEM0_API_KEY=your_mem0_api_key  # Optional, for memory features
```

### Dependencies

```bash
pip install deepeval openai python-dotenv mem0
```

---

## Test Types

| Test Type | Metric Used | Description |
|-----------|-------------|-------------|
| `role_adherance` | `RoleAdherenceMetric` | Evaluates if the chatbot stays in character |
| `topic_adherance` | `TopicAdherenceMetric` | Evaluates if responses stay within defined topics |
| `turn_context_relavancy` | `TurnContextualRelevancyMetric` | Evaluates if responses are relevant to the retrieval context (memories) |

---

## Commands

### Generate Test Cases

Generate conversational test cases for a specific character using simulation.

```bash
python test.py --generate_test=<test_type> --name=<character_name> [options]
```

#### Examples

```bash
# Generate role adherance test case for Ananya
python test.py --generate_test=role_adherance --name=ananya

# Generate topic adherance test case for Arjun with 5 simulations
python test.py --generate_test=topic_adherance --name=arjun --max-simulations=5

# Generate turn context relevancy test case with fixed memories
python test.py --generate_test=turn_context_relavancy --name=kaavya --with-fixed-memories

# Generate with all options
python test.py --generate_test=turn_context_relavancy --name=ishaan --max-simulations=3 --with-fixed-memories
```

### Run Evaluations

Run evaluations on generated test cases.

```bash
python test.py --run_test=<test_type> [options]
```

#### Examples

```bash
# Run role adherance evaluation
python test.py --run_test=role_adherance

# Run topic adherance evaluation with custom threshold
python test.py --run_test=topic_adherance --threshold=0.7

# Run turn context relevancy evaluation
python test.py --run_test=turn_context_relavancy --threshold=0.5

# Run ALL test types at once
python test.py --run_test=all --threshold=0.7
```

---

## Command Options

### For `--generate_test`

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--name` | Yes | - | Character name (ananya, arjun, kaavya, ishaan) |
| `--max-simulations` | No | 10 | Max number of user simulation turns |
| `--with-fixed-memories` | No | False | Use fixed memories instead of Mem0 API |

### For `--run_test`

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--threshold` | No | 0.5 | Passing threshold for metrics (0.0 to 1.0) |

---

## Directory Structure

```
test_suite/
├── test.py                      # Main test script
├── ira_chat.py                  # IraChat class with Mem0 integration
├── system_prompt.txt            # System prompt template
├── TEST_SUITE.md                # This documentation
│
├── role_adherance/              # Role adherance test type
│   ├── var.json                 # Contains chatbot_role
│   ├── ananya/
│   │   ├── scenario.json        # Scenario definition
│   │   └── test_case.json       # Generated test case
│   ├── arjun/
│   ├── kaavya/
│   └── ishaan/
│
├── topic_adherance/             # Topic adherance test type
│   ├── var.json                 # Contains chatbot_role and topics
│   ├── ananya/
│   ├── arjun/
│   ├── kaavya/
│   └── ishaan/
│
└── turn_context_relavancy/      # Turn context relevancy test type
    ├── var.json                 # Contains chatbot_role
    ├── ananya/
    ├── arjun/
    ├── kaavya/
    └── ishaan/
```

---

## Configuration Files

### var.json (Role Adherance)

```json
{
  "chatbot_role": "हाय, मैं तुम्हारी human दोस्त ईरा हूँ..."
}
```

### var.json (Topic Adherance)

```json
{
  "chatbot_role": "Ira is a humanistic and witty friend...",
  "topics": [
    "emotional support",
    "mental health",
    "work-life balance",
    "stress management",
    "personal relationships",
    "self-reflection",
    "motivation",
    "creative brainstorming"
  ]
}
```

### scenario.json (Character Scenario)

```json
{
  "scenario": "Description of the scenario...",
  "expected_outcome": "What the user expects...",
  "user_description": "Description of the user persona...",
  "turns": [
    {
      "role": "user",
      "content": "Initial message from user..."
    }
  ]
}
```

---

## Examples

### Complete Workflow

```bash
# 1. Generate test cases for all characters
python test.py --generate_test=role_adherance --name=ananya --max-simulations=5
python test.py --generate_test=role_adherance --name=arjun --max-simulations=5
python test.py --generate_test=role_adherance --name=kaavya --max-simulations=5
python test.py --generate_test=role_adherance --name=ishaan --max-simulations=5

# 2. Run evaluation
python test.py --run_test=role_adherance --threshold=0.7

# 3. View results (opens Confident AI dashboard)
deepeval view
```

### Using Fixed Memories

When `--with-fixed-memories` is used, the following memories are injected:

```python
FIXED_MEMORIES = [
    {"memory": "User enjoys playing badminton on weekends."},
    {"memory": "User finds work less enjoyable lately."},
    {"memory": "User values spending quality time with family."},
    {"memory": "User prefers relaxing by watching videos online."},
    {"memory": "User tries to stay calm during stressful situations."},
]
```

This is useful for testing without requiring the Mem0 API.

### Running All Tests

```bash
# Run all test types with a single command
python test.py --run_test=all --threshold=0.7
```

This will:
1. Load test cases from all test type directories
2. Create appropriate metrics for each type
3. Run evaluation with all metrics
4. Open the results viewer

---

## Troubleshooting

### Timeout Errors

If you encounter timeout errors:

```bash
export DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE=300
```

### Missing Memories

If memories are not loading:
- Check that `MEM0_API_KEY` is set in `.env`
- Or use `--with-fixed-memories` flag

### Rate Limit Errors

Reduce `--max-simulations` or wait before retrying.

---

## Available Characters

| Character | Description |
|-----------|-------------|
| `ananya` | Freelance designer dealing with creative blocks |
| `arjun` | Senior Engineering Manager with burnout |
| `kaavya` | Grieving a 3-year relationship |
| `ishaan` | Introvert preparing for a big family wedding |

---

## Metrics Details

### RoleAdherenceMetric
- Evaluates if the chatbot maintains its defined role/persona
- Uses the `chatbot_role` from `var.json`

### TopicAdherenceMetric
- Evaluates if responses stay within predefined topics
- Uses the `topics` list from `var.json`

### TurnContextualRelevancyMetric
- Evaluates if responses are relevant to the provided retrieval context
- Each assistant turn includes `retrieval_context` (memories)
- Only available when generating with `turn_context_relavancy` test type

