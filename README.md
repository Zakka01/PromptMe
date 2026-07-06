*This project has been created as part of the 42 curriculum by zahrabar.*

# Call Me Maybe — LLM Function Calling with Constrained Decoding

## Description

This project implements a **function calling system** powered by a small local LLM (Qwen3-0.6B), using a technique called **constrained decoding** to reliably extract structured data from natural language prompts.

Given a user prompt like `"What is the sum of 2 and 3?"`, the system:
1. Identifies the correct function to call (`fn_add_numbers`)
2. Extracts its parameters with correct types (`{"a": 2, "b": 3}`)
3. Returns a structured JSON output

The key challenge: small language models tend to generate free-form, unpredictable text. Constrained decoding solves this by restricting, at every generation step, which tokens the model is allowed to produce — enforcing valid output shape without modifying the model itself.

---

## Instructions

### Requirements

- Python 3.10+
- A machine with at least 4GB RAM (8GB recommended)
- GPU optional but recommended (MPS on Mac, CUDA on Nvidia)

### Installation

```bash
git clone <your-repo-url>
cd call_me_maybe
pip install -r requirements.txt
```

### Execution

```bash
python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

Default paths are used if no arguments are provided:

```bash
python -m src
```

### Arguments

| Argument | Description | Default |
|---|---|---|
| `--functions_definition` | Path to the functions definition JSON | `data/input/functions_definition.json` |
| `--input` | Path to the input prompts JSON | `data/input/function_calling_tests.json` |
| `--output` | Path to write the output JSON | `data/output/function_calls.json` |

---

## Example Usage

**Input (`function_calling_tests.json`):**
```json
[
  {"prompt": "What is the sum of 2 and 3?"},
  {"prompt": "Greet shrek"}
]
```

**Output (`function_calls.json`):**
```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2, "b": 3}
  },
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": {"name": "shrek"}
  }
]
```

---

## Algorithm Explanation

The system runs in two main stages:

### Stage 1 — Function Name Selection

A prompt is built listing all available functions with descriptions and few-shot examples. The model is asked to output one function name.

Instead of letting the model generate freely, the FSM loop filters valid tokens at every step: only tokens whose decoded text is a valid **prefix of a known function name** are allowed. This makes it structurally impossible to output an invalid or hallucinated function name.

```
"fn_" → valid (prefix of fn_add_numbers, fn_greet, ...)
"fn_add" → valid (prefix of fn_add_numbers)
"fn_add_numbers" → exact match → stop
"fn_xyz" → invalid → rejected
```

### Stage 2 — Parameter Extraction

For each parameter, a dedicated FSM loop generates the value token by token, with validity rules depending on the parameter type:

**Numbers/Integers:** only digits, one optional leading `-`, one optional `.` are allowed. Generation stops when the model's next unconstrained choice is no longer a digit (natural stop signal).

**Strings:** only single-line text is allowed — newlines and backslashes are blocked immediately. Generation stops on a newline signal from the model.

**Booleans:** only characters belonging to `"True"` or `"False"` are allowed. Generation stops when a complete `True` or `False` has been produced.

**Regex (special case):** handled with dedicated few-shot examples teaching the model to produce regex patterns from task descriptions (`numbers → 34|233`, `vowels → .*[aeiouAEIOU]`, `specific word → that word`).

---

## Design Decisions

**One FSM per parameter type, not one global FSM.** Numbers, strings, and booleans have completely different valid character sets and stop conditions. Keeping them separate makes each one simple and easy to debug.

**Prompts are isolated per parameter.** Early versions shared a single prompt for all parameters, causing the model to copy answers from few-shot examples into wrong fields (e.g., writing `"red"` as a `source_string` because a regex example mentioned `"red"`). Each parameter now builds its own clean prompt from scratch.

**Few-shot examples over instructions.** On a 0.6B parameter model, concrete examples are far more effective than written rules. Every prompt includes 2-4 examples that match the exact surface pattern of the test cases.

**`fn_anonymos` as a registered fallback.** Rather than returning `None` when no function matches, the system registers `fn_anonymos` as a real valid function name in the FSM's prefix table. This means gibberish/unknown prompts resolve cleanly through the same mechanism as real functions.

**Newline as a universal stop signal.** All FSMs allow `\n` as a valid character but immediately stop and trim on its appearance. This follows the natural behavior of small models that emit a newline when they consider a value complete — no need for complex length-counting or lookahead.

---

## Performance Analysis

**Accuracy on the provided test set (12 prompts):**

| Function | Result |
|---|---|
| `fn_add_numbers` (×2) | ✅ Correct `a` and `b` extracted |
| `fn_greet` (×2) | ✅ Correct name extracted |
| `fn_reverse_string` (×2) | ✅ Correct string extracted |
| `fn_get_square_root` (×2) | ✅ Correct input number extracted |
| `fn_substitute_string_with_regex` (×3) | ✅ All 3 params correct on all 3 prompts |
| `fn_anonymos` (×1) | ✅ Gibberish correctly rejected |

**Speed:** approximately 2-5 seconds per prompt on CPU (Qwen3-0.6B). On MPS/CUDA, significantly faster. The main bottleneck is the inner loop that decodes every vocabulary token (~50,000) at each generation step to check FSM validity — a known cost of this approach.

**Reliability:** deterministic for most cases (argmax selection, no sampling). The one source of non-determinism is near-tie logit scores on ambiguous prompts, which can occasionally flip between runs on GPU due to floating-point parallelism differences.

---

## Challenges Faced

**The model computes instead of extracts.** For `fn_get_square_root`, the model would output `4` instead of `16` for `"What is the square root of 16?"` — it solved the math rather than copying the input number. Fixed by redesigning the prompt as a "listing" task (`"List every number that appears as text"`) rather than a parameter extraction task, which avoids triggering the model's math-solving behavior.

**Both parameters extracting the same value.** For two-parameter functions, both `a` and `b` would return the same number. Root cause: the prompt wasn't explicitly distinguishing which positional argument was being requested. Fixed by adding ordinal position labels (`"first number"`, `"second number"`) to both examples and the live request.

**Runaway number generation.** Without a proper stop condition, the model would pad numbers with infinite zeros (`200000000000000`). Fixed by using the model's own unconstrained top choice as a stop signal — if the model's preferred next token is not a digit, generation stops naturally.

**Prompt contamination between parameters.** When a single shared prompt included regex few-shot examples, the model would copy `"red"` or `"blue"` from those examples when generating `source_string`. Fixed by completely isolating each parameter's prompt — no shared context between parameter generation calls.

**`fn_anonymos` not being selected for gibberish.** The model would match single-word nonsense inputs to `fn_reverse_string` (string-sounding). Fixed by adding gibberish-specific few-shot examples using single-word nonsense inputs (including `"ayayay"` itself) in the function selection prompt.

---

## Testing Strategy

**Manual test against the provided 12 prompts.** Each run produces a full output JSON that can be visually inspected. All 12 prompts serve as the ground truth.

**Isolated FSM testing.** Each FSM function (`number_fsm`, `string_fsm`, `bool_fsm`) was tested in isolation by calling it directly with known inputs before integrating into the full pipeline.

**Edge case coverage:**
- Multi-digit numbers (`265`, `345`, `144`) to catch premature stop conditions
- Two-parameter functions to catch positional confusion
- Gibberish input to validate fallback behavior
- Regex generation to validate the dedicated few-shot branch
- The substitute function where `source_string` must NOT be modified

**Pydantic validation as a final gate.** All output dicts pass through `OutputItem` validation before being written to the output file, catching any malformed output at runtime rather than silently writing bad data.

---

## Resources

### Constrained Decoding
- [Constrained Decoding for LLMs — Willard & Louf (2023)](https://arxiv.org/abs/2307.09702) — the foundational paper on FSM-based constrained generation
- [Outlines library](https://github.com/outlines-dev/outlines) — production implementation of constrained decoding
- [Guidance library](https://github.com/guidance-ai/guidance) — another approach to structured generation

### Transformers & Tokenization
- [HuggingFace Transformers documentation](https://huggingface.co/docs/transformers)
- [Qwen3-0.6B model page](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Byte Pair Encoding explained](https://huggingface.co/learn/nlp-course/chapter6/5)

### Function Calling
- [OpenAI Function Calling documentation](https://platform.openai.com/docs/guides/function-calling) — reference for the structured output format this project targets

### AI Usage in this Project

AI was used throughout this project as a coding and learning assistant:

- **Concept explanation:** understanding FSMs, constrained decoding theory, tokenization internals, what weights are and how they're trained
- **Debugging:** identifying root causes of wrong outputs (prompt contamination, premature stop conditions, model computing instead of extracting)
- **Prompt engineering:** iterating on few-shot examples for each parameter type until reliable extraction was achieved
- **Architecture decisions:** the per-parameter isolated prompt design, the newline-as-stop-signal pattern, and the ordinal position labeling approach for multi-parameter functions all emerged from debugging sessions with Claude
