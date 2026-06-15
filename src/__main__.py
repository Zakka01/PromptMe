import json
from .llm_model import LlmModel
from .parser import Parser
import torch


def llm_pipeline(llm: LlmModel, functions: dict, prompts: list[str]) -> None:
    function_block = ""
    for value in functions.values():
        function_block += (f"{value.name}: {value.description}\n")

    for prompt in prompts:
        print(f"USER: {prompt}")

        base_prompt = (
            "Choose exactly ONE function name.\n"
            "Return ONLY the function name, nothing else.\n\n"
            + function_block +
            f"\nUSER: {prompt}\n"
        )

        input_ids = llm.encode(base_prompt)[0].tolist()
        generated = []

        valid_names = set(functions.keys())

        for a in range(10):

            logits = llm.get_logits_from_input_ids(input_ids)

            valid_token_ids = []
            for token_id, score in enumerate(logits):
                candidate = llm.decode(generated + [token_id])
            
                if any(fn.startswith(candidate) for fn in valid_names):
                    valid_token_ids.append(token_id)

            if not valid_token_ids:
                break


            for token_id in sorted(valid_token_ids, key=lambda t: logits[t], reverse=True)[:10]:
                print(repr(llm.decode([token_id])), logits[token_id])


            best_token_id = max(valid_token_ids, key=lambda t: logits[t])
            input_ids.append(best_token_id)
            generated.append(best_token_id)

            text = llm.decode(generated).strip()

            if text in valid_names:
                # print(f"ANSWER: {text}\n")
                break


def main() -> None:
    parser = Parser()
    llm = LlmModel()

    args = parser.parse_args()
    input_location = args["input"]
    functions_def_location = args["functions_definition"]

    functions: dict = parser.parse_functions_definition(functions_def_location)
    prompts: list[str] = parser.parse_prompt(input_location)

    llm_pipeline(llm, functions, prompts)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")