import json
import os
from llm_sdk.llm_sdk import Small_LLM_Model
from .parser import Parser
from .constrained_decoding import ConstrainedDecoder
from .data_validation import FunctionValidator, ParameterSpec


def main() -> None:
    parser = Parser()
    llm = Small_LLM_Model()

    args = parser.parse_args()
    input_location = args["input"]
    functions_def_location = args["functions_definition"]
    output_location = args["output"]

    parser.parse_functions_definition(functions_def_location)
    parser.parse_prompt(input_location)

    functions: dict = parser.functions
    prompts: list[str] = parser.prompts

    decoder = ConstrainedDecoder(functions, prompts, llm)
    for prompt in prompts:
        decoder.constrained_decoding(prompt)

    res = decoder.output
    os.makedirs(os.path.dirname(output_location), exist_ok=True)

    try:
        with open(output_location, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
            print(f"Results saved to {output_location}")
    except OSError as e:
        print(f"Failed to write results to {output_location}: {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")