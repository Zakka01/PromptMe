import json
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

    parser.parse_functions_definition(functions_def_location)
    parser.parse_prompt(input_location)

    functions: dict = parser.functions
    prompts: list[str] = parser.prompts

    decoder = ConstrainedDecoder(functions, prompts, llm)
    for prompt in prompts:
        decoder.constrained_decoding(prompt)


if __name__ == "__main__":
    # try:
    main()
    # except Exception as e:
    #     print(f"Error: {e}")