import sys
import json
from typing import List, Any
from .data_validation import FunctionValidator, PromptValidator, ValidationError, ParameterSpec


class Parser:
    def __init__(self):
        self.functions = {}
        self.prompts = []

    def parse_args(self):
        args = {
            "functions_definition": "data/input/functions_definition.json",
            "input": "data/input/function_calling_tests.json",
            "output": "data/output/function_calls.json",
        }

        argv = sys.argv[1:]

        i = 0
        while i < len(argv):
            if argv[i] == "--functions_definition":
                args["functions_definition"] = argv[i + 1]
                i += 2

            elif argv[i] == "--input":
                args["input"] = argv[i + 1]
                i += 2

            elif argv[i] == "--output":
                args["output"] = argv[i + 1]
                i += 2

            else:
                raise ValueError(f"Unknown argument: {argv[i]}")
    
        return args

    def parse_functions_definition(self, file_path: str) -> None:
        try:
            with open(file_path, "r") as f:
                content = json.load(f)

            validated = []

            for item in content:
                validated.append(FunctionValidator(**item))

            if not any(fn.name == "fn_anonymos" for fn in validated):
                validated.append(
                    FunctionValidator(
                        name="fn_anonymos",
                        description=(
                            "Fallback function "
                            "when no available function fits the prompt. "
                        ),
                        parameters={"name": ParameterSpec(type="string")},
                        returns=ParameterSpec(type="string"),
                    )
                )

            for v in validated:
                self.functions[v.name] = v

        except ValidationError as e:
            print(f"Error: {e.errors()[0]['msg']}")
            exit(1)

    def parse_prompt(self, file_path: str) -> None:
        try:
            with open(file_path, "r") as f:
                content = json.load(f)

            validated = []

            for item in content:
                validated.append(PromptValidator(**item))

            for v in validated:
                self.prompts.append(v.prompt)

        except ValidationError as e:
            print(f"Error: {e.errors()[0]["msg"]}")
            exit(1)
