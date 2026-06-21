from typing import List
import numpy as np
import re
from llm_sdk.llm_sdk import Small_LLM_Model


class ConstrainedDecoder:

    def __init__(self, functions: dict, prompts: List, llm: Small_LLM_Model) -> None:
        self.functions = functions
        self.prompts = prompts 
        self.llm = llm

    def is_valid_int_prefix(self, s: str, prompt: str):
        s = s.strip()
        if s == "" or s == "-":
            return True
        if s.startswith("-"):
            s = s[1:]
        if not s.isdigit():
            return False
        prompt_numbers = re.findall(r'\d+', prompt)
        return any(num.startswith(s) for num in prompt_numbers)

    def is_int_done(self, s: str, prompt: str) -> bool:
        s = s.strip()
        nums = re.findall(r'\d+', prompt)
        if s not in nums:
            return False
        # only done if no longer number in the prompt also starts with s
        return not any(len(n) > len(s) and n.startswith(s) for n in nums)

    def number_fsm(self, function_name: str, prompt: str, pname: str):

        base_prompt = (
            f"You are extracting one number that LITERALLY appears as text in the REQUEST.\n"
            f"Do NOT calculate, solve, or do math. The VALUE must be copy-pasted from the "
            f"REQUEST text, character for character. If the VALUE you're about to give does "
            f"not appear in the REQUEST text, you are WRONG.\n\n"
            "EXAMPLES:\n"
            "REQUEST: What is the sum of 7 and 9?\n"
            "PARAMETER: a (first number mentioned)\n"
            "VALUE: 7\n\n"
            "REQUEST: What is the sum of 7 and 9?\n"
            "PARAMETER: b (second number mentioned)\n"
            "VALUE: 9\n\n"
            "REQUEST: What is the square root of 25?\n"
            "PARAMETER: a\n"
            "VALUE: 25\n\n"
            "REQUEST: What is the square root of 81?\n"
            "PARAMETER: a\n"
            "VALUE: 81\n\n"
            "REQUEST: Calculate the square root of 64\n"
            "PARAMETER: a\n"
            "VALUE: 64\n\n"
            "REQUEST: Calculate the square root of 100\n"
            "PARAMETER: a\n"
            "VALUE: 100\n\n"
            f"REQUEST: {prompt}\n"
            f"PARAMETER: {pname}\n"
            "VALUE:"
        )

        input_ids = self.llm.encode(base_prompt)[0].tolist()
        generated = []

        for _ in range(15):
            logits = self.llm.get_logits_from_input_ids(input_ids)

            valid_tokens = []
            for tid, score in enumerate(logits):
                candidates = self.llm.decode(generated + [tid]).strip()
                if self.is_valid_int_prefix(candidates, prompt):
                    valid_tokens.append(tid)

            if not valid_tokens:
                break

            best_token_id = max(valid_tokens, key=lambda t: logits[t])
            input_ids.append(best_token_id)
            generated.append(best_token_id)

            text = self.llm.decode(generated).strip()
            if self.is_int_done(text, prompt):
                    break

        number = self.llm.decode(generated).strip()

        return number

    def get_params_fsm(self, function_name: str, prompt: str) -> dict:

        params: dict = {}
        function_def = self.functions[function_name]
        func_params = function_def.parameters
        print(f"\n{prompt}")

        for pname, pspec in func_params.items():
            value = None
            param_type = pspec.type

            if param_type == "string":
                ...

            elif param_type in ("number", "integer"):
                value = self.number_fsm(function_name, prompt, pname)

            elif param_type == "boolean":
                ...

            params[pname] = value
            
        print(params)
        return params

    def get_function_name(self, prompt: str, base_prompt: str) -> str:

        input_ids = self.llm.encode(base_prompt)[0].tolist()
        generated = []

        valid_names = set(self.functions.keys())

        for a in range(10):

            logits = self.llm.get_logits_from_input_ids(input_ids)

            valid_token_ids = []
            for token_id, score in enumerate(logits):
                candidate = self.llm.decode(generated + [token_id]).strip()

                if any(fn.startswith(candidate) for fn in valid_names):
                    valid_token_ids.append(token_id)

            if not valid_token_ids:
                break

            best_token_id = max(valid_token_ids, key=lambda t: logits[t])
            input_ids.append(best_token_id)
            generated.append(best_token_id)

            name = self.llm.decode(generated).strip()

            if name in valid_names:
                return name

    def constrained_decoding(self, prompt: str) -> dict:
        output_dict: dict = {}
        output_dict["prompt"] = prompt

        function_block = ""
        for value in self.functions.values():
            function_block += (f"{value.name}: {value.description}\n")

        base_prompt = (
             "You are a function selector.\n"
             "Your task is to choose exactly ONE function name from the available list.\n"
             "Return ONLY the function name exactly as written. No extra text.\n"
     
                 "RULES:\n"
                 "- Do NOT guess\n"
                 "- Do NOT pick a function if unsure\n"
                 "- if prompt is not clear use fn_anonymos"
                 "- Prefer fn_anonymos if not explicitly matching\n\n"

             "AVAILABLE FUNCTIONS:\n"
             f"{function_block}\n\n"

             "EXAMPLES:\n"
             "USER: What is the sum of 2 and 3?\n"
             "ANSWER: fn_add_numbers\n\n"
             
             "USER: Greet shrek\n"
             "ANSWER: fn_greet\n\n"
             
             "USER: Unknown task is unknown\n"
             "ANSWER: fn_anonymos\n\n"
             
             f"USER: {prompt}\n"
             "ANSWER:"
        )

        function_name = self.get_function_name(prompt, base_prompt)
        output_dict["name"] = function_name

        output_dict["parameters"] = self.get_params_fsm(function_name, prompt)