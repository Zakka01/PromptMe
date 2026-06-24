from typing import List
import numpy as np
import json
from llm_sdk.llm_sdk import Small_LLM_Model


class ConstrainedDecoder:

    def __init__(self, functions: dict, prompts: List, llm: Small_LLM_Model) -> None:
        self.functions = functions
        self.prompts = prompts 
        self.llm = llm
        self.output = []

    def number_fsm(self, function_name: str, prompt: str, pname: str, previously_gen: str):

        base_prompt = (
            f"Function: {function_name}\n"
            f"User request: {prompt}\n\n"
            "Extract function parameters.\n"
            "Do not solve the request.\n"
            "Do not calculate.\n\n"
            f"{previously_gen}"
        )

        input_ids = self.llm.encode(base_prompt)[0].tolist()
        generated = ""
        allowed = set("0123456789-.\n")
    
        for _ in range(20):
            logits = self.llm.get_logits_from_input_ids(input_ids)
            valid_tokens = []
            for tid, score in enumerate(logits):

                piece = self.llm.decode([tid])

                if piece == "" or any(ch not in allowed for ch in piece):
                    continue
                candidate = generated + piece
                if candidate.count(".") >= 2 or candidate.count("-") >= 2:
                    continue
                if "-" in candidate and not candidate.startswith("-"):
                    continue

                valid_tokens.append((tid, piece))
    
            if not valid_tokens:
                break
    
            best_tid, best_piece = max(valid_tokens, key=lambda t: logits[t[0]])
            input_ids.append(best_tid)
            generated += best_piece
    
            if "\n" in generated:
                generated = generated.split("\n")[0]
                break

        return float(generated.strip())

    def regex_function_prompt(self, prompt: str, pname: str) -> str:
        if pname == "regex":
            return (
                f"User request: {prompt}\n\n"
                "Generate the regex value for parameter 'regex'.\n"
                "  - for specific numbers in the text, join them with | (e.g. 34 and 233 -> 34|233)\n"
                "  - for vowels, use: .*[aeiouAEIOU]\n"
                "  - for a specific word, use just that word\n\n"
                "Request: Replace all numbers in \"I have 12 cats and 99 dogs\" with X\n"
                "regex: 12|99\n\n"
                "Request: Replace all vowels in 'hello world' with stars\n"
                "regex: .*[aeiouAEIOU]\n\n"
                "Request: Substitute the word 'red' with 'blue' in 'the red car is red'\n"
                "regex: red\n\n"
                f"Request: {prompt}\n"
                "regex:"
            )
        elif pname == "replacement":
            return (
                f"User request: {prompt}\n\n"
                "Extract the replacement value (the new word/text to insert).\n\n"
                "Request: Replace all vowels in 'hello world' with stars\n"
                "replacement: stars\n\n"
                "Request: Substitute the word 'red' with 'blue' in 'the red car is red'\n"
                "replacement: blue\n\n"
                "Request: Replace all numbers in \"I have 12 cats\" with X\n"
                "replacement: X\n\n"
                f"Request: {prompt}\n"
                "replacement:"
            )
        else:
            return (
                f"User request: {prompt}\n\n"
                "Extract the ORIGINAL source string, before any replacement happens. "
                "Do NOT apply the substitution yourself.\n\n"
                "Request: Substitute the word 'red' with 'blue' in 'the red car is red'\n"
                "WRONG: the blue car is blue\n"
                "source_string: the red car is red\n\n"
                "Request: Substitute the word 'cat' with 'dog' in 'my cat is a cat'\n"
                "WRONG: my dog is a dog\n"
                "source_string: my cat is a cat\n\n"
                f"Request: {prompt}\n"
                "source_string:"
            )

    def string_fsm(self, function_name: str, prompt: str, pname: str, previously_gen: str):
        
        if function_name == "fn_substitute_string_with_regex":
            base_prompt = self.regex_function_prompt(prompt, pname)
        else:
            base_prompt = (
                f"Function: {function_name}\n"
                f"User request: {prompt}\n\n"
                f"Extract the value for parameter '{pname}'.\n"
                "Do not solve the request. Copy the relevant text exactly.\n\n"
                f"{pname}:"
            )

        input_ids = self.llm.encode(base_prompt)[0].tolist()
        generated = ""

        for _ in range(30):
            logits = self.llm.get_logits_from_input_ids(input_ids)

            valid_tokens = []
            for tid, score in enumerate(logits):
                piece = self.llm.decode([tid])

                if piece == "" or piece == "\n":
                    continue

                candidate = generated + piece
                if candidate.count("'") > 2 or candidate.count('"') > 2:
                    continue

                valid_tokens.append((tid, piece))

            if not valid_tokens:
                break

            best_tid, best_piece = max(valid_tokens, key=lambda t: logits[t[0]])
            input_ids.append(best_tid)
            generated += best_piece

            if generated.count("'") == 2 or "\n" in generated:
                if "'" in generated:
                    generated = generated.replace("'", "").strip()
                if '"' in generated:
                    generated = generated.replace('"', "").strip()
                elif "\n" in generated:
                    generated = generated.split("\n")[0]
                break

        return generated.strip()

    def get_params_fsm(self, function_name: str, prompt: str) -> dict:

        params: dict = {}
        function_def = self.functions[function_name]
        func_params = function_def.parameters

        for pname, pspec in func_params.items():

            previously_gen = ""
            for name, val in params.items():
                previously_gen += f"{name}={val}\n"
            previously_gen += f"{pname}="

            value = None
            param_type = pspec.type

            if param_type == "string":
                value = self.string_fsm(function_name, prompt, pname, previously_gen)

            elif param_type in ("number", "integer"):
                value = self.number_fsm(function_name, prompt, pname, previously_gen)

            elif param_type == "boolean":
                ...

            params[pname] = value
            
        return params

    def get_function_name(self, prompt: str) -> str:

        function_block = ""
        for value in self.functions.values():
            function_block += (f"{value.name}: {value.description}\n")

        base_prompt = (
            "You are a function selector.\n"
            "Your task is to choose exactly ONE function name from the available list.\n"
            "Return ONLY the function name exactly as written. No extra text.\n\n"
            "AVAILABLE FUNCTIONS:\n"
            f"{function_block}\n\n"
            "EXAMPLES:\n"
            "USER: What is the sum of 2 and 3?\n"
            "ANSWER: fn_add_numbers\n\n"
            "USER: Greet shrek\n"
            "ANSWER: fn_greet\n\n"
            "USER: blah blah nothing here\n"
            "ANSWER: fn_anonymos\n\n"
            "USER: lalala\n"
            "ANSWER: fn_anonymos\n\n"
            "USER: ayayay\n"
            "ANSWER: fn_anonymos\n\n"
            f"USER: {prompt}\n"
            "ANSWER:"
        )

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

    def constrained_decoding(self, prompt: str) -> List:
        function_name = self.get_function_name(prompt)

        if function_name == "fn_anonymos":
            self.output.append({
                "prompt": prompt,
                "name": function_name,
                "parameters": None
            })
        else:
            self.output.append({
                "prompt": prompt,
                "name": function_name,
                "parameters": self.get_params_fsm(function_name, prompt)
            })
