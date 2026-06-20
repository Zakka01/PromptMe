from typing import List
import numpy as np
from llm_sdk.llm_sdk import Small_LLM_Model
from src.data_validation import FunctionValidator


class ConstrainedDecoder:

    def __init__(self, functions: dict, prompts: List, llm: Small_LLM_Model) -> None:
        self.functions = functions
        self.prompts = prompts 
        self.llm = llm

    def format_full_definition(self, fn: FunctionValidator) -> str:
        params_str = ", ".join(
            [f"{k}: {v.type}" for k, v in fn.parameters.items()]
        )
        return (
            f"{fn.name}({params_str}) -> "
            f"{fn.returns.type}: {fn.description}"
        )

    def predict_top_tokens(
            self,
            prompt_message: str,
            previous_tokens: str = "",
            top_k: int = 10,
        ) -> List[str]:
        prompt = (
            f"<|im_start|>user\n{prompt_message}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n{previous_tokens}"
        )
    
        input_ids = self.llm.encode(prompt)[0].tolist()
        logits = self.llm.get_logits_from_input_ids(input_ids)
        logits = np.asarray(logits)
    
        sorted_token_ids = np.argsort(-logits)
    
        top_ids = sorted_token_ids[:top_k]
        return [self.llm.decode([int(tid)]) for tid in top_ids]
    
    
    def number_fsm(self, function_name: str, prompt: str, pname: str, already_gen):
    
        function_def = self.functions[function_name]
    
        prompt = (
            f"To solve the prompt {prompt}, you will use "
            f"the following function: "
            f"{self.format_full_definition(function_def)}. "
            "Provide each parameter. Keep it concise and "
            "don't add custom fields."
        )
    
        argument_progress = ""
        max_steps = 80  # safety cap
        top_k = 50
    
        allowed_chars = set("-0123456789.\n")
    
        for _ in range(max_steps):
            candidates = self.predict_top_tokens(
                prompt_message=prompt,
                previous_tokens=already_gen + argument_progress,
                top_k=top_k,
            )
    
            picked: str | None = None
    
            for gen_raw in candidates:
                gen_raw = gen_raw.replace("Ġ", " ")
                gen = gen_raw.replace("Ċ", "\n")
                if gen_raw.strip() == "":
                    continue
    
                if any(ch not in allowed_chars for ch in gen):
                    continue
    
                candidate = argument_progress + gen
    
                if candidate.count(".") >= 2:
                    continue
                if candidate.count("-") >= 2:
                    continue
                if candidate.count("-") == 1 and not candidate.startswith("-"):
                    continue

                picked = gen
                break
    
            if picked is None or picked == "":
                break
    
            argument_progress += picked
    
            # stop once we hit newline (end-of-field)
            if "\n" in argument_progress:
                head = argument_progress.split("\n")[0]
                if head != "":              # FIX 2: ignore leading empty newline
                    argument_progress = head
                    break
                argument_progress = ""      # reset and keep looping
                continue
    
        try:
            if "." in argument_progress:    # FIX 3: preserve int vs float
                return float(argument_progress)
            return int(argument_progress)
        except Exception:
            return None                     # FIX 3: don't fake a 0.0 success

    def get_params_fsm(self, function_name: str, prompt: str) -> dict:

        params: dict = {}
        function_def = self.functions[function_name]
        func_params = function_def.parameters

        for pname, pspec in func_params.items():

            param_type = pspec.type
            already_gen = ""
            for generated_name, generated_val in params.items():
                already_gen += f"{generated_name}={generated_val}\n"
            already_gen += f"{pname}="

            value: object = None

            if param_type == "string":
                ...

            elif param_type in ("number", "integer"):
                value = self.number_fsm(function_name, prompt, pname, already_gen)

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