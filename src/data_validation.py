from pydantic import BaseModel, model_validator, field_validator, ValidationError
from typing import List, Dict


class ParameterSpec(BaseModel):
    type: str

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        allowed = {"integer", "string", "boolean", "number"}
        if v not in allowed:
            raise ValueError("Invalid type")
        return v



class FunctionValidator(BaseModel):
    name: str
    description: str
    parameters: Dict[str, ParameterSpec]
    returns: ParameterSpec

    @field_validator("name", "description")
    @classmethod
    def validate_strings(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Field cannot be empty or only spaces")
        return value

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, value: dict) -> dict:
        if not value:
            raise ValueError("Parameters cannot be empty")

        for key in value.keys():
            if not key.strip():
                raise ValueError(
                    "Parameter names cannot be empty "
                    "or only spaces"
                )
        return value


class PromptValidator(BaseModel):
    prompt: str

    @field_validator("prompt")
    def validate_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Prompt cannot be empty or only spaces")
        return value