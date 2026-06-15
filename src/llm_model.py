from llm_sdk.llm_sdk import Small_LLM_Model


class LlmModel(Small_LLM_Model):

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B", *, device: str | None = None, dtype: torch.dtype | None = None, trust_remote_code: bool = True) -> None:
        super().__init__(model_name, device=device, dtype=dtype, trust_remote_code=trust_remote_code)

    def decode(self, ids: torch.Tensor | list[int]) -> str:
        return super().decode(ids)

    def encode(self, text: str) -> torch.Tensor:
        return super().encode(text)

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        return super().get_logits_from_input_ids(input_ids)
