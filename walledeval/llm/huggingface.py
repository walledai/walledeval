# walledeval/llm/__init__.py

from transformers import pipeline

from huggingface_hub import list_models
# from huggingface_hub.hf_api import ModelInfo

# from typing import Generator

from walledeval.llm.core import LLM

__all__ = [
    "hf_models",
    "HF_LLM",
]


def hf_models():
    """List all LLM models supported by HuggingFace for Text Generation.

    Returns:
        Generator[ModelInfo]: models supporting text generation on
        HuggingFace Hub.
    """
    return list_models(filter="text-generation")


class HF_LLM(LLM):
    def __init__(self, id: str, system_prompt: str = "", **kwargs):
        super().__init__(id, system_prompt)
        self.pipeline = pipeline(
            "text-generation",
            model=id,
            trust_remote_code=True,
            **kwargs
        )

    def generate(self, text: str, max_new_tokens: int = 256) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": text},
        ]

        prompt = self.pipeline.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
        )

        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )[0]["generated_text"][len(prompt):].strip()

        return outputs
