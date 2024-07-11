# walledeval/llm/huggingface.py

from transformers import pipeline, TextGenerationPipeline
from huggingface_hub import list_models

from typing import Optional, Union

from walledeval.dtypes import Message, Messages, LLMType
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
    def __init__(self,
                 id: str,
                 system_prompt: str = "",
                 type: Optional[Union[LLMType, int]] = LLMType.NEITHER,
                 **kwargs):
        super().__init__(id, system_prompt, type)
        self.pipeline: TextGenerationPipeline = pipeline(
            "text-generation",
            model=id,
            trust_remote_code=True,
            **kwargs
        )

    def _generate(self,
                  prompt: str,
                  max_new_tokens: int = 256,
                  temperature: float = 0.6) -> str:
        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            return_full_text=False
        )[0]["generated_text"].strip()
        # [len(prompt):]

        return outputs

    def chat(self,
             text: Messages,
             max_new_tokens: int = 256,
             temperature: float = 0.0) -> str:
        if isinstance(text, str):
            messages = [{
                "role": "user",
                "content": text
            }]
        elif isinstance(text, list) and isinstance(text[0], Message):
            messages = [
                dict(msg)
                for msg in text
            ]
        elif isinstance(text, list) and isinstance(text[0], dict):
            messages = text
        else:
            raise TypeError("Unsupported format for parameter 'text'")

        if (
            len(self.system_prompt.strip()) > 0 and
            messages[0]["role"] != "system"
        ):
            messages.insert(0, {
                "role": "system",
                "content": self.system_prompt
            })

        prompt = self.pipeline.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
        )

        return self._generate(prompt, max_new_tokens, temperature)

    def complete(self,
                 text: str,
                 max_new_tokens: int = 256,
                 temperature: float = 0.0) -> str:
        return self._generate(
            text,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )

    def generate(self,
                 text: Messages,
                 max_new_tokens: int = 256,
                 temperature: float = 0.0,
                 instruct: Optional[bool] = None) -> str:
        type = None
        if instruct is None:
            if self.type == LLMType.BASE:
                type = LLMType.BASE
            else:
                type = LLMType.INSTRUCT
        elif instruct:
            type = LLMType.INSTRUCT
        else:
            type = LLMType.BASE

        if type == LLMType.INSTRUCT:
            return self.chat(
                text,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
        else:
            if not isinstance(text, str):
                raise ValueError("Unsupported type for input 'text'")
            return self.complete(
                text,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
