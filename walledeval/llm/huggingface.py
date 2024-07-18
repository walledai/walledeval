from transformers import pipeline, TextGenerationPipeline
from huggingface_hub import list_models
from typing import Optional, Union

from walledeval.types import Messages, LLMType
from walledeval.util import transform_messages
from walledeval.llm.core import LLM
from walledeval.attacks.gcg import gcg_attack

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
        # Initialize gcg mutator
        self.mutator = gcg_attack

        self.pipeline.tokenizer.padding_side = 'left'

    def _generate(self,
                  prompt: str,
                  max_new_tokens: int = 256,
                  temperature: float = 0.1) -> str:
        # Use the mutator to alter prompt based on gradients
        mutated_prompt = self.mutator(self.pipeline.model, self.pipeline.tokenizer, prompt, "desired_target")
        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("")
        ]

        outputs = self.pipeline(
            mutated_prompt,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            return_full_text=False
        )[0]["generated_text"].strip()

        return outputs

    def chat(self,
             text: Messages,
             max_new_tokens: int = 256,
             temperature: float = 0.1) -> str:
        messages = transform_messages(text.messages, self.system_prompt)
        prompt = self.pipeline.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
        )

        return self._generate(prompt, max_new_tokens, temperature)

    def complete(self,
                 text: str,
                 max_new_tokens: int = 256,
                 temperature: float = 0.1) -> str:
        return self._generate(
            text,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )

    def generate(self,
                 text: Union[str, Messages],
                 max_new_tokens: int = 256,
                 temperature: float = 0.1,
                 instruct: Optional[bool] = None) -> str:
        if isinstance(text, str):
            return self.complete(text, max_new_tokens, temperature)
        else:
            return self.chat(text, max_new_tokens, temperature)
