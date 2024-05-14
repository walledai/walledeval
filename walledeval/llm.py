# llmtest/llm/__init__.py

from abc import ABC, abstractmethod
from anthropic import Anthropic
from transformers import pipeline
from huggingface_hub import list_models

__all__ = [
    "hf_models",
    "LLM",
    "HF_LLM",
    "Claude"
]

def hf_models() -> list[str]:
    return list_models(filter="text-generation")

class LLM(ABC):
    def __init__(self, name: str, system_prompt: str = ""):
        self.name = name
        self.system_prompt = system_prompt
    
    @abstractmethod
    def generate(self, text: str) -> str:
        return ""
    
    def __call__(self, text: str) -> str:
        return self.generate(text)

class HF_LLM(LLM):
    def __init__(self, id: str, system_prompt: str = "", **kwargs):
        super().__init__(id, system_prompt)
        self.pipeline = pipeline(
            "text-generation", id, 
            trust_remote_code=True,
            **kwargs
        )
    
    def generate(self, text: str) -> str:
        text = self.pipeline([
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": text},
        ], max_new_tokens = 128)[0]['generated_text'][-1]
        return text
    
class Claude(LLM):
    def __init__(self, api_key: str, system_prompt: str = ""):
        super().__init__("Claude 3 Opus", system_prompt)
        self.client = Anthropic(api_key=api_key)
    
    def generate(self, text: str, 
                 max_tokens: int = 1024,
                 temperature: float = 0) -> str:
        message = self.client.messages.create(
            max_tokens=max_tokens,
            messages=[{
                "role": "user",
                "content": text
            }],
            temperature=temperature,
            system=self.system_prompt,
            model="claude-3-opus-20240229",
        )
        output = message.content[0].text
        return output