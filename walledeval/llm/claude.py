# walledeval/llm/claude.py

from anthropic import Anthropic

from walledeval.llm.core import LLM

__all__ = [
    "Claude"
]


class Claude(LLM):
    def __init__(self, model_id: str, api_key: str, system_prompt: str = ""):
        super().__init__(model_id, system_prompt)
        self.client = Anthropic(api_key=api_key)

    @classmethod
    def opus(cls, api_key: str, system_prompt: str = ""):
        return cls(
            "claude-3-opus-20240229",
            api_key, system_prompt
        )

    @classmethod
    def sonnet(cls, api_key: str, system_prompt: str = ""):
        return cls(
            "claude-3-sonnet-20240229",
            api_key, system_prompt
        )

    @classmethod
    def haiku(cls, api_key: str, system_prompt: str = ""):
        return cls(
            "claude-3-haiku-20240307",
            api_key, system_prompt
        )

    def generate(self, text: str,
                 max_new_tokens: int = 1024,
                 temperature: float = 0) -> str:
        message = self.client.messages.create(
            max_tokens=max_new_tokens,
            messages=[{
                "role": "user",
                "content": text
            }],
            temperature=temperature,
            system=self.system_prompt,
            model=self.name
        )
        output = message.content[0].text
        return output
