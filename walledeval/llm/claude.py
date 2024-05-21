# walledeval/llm/claude.py

from anthropic import Anthropic

from walledeval.llm.core import LLM

__all__ = [
    "Claude"
]


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