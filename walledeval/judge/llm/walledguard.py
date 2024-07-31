# walledeval/judge/llm/walledguard.py

from walledeval.llm import HF_LLM
from walledeval.types import LLMType
from walledeval.prompts import PromptTemplate

from walledeval.judge.llm.guard import LLMGuardJudge

__all__ = ["WalledGuard"]


class WalledGuardJudge(LLMGuardJudge):
    def __init__(self, 
                 name: str = "walledai/walledguard-c",
                 model_kwargs=None, 
                 device_map="auto",
                 use_chat_template: bool = True,
                 template_preset: str = "walledguard",
                 **kwargs):
        llm = HF_LLM(
            name,
            type=(LLMType.INSTRUCT if use_chat_template else LLMType.BASE),
            model_kwargs=model_kwargs,
            device_map=device_map,
            **kwargs
        )
        
        template = PromptTemplate.from_preset(f"judges/{template_preset}")
        
        super().__init__(
            name, llm=llm,
            template=template,
            use_chat_template=use_chat_template
        )
    
    def generate(self, response: str, **kwargs) -> str:
        return super().generate(response,
                                max_new_tokens = 20,
                                **kwargs)