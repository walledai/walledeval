# walledeval/judge/llm/wildguard.py

from walledeval.llm import HF_LLM
from walledeval.types import LLMType
from walledeval.prompts import PromptTemplate

from walledeval.judge.llm.guard import LLMGuardJudge, LLMGuardOutput

__all__ = ["WalledGuard"]


class WildGuardJudge(LLMGuardJudge):
    def __init__(self, 
                 name: str = "allenai/wildguard",
                 model_kwargs=None, 
                 device_map="auto",
                 use_chat_template: bool = True,
                 template_preset: str = "wildguard",
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
        prompt = self._template.format(
            response = response,
            prompt = "" if "response" in self._template.params else response,
            **kwargs
        )
        
        return self._llm.generate(
            prompt,
            instruct=self.llm_instruct,
            **kwargs
        )