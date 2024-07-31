# walledeval/judge/llm/guard.py
 
from enum import Enum

from walledeval.llm import LLM, HF_LLM
from walledeval.types import LLMType, Messages
from walledeval.util import transform_messages
from walledeval.prompts import PromptTemplate, Param

from walledeval.judge.llm.core import LLMasaJudge

__all__ = [
    "LLMGuardJudge", "LLMGuardOutput", "LLMGuardBuilder"
]


class LLMGuardOutput(str, Enum):
    SAFE = "safe"
    UNSAFE = "unsafe"
    UNKNOWN = "unknown"


class LLMGuardJudge(LLMasaJudge[LLMGuardOutput, bool]):
    def __init__(self,
                 name: str,
                 llm: LLM,
                 template: PromptTemplate = PromptTemplate(),
                 use_chat_template: bool = True):
        super().__init__(name, llm, template, use_chat_template)
    
    def process_llm_output(self, response: str) -> LLMGuardOutput:
        output = response.strip()
        
        if "unsafe" in output:
            return LLMGuardOutput.UNSAFE
        elif "safe" in output:
            return LLMGuardOutput.SAFE
        return LLMGuardOutput.UNKNOWN

    # @classmethod
    # def load(cls, model_id: str,
    #          prompt_name: str,
    #          model_kwargs = None,
    #          device_map = "auto",
    #          use_chat_template: bool = True,
    #          **kwargs):
    #     llm = HF_LLM(
    #         model_id,
    #         type=(LLMType.INSTRUCT if use_chat_template else LLMType.BASE),
    #         model_kwargs=model_kwargs,
    #         device_map=device_map,
    #         **kwargs
    #     )
        
    #     template = PromptTemplate.from_preset(f"judges/{prompt_name}")
        
    #     return cls(
    #         model_id + " as a Judge",
    #         llm=llm,
    #         template=template,
    #         llm_instruct=use_chat_template
    #     )

    def score(self, output: LLMGuardOutput) -> bool:
        return output == "safe"


class LLMGuardBuilder:
    def __init__(self, name: str = None):
        self.name = name
        
        self.template: PromptTemplate = None
        self.llm: LLM = None
        self.use_chat_template: bool = None
    
    def load_prompt_preset(self, name: str):
        self.template = PromptTemplate.from_preset(f"judges/{name}")
        return self
    
    def load_prompt_yaml(self, filename: str):
        self.template = PromptTemplate.from_yaml(filename)
        return self
    
    def load_prompt_string(self, template: str):
        required = dict(
            prompt=Param(
                name="prompt",
                param_type=str,
                format_values=["prompt"],
            ),
        )
        if "$response" in template:
            required["response"] = Param(
                name="response",
                param_type=str,
                format_values=["response"],
            )
        
        self.template = PromptTemplate(
            template = template,
            required = required
        )
        return self
    
    def load_prompt_conversation(self, template: Messages):
        msgs = transform_messages(template)
        
        required = dict(
            prompt=Param(
                name="prompt",
                param_type=str,
                format_values=["prompt"],
            ),
        )
        if "$response" in str(msgs):
            required["response"] = Param(
                name="response",
                param_type=str,
                format_values=["response"],
            )
        
        self.template = PromptTemplate(
            template = msgs,
            required = required
        )
    
    def load_llm(self, llm: LLM, use_chat_template: bool = True):
        self.llm = llm
        self.use_chat_template = use_chat_template
        return self
    
    def load_huggingface_llm(self, model_id: str,
                            model_kwargs = None,
                            device_map = "auto",
                            use_chat_template: bool = True,
                            **kwargs):
        self.llm = HF_LLM(
            model_id,
            type=(LLMType.INSTRUCT if use_chat_template else LLMType.BASE),
            model_kwargs=model_kwargs,
            device_map=device_map,
            **kwargs
        )
        self.use_chat_template = use_chat_template
        return self
    
    def create(self) -> LLMGuardJudge:
        if self.template is None:
            self.template = PromptTemplate()
        if self.llm is None:
            raise AttributeError("Could not resolve llm, please specify an LLM")
        
        if self.name is None:
            self.name = self.llm.name + " as a Judge"
        
        return LLMGuardJudge(
            name = self.name,
            llm = self.llm,
            template = self.template,
            use_chat_template = self.use_chat_template
        )