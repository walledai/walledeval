# walledeval/judge/llm/llamaguard.py

import torch
from typing import Union

from walledeval.llm import HF_LLM
from walledeval.types import LLMType
from walledeval.prompts import PromptTemplate

from walledeval.judge.llm.guard import LLMGuardJudge

__all__ = [
    "LlamaGuardJudge"
]


class LlamaGuardJudge(LLMGuardJudge):
    _VERSIONS = [
        "meta-llama/LlamaGuard-7b",
        "meta-llama/Meta-Llama-Guard-2-8B",
        "meta-llama/Llama-Guard-3-8B"
    ]   
    
    def __init__(self, version: Union[int, str], 
                 model_kwargs={"torch_dtype": torch.bfloat16}, 
                 device_map="auto",
                 template_preset: str = "llamaguard",
                 **kwargs):
        if isinstance(version, int):
            if version < 0 or version > len(self._VERSIONS):
                raise ValueError(f"Invalid Version {version}")
            
            self.model_id = self._VERSIONS[(version-1) % 3]
        elif isinstance(version, str):
            self.model_id = version

        self.version = version

        llm = HF_LLM(
            self.model_id,
            type=LLMType.INSTRUCT,
            model_kwargs=model_kwargs,
            device_map=device_map,
            **kwargs
        )
        
        super().__init__(self.model_id, llm, 
                         template=PromptTemplate.from_preset(f"judges/{template_preset}"), 
                         use_chat_template=True)
    