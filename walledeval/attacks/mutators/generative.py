# walledeval/attacks/mutators/generative.py

from walledeval.llm import LLM
from walledeval.prompts import PromptTemplate
from walledeval.attacks.mutators.core import Mutator


class GenerativeMutator(Mutator):
    def __init__(self, name: str, 
                 llm: LLM,
                 prompt_template: PromptTemplate = PromptTemplate()):
        super().__init__(name)
        
        self.llm = llm
        self.prompt_template = prompt_template
        
        if "prompt" not in prompt_template.params:
            raise ValueError("Cannot use this prompt template, no parameter 'prompt' found")
    
    @classmethod
    def from_preset(cls, name: str, llm: LLM):
        template = PromptTemplate.from_preset(f"mutations/{name}")
        return cls(name, llm, template)
        
    
    def mutate(self, prompt: str, **kwargs) -> str:
        query = self.prompt_template.format(None, prompt = prompt, **kwargs)
        output = self.llm.generate(query, instruct=True)
        return output

