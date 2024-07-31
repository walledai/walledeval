# walledeval/attacks/wildteaming.py

import random

from walledeval.llm import LLM
from walledeval.attacks.mutators import GenerativeMutator
from walledeval.attacks.core import PromptAttack


class WildTeamingAttack(PromptAttack):
    _TACTICS = [
        "autodan/revise", "masterkey/rephrase",
        "renellm/alter-sentence-structure",
        "renellm/change-style",
        "renellm/insert-meaningless-characters",
        "renellm/misspell-sensitive-words",
        "renellm/paraphrase-fewer-words",
        "renellm/translation",
        "future-tense", "past-tense"
    ]
    
    def __init__(self, llm: LLM,
                 num_tactics: int = 10, 
                 seed: int = None):
        super().__init__("WildTeamingAttack")
        
        random.seed(seed)
        tactic_names = random.choices(self._TACTICS, k=num_tactics)
        
        self.mutators = [
            GenerativeMutator.from_preset(name, llm)
            for name in tactic_names
        ]
        
        self.llm = llm
    
    
    def single_attack(self, sample: str, **kwargs) -> list[str]:
        samples = []
        for mutator in self.mutators:
            samples.append(mutator.mutate(sample, **kwargs))
        
        # TODO: Implement Pruning Methods for Samples
        
        return samples
