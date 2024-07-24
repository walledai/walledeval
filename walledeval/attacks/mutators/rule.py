# walledeval/attacks/mutators/rule.py

import re

from walledeval.attacks.mutators.core import Mutator

__all__ = [
    "DisemvowelMutator", "LeetSpeakMutator"
]


class DisemvowelMutator(Mutator):
    def __init__(self):
        super().__init__("disemvowel")

    def mutate(self, prompt: str, **kwargs) -> str:
        return re.sub("[aeiouAEIOU]+", "", prompt)

class LeetSpeakMutator(Mutator):
    def __init__(self):
        super().__init__("leetspeak")
    
    def mutate(self, prompt: str, **kwargs) -> str:
        leet_dict = {
            'a': '@', 'e': '3',
            'i': '!', 'o': '0',
            'are': 'r', 'be': 'b'
        }
        
        for key, val in leet_dict.items():
            prompt = prompt.replace(key, val)
        
        return prompt