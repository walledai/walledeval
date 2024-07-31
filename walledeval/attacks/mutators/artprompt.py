# walledeval/attacks/mutators/artprompt.py

from collections.abc import Iterable
from walledeval.attacks.mutators.core import Mutator
from walledeval.prompts import PromptTemplate
from art import text2art

class MaskingMutator(Mutator):
    def __init__(self, samples: Iterable[str] = []):
        super().__init__("MaskingMutator")
        self.template = PromptTemplate.from_preset("mutations/artprompt/mask")
        self.samples = list(samples)
    
    def encode(self, text: str, words_to_mask: Iterable[str]) -> str:
        for word in words_to_mask:
            text = text.replace(word, "[MASK]")
        return text
    
    def mutate(self, prompt: str, **kwargs) -> str:
        words_to_mask = kwargs.get('words_to_mask', [])
        examples = ""
        for i, sample in enumerate(self.samples):
            examples += f"Example {i}:\n{self.encode(sample, words_to_mask)}\n\n"
        
        return self.template.format(
            prompt=self.encode(prompt, words_to_mask),
            examples=examples.strip(),
            **kwargs
        )

class CloakingMutator(Mutator):
    SEP_DICT = {
        'cards': '*',
        'keyboard': '*',
        'puzzle': '*',
        'block': '*',
        'hollywood': '*',
        'ghoulish': '*'
    }

    def __init__(self, font: str = 'standard', samples: Iterable[str] = []):
        super().__init__("CloakingMutator")
        self.font = font
        self.template = PromptTemplate.from_preset("mutations/artprompt/cloak")
        self.samples = list(samples)
    
    def generate_ascii_art(self, word: str, ascii_coding: str) -> str:
        ascii_art = text2art(word, font=ascii_coding).strip().split('\n')
        sep = self.SEP_DICT[ascii_coding]
        res = [sep.join(line) for line in ascii_art]
        return '\n'.join(res)
    
    def encode(self, text: str, words_to_cloak: Iterable[str]) -> str:
        ascii_art_parts = []
        for word in words_to_cloak:
            ascii_art = self.generate_ascii_art(word, self.font)
            ascii_art_parts.append(f"ASCII art for '{word}':\n{ascii_art}")
        ascii_art_text = "\n\n".join(ascii_art_parts)
        return ascii_art_text
    
    def mutate(self, prompt: str, **kwargs) -> str:
        words_to_cloak = kwargs.get('words_to_cloak', [])
        examples = ""
        for i, sample in enumerate(self.samples):
            examples += f"Example {i}:\n{self.encode(sample, words_to_cloak)}\n\n"
        
        return self.template.format(
            prompt=self.encode(prompt, words_to_cloak),
            examples=examples.strip(),
            **kwargs
        )
