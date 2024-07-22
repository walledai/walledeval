# walledeval/attacks/mutators/ArtPrompt.py

from collections.abc import Iterable
from walledeval.attacks.mutators.core import Mutator
from art import text2art

class Masking(Mutator):
    def __init__(self, mask_char: str = '*'):
        super().__init__("Masking")
        self.mask_char = mask_char
    
    def encode(self, text: str, words_to_mask: Iterable[str]) -> str:
        for word in words_to_mask:
            text = text.replace(word, self.mask_char * len(word))
        return text
    
    def mutate(self, prompt: str, **kwargs) -> str:
        words_to_mask = kwargs.get('words_to_mask', [])
        return self.encode(prompt, words_to_mask)

class Cloaking(Mutator):
    def __init__(self, font: str = 'standard'):
        super().__init__("Cloaking")
        self.font = font
    
    def encode(self, text: str, words_to_cloak: Iterable[str]) -> str:
        for word in words_to_cloak:
            ascii_art = text2art(word, font=self.font).strip()
            text = text.replace(word, ascii_art)
        return text
    
    def mutate(self, prompt: str, **kwargs) -> str:
        words_to_cloak = kwargs.get('words_to_cloak', [])
        return self.encode(prompt, words_to_cloak)
