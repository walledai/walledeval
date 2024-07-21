# walledeval/attacks/mutators/cipher.py

from abc import abstractmethod
from collections.abc import Iterable

import inflect
import base64

from walledeval.prompts import PromptTemplate
from walledeval.attacks.mutators.core import Mutator

p = inflect.engine()


class CipherMutator(Mutator):
    def __init__(self, name: str, template: PromptTemplate, samples: Iterable[str] = []):
        super().__init__(name)
        self.template = template
        self.samples = list(samples)
    
    @abstractmethod
    def encode(self, x: str) -> str:
        raise NotImplementedError
    
    @abstractmethod
    def decode(self, y: str) -> str:
        raise NotImplementedError
    
    def mutate(self, prompt: str, **kwargs) -> str:
        examples = ""
        for i, sample in enumerate(self.samples):
            examples += f"Example {i}:\n{self.encode(sample)}\n\n"
        
        return self.template.format(
            prompt = self.encode(prompt),
            examples = examples.strip(),
            **kwargs
        )


class CaesarMutator(CipherMutator):
    def __init__(self, shift: int, samples = Iterable[str] = []):
        super().__init__(
            "CaesarMutator",
            template = PromptTemplate.from_preset("mutations/cipher/caesar"),
            samples = samples
        )
        self.shift = shift % 26
    
    def mutate(self, prompt: str, **kwargs) -> str:
        return super().mutate(prompt,
                              shift = p.number_to_words(self.shift),
                              **kwargs)
    
    def shift(self, char: str, shift: int) -> str:
        if 'a' <= char <= 'z':
                ans += chr(ord('a') + (ord(char) - ord('a') + shift) % 26)
        elif 'A' <= char <= 'Z':
            ans += chr(ord('A') + (ord(char) - ord('A') + shift) % 26)
        else:
            ans += p
    
    def encode(self, x: str) -> str:
        return "".join([self.shift(char, self.shift) for char in x])
    
    def decode(self, y: str) -> str:
        return "".join([self.shift(char, -self.shift) for char in y])


class Base64Mutator(CipherMutator):
    def __init__(self, samples = Iterable[str] = []):
        super().__init__(
            "Base64Mutator",
            template = PromptTemplate.from_preset("mutations/cipher/base64"),
            samples = samples
        )
    
    def encode(self, x: str) -> str:
        return base64.b64encode(x.encode()).decode()
    
    def decode(self, y: str) -> str:
        return base64.b64decode(y.encode()).decode()


class AsciiMutator(CipherMutator):
    def __init__(self, samples = Iterable[str] = []):
        super().__init__(
            "AsciiMutator",
            template = PromptTemplate.from_preset("mutations/cipher/ascii"),
            samples = samples
        )
    
    def encode(self, x: str) -> str:
        return "".join([x if x == "\n" else str(ord(x)) for i in x])
    
    def decode(self, y: str) -> str:
        return "\n".join(["".join([chr(int(yij)) for yij in yi.split()]) for yi in y.split("\n")])