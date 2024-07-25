# walledeval/attacks/mutators/translation.py

from typing import Iterable
from googletrans import Translator
from walledeval.attacks.mutators.core import Mutator

class TranslationMutator(Mutator):
    def __init__(self, language: str):
        super().__init__("TranslationMutator")
        self.language = language
        self.translator = Translator()

    def mutate(self, prompt: str, **kwargs) -> str:
        result = self.translator.translate(prompt, dest=self.language, src="en")
        translated_prompt = result.text
        return translated_prompt