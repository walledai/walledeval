# walledeval/attacks/mutators/translation.py

from typing import Iterable
from googletrans import Translator
from walledeval.attacks.mutators.core import Mutator

class TranslationMutator(Mutator):
    def __init__(self, language: str, samples: Iterable[str] = []):
        super().__init__("TranslationMutator")
        self.language = language
        self.translator = Translator()
        self.samples = list(samples)

    def mutate(self, prompt: str, **kwargs) -> str:
        examples = ""
        for i, sample in enumerate(self.samples):
            translated_sample = self.translate_text(sample)
            examples += f"Example {i+1}:\n{translated_sample}\n\n"
        
        translated_prompt = self.translate_text(prompt)
        extra_info = "\n".join([f"{key}: {value}" for key, value in kwargs.items()])
        
        return f"Translated Prompt:\n{translated_prompt}\n\nExamples:\n{examples.strip()}\n\nExtra Information:\n{extra_info}"
    
    def translate_text(self, text: str) -> str:
        result = self.translator.translate(text, dest=self.language)
        return result.text
