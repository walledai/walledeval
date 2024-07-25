# walledeval/attacks/mutators/sorry

from typing import Iterable
from walledeval.prompts import PromptTemplate
from walledeval.attacks.mutators.core import Mutator


class QuestionMutator(Mutator):
    def __init__(self, samples: Iterable[str] = []):
        super().__init__("QuestionMutator")
        self.template = PromptTemplate.from_preset("mutations/sorry/question")
        self.samples = list(samples)

    def encode(self, text: str) -> str:
        if not text.endswith("?"):
            text += "?"
        return text

    def mutate(self, prompt: str, **kwargs) -> str:
        examples = ""
        for i, sample in enumerate(self.samples):
            examples += f"Example {i+1}:\n{self.encode(sample)}\n\n"
        
        return self.template.format(
            prompt=self.encode(prompt),
            examples=examples.strip(),
            **kwargs
        )

class SlangMutator(Mutator):
    def __init__(self, samples: Iterable[str] = []):
        super().__init__("SlangMutator")
        self.template = PromptTemplate.from_preset("mutations/sorry/slang")
        self.samples = list(samples)

    def encode(self, text: str) -> str:

        slang_replacements = {
            "you": "u",
            "are": "r",
            "please": "pls",
            "people": "ppl",
            "really": "rly",
            "have to": "hafta",
            "to": "2",
            "for": "4",
            "before": "b4",
            "great": "gr8",
            "see you": "cya",
            "be right back": "brb",
            "oh my god": "omg",
            "laughing out loud": "lol"
        }

        for word, slang in slang_replacements.items():
            text = text.replace(word, slang)
        return text

    def mutate(self, prompt: str, **kwargs) -> str:
        examples = ""
        for i, sample in enumerate(self.samples):
            examples += f"Example {i+1}:\n{self.encode(sample)}\n\n"
        
        return self.template.format(
            prompt=self.encode(prompt),
            examples=examples.strip(),
            **kwargs
        )

class UncommonDialectMutator(Mutator):
    def __init__(self, samples: Iterable[str] = []):
        super().__init__("UncommonDialectMutator")
        self.template = PromptTemplate.from_preset("mutations/sorry/uncommon_dialect")
        self.samples = list(samples)

    def encode(self, text: str) -> str:

        uncommon_dialect_replacements = {
            "hello": "howdy",
            "friend": "mate",
            "yes": "aye",
            "no": "nay",
            "child": "wee one",
            "food": "grub",
            "money": "quid",
            "good": "grand",
            "bad": "dreadful"
        }

        for word, dialect in uncommon_dialect_replacements.items():
            text = text.replace(word, dialect)
        return text

    def mutate(self, prompt: str, **kwargs) -> str:
        examples = ""
        for i, sample in enumerate(self.samples):
            examples += f"Example {i+1}:\n{self.encode(sample)}\n\n"
        
        return self.template.format(
            prompt=self.encode(prompt),
            examples=examples.strip(),
            **kwargs
        )

class TranslationMutator(Mutator):
    def __init__(self, language: str, samples: Iterable[str] = []):
        super().__init__("TranslationMutator")
        self.language = language
        self.template = PromptTemplate.from_preset("mutations/sorry/translation")
        self.samples = list(samples)

    def encode(self, text: str) -> str:
        try:
            translation = self.translator.translate(text, dest=self.language, src='en')
            return translation.text
        except Exception as e:
            print(f"Translation error: {e}")
            return text

    def mutate(self, prompt: str, **kwargs) -> str:
        examples = ""
        for i, sample in enumerate(self.samples):
            examples += f"Example {i+1}:\n{self.encode(sample)}\n\n"

        return self.template.format(
            prompt=self.encode(prompt),
            examples=examples.strip(),
            language=self.language,
            **kwargs
        )
