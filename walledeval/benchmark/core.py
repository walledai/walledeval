# walledeval/benchmark/core.py

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional

from datasets import load_dataset, Dataset

from walledeval.types import (
    MultipleChoiceQuestion, MultipleResponseQuestion, 
    OpenEndedQuestion,
    Prompt,
    AutocompletePrompt,
    SystemAssistedPrompt
)

__all__ = [
    "Benchmark", "HuggingFaceBenchmark",
    "MultipleChoiceBenchmark",
    "MultipleResponseBenchmark",
    "OpenEndedBenchmark",
    "PromptBenchmark",
    "AutocompleteBenchmark",
    "SystemAssistedBenchmark"
]

T = TypeVar('T')


class Benchmark(ABC, Generic[T]):
    """Generic Benchmark for some datatype T.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def sample(self, samples: Optional[int] = None) -> list[T]:
        pass


class HuggingFaceBenchmark(Benchmark[T], ABC):
    def __init__(self, name: str, dataset: Dataset):
        super().__init__(name)
        self.dataset = dataset

    @classmethod
    def from_hub(cls, name: str,
                 config: Optional[str] = None,
                 split: str = "train",
                 **ds_kwargs):
        dataset = load_dataset(name, config, split=split, **ds_kwargs)
        return cls(
            name + ("/" + config if config else "") + "/" + split, 
            dataset
        )

    @abstractmethod
    def convert(self, sample: dict) -> T:
        pass

    def __len__(self) -> int:
        return len(self.dataset)

    def sample(self, samples: Optional[int] = None) -> list[T]:
        # take first n samples, and convert it to a list
        samples_lst = self.dataset.select(
            [i for i in range(min(samples, len(self.dataset)))]
        ).to_list()
        return [self.convert(sample) for sample in samples_lst]


class MultipleChoiceBenchmark(HuggingFaceBenchmark[MultipleChoiceQuestion]):
    def convert(self, sample: dict) -> MultipleChoiceQuestion:
        return MultipleChoiceQuestion(
            question=sample["question"],
            choices=sample["choices"],
            answer=sample["answer"]
        )


class MultipleResponseBenchmark(
    HuggingFaceBenchmark[MultipleResponseQuestion]
):
    def convert(self, sample: dict) -> MultipleResponseQuestion:
        return MultipleResponseQuestion(
            question=sample["question"],
            choices=sample["choices"],
            answers=sample["answers"]
        )


class OpenEndedBenchmark(HuggingFaceBenchmark[OpenEndedQuestion]):
    def convert(self, sample: dict) -> OpenEndedQuestion:
        return OpenEndedQuestion(
            question=sample["question"]
        )


class PromptBenchmark(HuggingFaceBenchmark[Prompt]):
    def convert(self, sample: dict) -> Prompt:
        return Prompt(
            prompt=sample["prompt"]
        )


class AutocompleteBenchmark(HuggingFaceBenchmark[AutocompletePrompt]):
    def convert(self, sample: dict) -> AutocompletePrompt:
        return AutocompletePrompt(
            prompt=sample["prompt"]
        )


class SystemAssistedBenchmark(HuggingFaceBenchmark[SystemAssistedPrompt]):
    def convert(self, sample: dict) -> SystemAssistedPrompt:
        return SystemAssistedPrompt(
            prompt=sample["prompt"],
            system=sample["system"]
        )