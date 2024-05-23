# walledeval/benchmark/core.py

from abc import ABC, abstractmethod

from typing import TypeVar, Generic, Optional

from datasets import load_dataset, Dataset

from walledeval.types import (
    MultipleChoiceQuestion, MultipleResponseQuestion, OpenEndedQuestion
)

__all__ = [
    "Benchmark", "HuggingFaceBenchmark",
    "MultipleChoiceBenchmark",
    "MultipleResponseBenchmark",
    "OpenEndedBenchmark"
]

T = TypeVar('T')

class Benchmark(ABC, Generic[T]):
    """Generic Benchmark for some datatype T.

    Args:
        ABC (_type_): _description_
        Generic (_type_): _description_
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
                 configuration: Optional[str] = None,
                 **ds_kwargs):
        dataset = load_dataset(name, configuration, **ds_kwargs)
        return cls(name + "/" + configuration, dataset)
    
    @abstractmethod
    def convert(self, sample: dict) -> T:
        pass
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def sample(self, samples: Optional[int] = None) -> list[T]:
        # take first n samples, and convert it to a list
        samples_lst = self.dataset.select([i for i in range(min(samples, len(self.dataset)))]).to_list()
        return [self.convert(sample) for sample in samples_lst]


class MultipleChoiceBenchmark(HuggingFaceBenchmark[MultipleChoiceQuestion]):
    def convert(self, sample: dict) -> MultipleChoiceQuestion:
        return MultipleChoiceQuestion(
            question = sample["question"],
            choices = sample["choices"],
            answer = sample["answer"]
        )

class MultipleResponseBenchmark(HuggingFaceBenchmark[MultipleResponseQuestion]):
    def convert(self, sample: dict) -> MultipleResponseQuestion:
        return MultipleResponseQuestion(
            question = sample["question"],
            choices = sample["choices"],
            answers = sample["answer"]
        )

class OpenEndedBenchmark(HuggingFaceBenchmark[OpenEndedQuestion]):
    def convert(self, sample: dict) -> OpenEndedQuestion:
        return OpenEndedQuestion(
            question = sample["question"]
        )
