# walledeval/benchmark/core.py

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional
from pydantic import BaseModel

from datasets import load_dataset
import datasets #Dataset

from walledeval.types import (
    MultipleChoiceQuestion, MultipleResponseQuestion, 
    OpenEndedQuestion,
    Prompt,
    AutocompletePrompt,
    SystemAssistedPrompt
)

__all__ = [
    "Dataset", "HuggingFaceDataset",
    "MultipleChoiceDataset",
    "MultipleResponseDataset",
    "OpenEndedDataset",
    "PromptDataset",
    "AutocompleteDataset",
    "SystemAssistedDataset"
]

T = TypeVar('T')
I = TypeVar('I', bound=BaseModel)


class Dataset(ABC, Generic[T]):
    """Generic Benchmark for some datatype T.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def sample(self, samples: Optional[int] = None) -> list[T]:
        pass


class _HuggingFaceDataset(Dataset[T], ABC):
    def __init__(self, name: str, dataset: datasets.Dataset):
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


class _HuggingFaceDatasetAlias:
    def __init__(self, model: type = Prompt):
        self.model = model
    
    def __call__(self, name: str, dataset: datasets.Dataset):
        return HuggingFaceDataset(name, dataset, self.model)
    
    def from_hub(self, 
                 name: str, 
                 config: Optional[str] = None, 
                 split: str = "train",
                 **ds_kwargs):
        return HuggingFaceDataset.from_hub(
            name, config, split, self.model, **ds_kwargs
        )


class HuggingFaceDataset(_HuggingFaceDataset):
    def __init__(self, name: str, dataset: datasets.Dataset, model: type = Prompt):
        _HuggingFaceDataset.__init__(self, name, dataset)
        self.model = model
        
    @classmethod
    def from_hub(cls, name: str,
                 config: Optional[str] = None,
                 split: str = "train",
                 model: type = Prompt,
                 **ds_kwargs):
        dataset = load_dataset(name, config, split=split, **ds_kwargs)
        return cls(
            name + ("/" + config if config else "") + "/" + split, 
            dataset,
            model
        )
    
    def __class_getitem__(cls, model: type = Prompt):
        # Refer to https://stackoverflow.com/questions/73464414/why-are-generics-in-python-implemented-using-class-getitem-instead-of-geti
        # for why it is implemented like this
        return _HuggingFaceDatasetAlias(model)

    def convert(self, sample: dict) -> I:
        fields = self.model.__fields__
        return self.model(**{
            field: sample[field]
            for field in fields.keys()
        })


class MultipleChoiceDataset(_HuggingFaceDataset[MultipleChoiceQuestion]):
    def convert(self, sample: dict) -> MultipleChoiceQuestion:
        return MultipleChoiceQuestion(
            question=sample["question"],
            choices=sample["choices"],
            answer=sample["answer"]
        )


class MultipleResponseDataset(
    _HuggingFaceDataset[MultipleResponseQuestion]
):
    def convert(self, sample: dict) -> MultipleResponseQuestion:
        return MultipleResponseQuestion(
            question=sample["question"],
            choices=sample["choices"],
            answers=sample["answers"]
        )


class OpenEndedDataset(_HuggingFaceDataset[OpenEndedQuestion]):
    def convert(self, sample: dict) -> OpenEndedQuestion:
        return OpenEndedQuestion(
            question=sample["question"]
        )


class PromptDataset(_HuggingFaceDataset[Prompt]):
    def convert(self, sample: dict) -> Prompt:
        return Prompt(
            prompt=sample["prompt"]
        )


class AutocompleteDataset(_HuggingFaceDataset[AutocompletePrompt]):
    def convert(self, sample: dict) -> AutocompletePrompt:
        return AutocompletePrompt(
            prompt=sample["prompt"]
        )


class SystemAssistedDataset(_HuggingFaceDataset[SystemAssistedPrompt]):
    def convert(self, sample: dict) -> SystemAssistedPrompt:
        return SystemAssistedPrompt(
            prompt=sample["prompt"],
            system=sample["system"]
        )