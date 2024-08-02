# walledeval/data/core.py

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional, Union
from pydantic import BaseModel

from datasets import load_dataset
import datasets

from walledeval.types import Prompt, Range
from walledeval.util import process_range

__all__ = [
    "Dataset", "HuggingFaceDataset"
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

    @abstractmethod
    def __getitem__(self, range: Range) -> list[T]:
        pass
    
    @abstractmethod
    def all(self) -> list[T]:
        pass


class _HuggingFaceDataset(Dataset[T], ABC):
    def __init__(self, name: str, dataset: datasets.Dataset):
        super().__init__(name)
        self.dataset = dataset

    @abstractmethod
    def convert(self, sample: dict) -> T:
        pass

    def __len__(self) -> int:
        return len(self.dataset)

    def sample(self, samples: Optional[int] = None) -> list[T]:
        # take first n samples, and convert it to a list
        return self[:samples]
    
    def __getitem__(self, range: Range) -> list[T]:
        if isinstance(range, int):
            return self.convert(self.dataset[range % len(self.dataset)])
        
        idx_list = process_range(range, len(self.dataset))

        samples_lst = self.dataset.select(idx_list).to_list()
        return [self.convert(sample) for sample in samples_lst]
    
    def all(self) -> list[T]:
        return self[:]
    
    def __iter__(self):
        for idx in range(len(self)):
            yield self.convert(self.dataset[idx])


class _HuggingFaceDatasetAlias:
    def __init__(self, model: type = Prompt):
        self.model = model
    
    def __call__(self, name: str, dataset: datasets.Dataset):
        return HuggingFaceDataset(name, dataset, self.model)
    
    def from_hub(self, 
                 name: str, 
                 config: Optional[str] = None, 
                 split: str = "DEFAULT",
                 **ds_kwargs):
        return HuggingFaceDataset.from_hub(
            name, config, split, self.model, **ds_kwargs
        )
    
    def from_list(self, name: str, lst: list[dict]):
        return HuggingFaceDataset.from_list(
            name, lst, self.model
        )
    
    def from_csv(self, filenames: Union[str, list[str]], **csv_kwargs):
        return HuggingFaceDataset.from_csv(
            filenames, self.model, **csv_kwargs
        )
    
    def from_json(self, filenames: Union[str, list[str]], **json_kwargs):
        return HuggingFaceDataset.from_json(
            filenames[0], self.model, **json_kwargs
        )


class HuggingFaceDataset(_HuggingFaceDataset):
    def __init__(self, name: str, dataset: datasets.Dataset, model: type = Prompt):
        _HuggingFaceDataset.__init__(self, name, dataset)
        self.model = model
        
    @classmethod
    def from_hub(cls, name: str,
                 config: Optional[str] = None,
                 split: str = "DEFAULT",
                 model: type = Prompt,
                 **ds_kwargs):
        dataset = load_dataset(name, config, **ds_kwargs)
        
        splits = tuple(dataset.keys())
        
        if split in splits:
            dataset = dataset[split]
        elif split == "DEFAULT":
            if "train" in splits:
                dataset = dataset["train"]
            elif "test" in splits:
                dataset = dataset["test"]
            else:
                split = splits[0]
                dataset = dataset[split]
        else:
            raise NameError(f"Requested split '{split}' not found in dataset {name}/{config}, select one of {splits}")
        
        return cls(
            name + ("/" + config if config else "") + ("/" + split if split != "DEFAULT" else ""), 
            dataset,
            model
        )
    
    @classmethod
    def from_list(cls, name: str, 
                  lst: list[Union[dict, str]], 
                  model: type = Prompt):
        lst = [
            it
            if isinstance(it, dict)
            else {"prompt": it} 
            for it in lst
        ]
        dataset = datasets.Dataset.from_list(lst)
        return cls(name, dataset, model)
    
    @classmethod
    def from_csv(cls, filenames: Union[str, list[str]], model: type = Prompt, **csv_kwargs):
        filenames = [filenames] if isinstance(filenames, str) else filenames
        dataset = load_dataset(
            "csv", 
            data_files=filenames,
            **csv_kwargs
        )['train']
        
        return cls(
            filenames[0],
            dataset,
            model
        )
    
    @classmethod
    def from_json(cls, filenames: Union[str, list[str]], model: type = Prompt, **json_kwargs):
        filenames = [filenames] if isinstance(filenames, str) else filenames
        dataset = load_dataset(
            "json", 
            data_files=filenames,
            **json_kwargs
        )['train']
        
        return cls(
            filenames[0],
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


