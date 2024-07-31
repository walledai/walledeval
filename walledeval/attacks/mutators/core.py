# walledeval/attacks/mutators/core.py

from abc import ABC, abstractmethod
from typing import Union
from collections.abc import Iterable

from walledeval.types.data import Range
from walledeval.util import process_range


class Mutator(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def mutate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError
    
    def __call__(self, prompt: str, **kwargs) -> str:
        return self.mutate(prompt, **kwargs)
    
    def __add__(self, other) -> str:
        return CompositeMutator(self, other)
    
    def __radd__(self, other) -> str:
        return CompositeMutator(other, self)


class CompositeMutator(Mutator):
    def __init__(self, *mutators):
        # flatten mutators array
        self.mutators = self.flatten(mutators)
    
    def flatten(self, mutators):
        # Recursively flatten mutators
        new_mutators = []
        
        for mutator in mutators:
            if isinstance(mutator, CompositeMutator):
                # I don't trust myself so I put this failsafe here
                new_mutators.extend(self.flatten(mutator.mutators))
            
            elif (
                isinstance(mutator, Iterable) and 
                isinstance(mutator[0], Mutator)
            ):
                new_mutators.extend(self.flatten(mutator))
            
            else:
                new_mutators.append(mutator)
        
        return new_mutators
        
    
    def mutate(self, prompt: str, **kwargs) -> str:
        for mutator in self.mutators:
            prompt = mutator(prompt, **kwargs)
        return prompt
    
    def __len__(self) -> int:
        return len(self.mutators)
    
    def __add__(self, other: Union[Mutator, Iterable[Mutator]]) -> Mutator:
        return CompositeMutator(self, other)
    
    def __radd__(self, other: Union[Mutator, Iterable[Mutator]]) -> Mutator:
        return CompositeMutator(other, self)
    
    def __iadd__(self, other: Union[Mutator, Iterable[Mutator]]) -> Mutator:
        self.mutators.append(self.flatten([other]))
        
    def add(self, *mutators):
        self.mutators.append(self.flatten(mutators))
    
    def pop(self, idx: int) -> Mutator:
        mutator = self.mutators.pop(idx)
        return mutator
        
    def __getitem__(self, range: Range) -> Mutator:
        if isinstance(range, int):
            return self.mutators[range]
        idx_list = process_range(range, len(self))
        
        return CompositeMutator([
            mutator for idx, mutator in enumerate(self.mutators)
            if idx in idx_list
        ])
    
    def __setitem__(self, range: Range, mutator: Union[Iterable[Mutator], Mutator]):
        if isinstance(range, int):
            self.mutators[range] = mutator
        else:
            idx_list = process_range(range, len(self))
            if len(idx_list) == 1:
                self.mutators[idx_list[0]] = mutator
            elif isinstance(mutator, Mutator):
                for idx in idx_list:
                    self.mutators[idx] = mutator
                
            elif len(mutator) == len(idx_list):
                for idx, mutator in zip(idx_list, mutator):
                    self.mutators[idx] = mutator
            
            elif len(mutator) < len(idx_list):
                raise ValueError("Insufficient values provided for range")
            else:
                raise ValueError("Insufficient range provided for values")
        
        # to remove all rogue CompositeMutators / sub-lists
        self.mutators = self.flatten(self.mutators)