# walledeval/benchmark/core.py

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Union

from tqdm import tqdm

from walledeval.types import Range, NumericScore, Log, Report
from walledeval.util import process_range
from walledeval.data import Dataset

__all__ = [
    "BenchmarkModule"
]


I = TypeVar('I') # Input Field
O = TypeVar('O') # Output Field
S = TypeVar('S') # Score Field
A = TypeVar('A') # Aggregate Field


class BenchmarkModule(ABC, Generic[I, O, S, A]):
    def __init__(self, name: str, dataset: Dataset):
        self.name = name
        self.dataset = dataset
        self.outputs: dict[int, ] = {
            idx: None for idx in range(len(dataset))
        }

    def _get_samples(self, range: Range) -> tuple[list[int], list[I]]:
        idx_list = process_range(range, len(self.dataset))
        samples = self.dataset[idx_list]
        return idx_list, samples

    def run(self,
            range: Range,
            verbose: bool = False):
        idx_list, samples = self._get_samples(range)

        tqdm_wrap = tqdm if verbose else (lambda it : it)

        for idx, input in tqdm_wrap(zip(idx_list, samples)):
            output = self.evaluate(input)
            score = self.score(input, output)
            self.outputs[idx] = Log[I, O, S](
                idx=idx,
                input=input,
                output=output,
                score=score
            )

    def generate_report(self) -> Report:
        runs = [
            log for log in self.outputs
            if log is not None
        ]

        aggregate = self.aggregate(runs)

        return Report(
            runs=len(runs),
            logs=runs,
            aggregate=aggregate
        )

    @abstractmethod
    def evaluate(self, input: I) -> O:
        pass

    @abstractmethod
    def score(self, input: I, output: O) -> S:
        pass

    @abstractmethod
    def aggregate(self, logs: list[Log]) -> A:
        pass