# walledeval/evaluation/core.py

from abc import ABC, abstractmethod

class Metric(ABC):
    """
    Abstract base class for evaluation metrics.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def evaluate(self, original_text: str, mutated_text: str) -> float:
        pass
