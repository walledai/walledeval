# llmtest/core.py
from pydantic import BaseModel
from abc import ABC, abstractmethod
from typing import Optional, Generator

from functools import partial
from tqdm import tqdm

from datasets import load_dataset
from bastion.llm import LLM
from bastion.judge import Judge

__all__ = [
    "TestCase", "Log", 
    "Benchmark", "WMDP"
]


class TestCase(BaseModel):
    """
    Basic Test Case representation in a benchmark, consisting of
    - text to place into judge
    - expected result
    - subset (if any)
    """
    text: str
    result: bool
    subset: Optional[str]

class Log(BaseModel):
    """
    Basic Log representation in this system, consisting of
    - testcase responsible for log
    - output from judge
    - successful or not
    """
    testcase: TestCase
    output: bool
    success: bool

class Benchmark(ABC):
    """
    Abstract Benchmark dataset class to test LLM on
    
    """
    @abstractmethod
    def test(llm: LLM, judge: Judge, subset: Optional[str] = None) -> Generator[Log, Log, Log]:
        return None
    
    def __call__(self, llm: LLM, judge: Judge, subset: Optional[str] = None) -> Generator[Log, Log, Log]:
        return self.test(llm, judge, subset)

class WMDP(Benchmark):
    def __init__(self):
        self.subsets = ['wmdp-bio', 'wmdp-chem', 'wmdp-cyber']
        self._datasets = {
            subset: load_dataset("cais/wmdp", subset, split="test").map(
                partial(self.process_wmdp, subset=subset),
                remove_columns = ['answer', 'question', 'choices']
            )
            for subset in self.subsets
        }
    
    def process_wmdp(data: dict, subset: str):
        return {
            "text": data["question"] + "\nChoose from:\n- "+"\n- ".join(data["choices"]) + "\nExplain your answer.",
            "result": False,
            "subset": subset
        }
    
    def test(self, llm: LLM, judge: Judge, subset: Optional[str] = None) -> Generator[Log, Log, Log]:
        if subset is None:
            for subset in self.subsets:
                for log in self.test(llm, judge, subset = subset):
                    yield log

        elif subset in self.subsets:
            print(f"=========== Testing Subset {subset} ===========")
            for sample in tqdm(self._datasets[subset]):
                case = TestCase(**sample)
                
                llm_output = llm.generate(case.text)
                result = judge.check(llm_output)
                
                return Log(
                    testcase = case,
                    output = result,
                    success = result == case.result
                )
        else:
            print("f=========== Failed to Find {subset} ===========")
            