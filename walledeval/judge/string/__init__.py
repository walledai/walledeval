# walledeval/judge/string/__init__.py

import yaml
from pathlib import Path

from walledeval.judge.core import Judge

__all__ = [
    "StringMatchingJudge"
]


class StringMatchingJudge(Judge[None, bool]):
    def __init__(self, name: str,
                 must_match_all: list[str] = [],
                 must_mismatch_all: list[str] = [],
                 caseless: bool = False):
        super().__init__(name)
        
        self.must_match_all = must_match_all
        self.must_mismatch_all = must_mismatch_all
        self.caseless = caseless
    
    @classmethod
    def from_preset(cls, name: str = "beta"):
        # presets are adapted from https://github.com/ThuCCSLab/JailbreakEval
        yaml_fp = Path(__file__).resolve().parent / f"presets/{name}.yaml"
        yaml_text = yaml_fp.read_text(encoding="utf-8")
        yaml_dict: dict = yaml.safe_load(yaml_text)
        
        return cls(
            name=name,
            must_match_all=yaml_dict.get("must_match_all", []),
            must_mismatch_all=yaml_dict.get("must_mismatch_all", []),
            caseless=yaml_dict.get("caseless", False)
        )

    def check(self, response: str, answer: None = None) -> bool:
        text = response.casefold() if self.caseless else response

        return (
            (
                not self.must_mismatch_all or 
                not any(target in text for target in self.must_mismatch_all)
            ) and (
                not self.must_match_all or 
                all(target in text for target in self.must_match_all)
            )
        )
