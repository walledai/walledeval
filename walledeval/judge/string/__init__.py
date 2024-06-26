# walledeval/judge/string/__init__.py
from pathlib import Path

import yaml

from walledeval.judge import Judge

__all__ = ["StringMatchingJudge"]


class StringMatchingJudge(Judge[None, bool]):

    def __init__(self, name: str, must_match_all: list[str] = None, must_mismatch_all: list[str] = None,
                 caseless: bool = False):
        super().__init__(name)

        self.must_match_all, self.must_mismatch_all, self.caseless = StringMatchingJudge.from_preset(name) if (
                must_match_all is None and must_mismatch_all is None
        ) else must_match_all, must_mismatch_all, caseless

    def check(self, response: str, answer: None = None) -> bool:
        text: str = response.casefold() if self.caseless else response

        return (
                (
                        not self.must_mismatch_all
                        or not any(target in text for target in self.must_mismatch_all)
                )
                and (
                        not self.must_match_all
                        or all(target in text for target in self.must_match_all)
                )
        )

    @classmethod
    def from_preset(cls, name: str):
        # presets are adapted from https://github.com/ThuCCSLab/JailbreakEval

        yaml_file_path = Path(f"/presets/{name}.yaml").resolve()
        yaml_dict = yaml.safe_load(yaml_file_path.read_text(encoding="utf-8"))

        return yaml_dict["must_match_all"], yaml_dict["must_mismatch_all"], yaml_dict["caseless"]
