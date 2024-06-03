# walledeval/types/llm.py

from enum import Enum

__all__ = ["LLMType"]


class LLMType(Enum):
    BASE = 0
    INSTRUCT = 1
    NEITHER = 2
