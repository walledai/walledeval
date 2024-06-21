# walledeval/types/data.py
from typing import Union

__all__ = ["Range"]

Range = Union[slice, int, list[int]]