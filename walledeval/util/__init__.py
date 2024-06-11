# walledeval/util/__init__.py

from walledeval.types.data import Range

__all__ = [
    "process_range"
]


def process_range(range: Range, length: int) -> list[int]:
    idx_list: list[int]

    if isinstance(range, int):
        if range < -length:
            raise IndexError(f"Index value {range} is too low.")
        if range >= length:
            raise IndexError(f"Index value {range} is too high")
        idx_list = [
            range % length
        ]

    elif isinstance(range, slice):
        idx_list = [
            i % length
            for i in range(length)[range]
        ]

    elif isinstance(range, list):
        idx_list = [
            idx % length for idx in range
            if -length <= idx < length
        ]

    else:
        raise ValueError(f"Invalid range value {range}")

    return idx_list