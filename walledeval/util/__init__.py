# walledeval/util/__init__.py

from walledeval.types.data import Range

__all__ = [
    "process_range"
]


def process_range(range_obj: Range, length: int) -> list[int]:
    idx_list: list[int]

    if isinstance(range_obj, int):
        if range_obj < -length:
            raise IndexError(f"Index value {range_obj} is too low.")
        if range_obj >= length:
            raise IndexError(f"Index value {range_obj} is too high")
        idx_list = [
            range_obj % length
        ]

    elif isinstance(range_obj, slice):
        idx_list = [
            i % length
            for i in range(length)[range_obj]
        ]

    elif isinstance(range_obj, list):
        idx_list = [
            idx % length for idx in range_obj
            if -length <= idx < length
        ]

    else:
        raise ValueError(f"Invalid range value {range_obj}")

    return idx_list