# walledeval/types/message.py

from typing import Union
from pydantic import BaseModel

__all__ = [
    "Message",
    "Messages"
]


class Message(BaseModel):
    role: str
    content: str


Messages = Union[
    list[Message],
    list[dict[str, str]],
    str
]