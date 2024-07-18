from typing import List, Union
from pydantic import BaseModel

__all__ = [
    "Message",
    "Messages"
]


class Message(BaseModel):
    role: str
    content: str

# changed this class to a concrete class from a union class as typeerror occur when running the test
class Messages(BaseModel):
    messages: List[Union[Message, dict]]

    def __init__(self, messages: List[Union[Message, dict]]):
        super().__init__(messages=messages)
