# walledeval/util/messages.py

from walledeval.types.message import Message, Messages

__all__ = [
    "transform_messages"
]


def transform_messages(input: Messages, system_prompt: str = "") -> list[dict[str, str]]:
    messages: list[dict[str, str]]
    if isinstance(input, str):
        messages = [{
            "role": "user",
            "content": input
        }]
    elif isinstance(input, list) and isinstance(input[0], Message):
        messages = [
            dict(msg)
            for msg in input
        ]
    elif isinstance(input, list) and isinstance(input[0], dict):
        messages = input
    
    else:
        raise TypeError("Unsupported format for messages item")
    
    if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": system_prompt})
    
    return messages


    