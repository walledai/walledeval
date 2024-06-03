# Supporting your own LLMs

Assuming one cannot put up their models on HuggingFace Hub, they can also define their own classes to add support for their own LLMs using the abstract `llm.LLM` class. 

To support your own LLMs, you can extend this class and implement the following methods:

- `__init__`: Instantiates the LLM, calls superclass instantiation
- `complete(text: str, max_new_tokens: int = 256, temperature: float = 0.0) -> str`: Completion of text string
- `chat(text: Messages, max_new_tokens: int = 256, temperature: float = 0.0) -> str`: Allows chatting

Here is an example of a dummy class defined to inherit the `LLM` class, which only supports next-token prediction.

```python
from walledeval.llm import LLM
from walledeval.types import LLMType, Messages

class DummyLLM(LLM):
    def __init__(self):
        super().__init__(name = "DummyLLM", system_prompt = "", type = LLMType.BASE)
    
    def complete(self,
                 text: str, 
                 max_new_tokens: int = 256, 
                 temperature: float = 0.0) -> str:
        return text[:max_new_tokens]
    
    def chat(self,
             text: Messages,
             max_new_tokens: int = 256,
             temperature: float = 0.0) -> str:
        raise NotImplementedError("DummyLLM does not support chat completion!")
```