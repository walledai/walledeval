# Input Types

We have added support for several types of input formats in LLMs (with more on the way!) to make our library easily extensible and usable.

Our LLM architecture supports the following input types:

| Input Type | Format | Example |
| ---------- | ------ | ------- |
| `str` | `"text to ask LLM as user"` | `"Hi, how are you today?"` |
| `list[dict[str, str]]` | List of dictionary objects with the following keys: <ul><li> `"role"`: Either one of `"system"`, `"user"`, `"assistant"`. </li><li> `"content"`: Any string or alternative input supported by the model tokenizer. </li></ul> | `[ {"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": "Hi, how are you today?"} ]` |
| `list[walledeval.types.Message]` | Similar to above, except the dictionary object is wrapped within a custom Pydantic model class | `[ Message(role="system", content="You are a helpful assistant"), Message(role="user", content="Hi, how are you today?") ]` |

These are supported under an encompassing `walledeval.types.Messages` class. The supported LLMs convert these into recognizable formats for the LLM to generate based on. Certain class methods cannot support some of these formats due to their expected formats.
