# Other API Support

WalledEval also currently supports the following alternative LLM types:


| Class | LLM Type |
| ----- | -------- |
| `Claude(model_id, api_key, system_prompt = "", type = LLMType.NEITHER)` | Claude 3 (`Claude.haiku`, `Claude.sonnet` and `Claude.opus` class methods exist to initiate the most recent versions of each of these models) |