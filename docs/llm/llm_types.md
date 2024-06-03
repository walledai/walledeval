# LLM Types

Our LLM architecture supports two types of models: `INSTRUCT` and `BASE`. The distinction between these two model types is as follows:

| LLM Type | Function | Corresponding Number |
| -------- | -------- | -------------------- |
| `BASE` | Next-token predictor LLMs that support text completion but are not tuned for chatting and role-based conversation. | `0` |
| `INSTRUCT` | Instruction-tuned / Chat-tuned LLMs that can take in a chat format and generate text for an assistant. | `1` |

These types fall under the `walledeval.types.LLMType` enumeration class, and we support a `NEITHER` flag (with corresponding number `2`) to ensure the LLM does not discriminate between types.