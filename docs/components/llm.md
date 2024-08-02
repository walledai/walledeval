# LLMs

WalledEval's LLM architecture aims to support various kinds of LLMs. These LLMs are used as **systems-under-test (SUTs)**, which allows generating question answers and prompt outputs. Below is a list of model families we attempt to support.

| Model Family                                                     | Supported Versions                    | WalledEval Class |
| ---------------------------------------------------------------- | ------------------------------------- | ---------------- |
| [GPT](https://platform.openai.com/docs/overview)                 | 3.5 Turbo, 4, 4 Turbo, 4o             | `llm.OpenAI`     |
| [Claude](https://docs.anthropic.com/en/docs/about-claude/models) | Sonnet 3.5, Opus 3, Sonnet 3, Haiku 3 | `llm.Claude`     |
| [Gemini](https://ai.google.dev/)                                 | 1.5 Flash, 1.5 Pro, 1.0 Pro           | `llm.Gemini`     |
| [Cohere Command](https://cohere.com/command)                     | R+, R, Base, Light                    | `llm.Cohere`     |

We also support a large variety of connectors to other major LLM runtimes, like HuggingFace and TogetherAI. Below is a list of some of the many connectors present in WalledEval.

| Connector                                                                             | Connector Type              | WalledEval Class  |
| ------------------------------------------------------------------------------------- | --------------------------- | ----------------- |
| [HuggingFace](https://huggingface.co/models)                                          | Local, runs LLM on computer | `llm.HF_LLM`      |
| [`llama.cpp`](https://github.com/ggerganov/llama.cpp)                                 | Local, runs LLM on computer | `llm.Llama`       |
| [Together](https://www.together.ai/)                                                  | Online, makes API calls     | `llm.Together`    |
| [Groq](https://groq.com/)                                                             | Online, makes API calls     | `llm.Groq`        |
| [Anyscale](https://www.anyscale.com/)                                                 | Online, makes API calls     | `llm.Anyscale`    |
| [OctoAI](https://octo.ai/)                                                            | Online, makes API calls     | `llm.OctoAI`      |
| [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service) | Online, makes API calls     | `llm.AzureOpenAI` |








The `HF_LLM` is an example of a LLM class that loads models from HuggingFace. Here, we load Unsloth's 4-bit-quantized Llama 3 8B model as follows. The type is essentially used to indicate that we are loading an instruction-tuned model so it does inference based on that piece of information. It is important that we do this because we don't want the model to autocomplete responses to the prompt, but instead complete chat responses to the prompt.

We can then prompt this LLM using the `chat` method, and we have tried to get it to generate a response the same way a Swiftie would.

WalledEval attempts