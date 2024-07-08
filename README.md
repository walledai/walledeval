# WalledEval: Testing LLMs Against Jailbreaks and Unprecedented Harms

<!-- [![Python Package tests status](https://github.com/three-body-analysis/tris/actions/workflows/python-package.yml/badge.svg)](https://github.com/three-body-analysis/tris/actions?query=workflow%3Apython-package) -->
<!-- [![Docs CI status](https://github.com/three-body-analysis/tris/actions/workflows/docs.yml/badge.svg)](https://three-body-analysis.github.io/tris/) -->
[![PyPI Latest Release](https://img.shields.io/pypi/v/walledeval.svg)](https://pypi.org/project/walledeval/)
[![PyPI Downloads](https://static.pepy.tech/badge/walledeval)](https://pepy.tech/project/walledeval)
[![GitHub Page Views Count](https://badges.toozhao.com/badges/01J0NWXGZ7XGDPFYWHZ9EX1F46/blue.svg)](https://github.com/walledai/walledeval)

**WalledEval** is a simple library to test LLM safety by identifying if text generated by the LLM is indeed safe. We purposefully test benchmarks with negative information and toxic prompts to see if it is able to flag prompts of malice.

> [!NOTE]  
> We have recently released `v0.1.0` of our codebase! This means that our documentation is not completely up-to-date with the current state of the codebase. However, we will be updating our documentation soon for all users to be able to quickstart using WalledEval! Till then, it is always best to consult the code or the `tests/` or `notebooks/` folders to have a better idea of how the codebase currently works.

## Announcement
> 🔥 Excited to announce the release of the community version of our guardrails: [WalledGuard](https://huggingface.co/walledai/walledguard-c)! **Walled Guard** comes in two versions: **Community** and **Advanced+**. We are releasing the community version under the Apache-2.0 License. To get access to the advanced version, please contact us at [admin@walled.ai](mailto:admin@walled.ai)

> 🔥 Excited to partner with IMDA Singapore AI Verify Foundation to build robust AI safety and controllability measures!

> 🔥 Grateful to [Tensorplex](https://www.tensorplex.ai/) for their support with computing resources!

## Installation

### Installing from PyPI

Yes, we have published WalledEval on PyPI! To install WalledEval and all its dependencies, the easiest method would be to use 
`pip` to query PyPI. This should, by default, be present in your Python installation. To, install run the following 
command in a terminal or Command Prompt / Powershell:

```bash
$ pip install walledeval
```

Depending on the OS, you might need to use `pip3` instead. If the command is not found, you can choose to use the
following command too:

```bash
$ python -m pip install walledeval
```

Here too, `python` or `pip` might be replaced with `py` or `python3` and `pip3` depending on the OS and installation 
configuration. If you have any issues with this, it is always helpful to consult 
[Stack Overflow](https://stackoverflow.com/).

### Installing from Source

To install from source, you need to get the following:

#### Git

Git is needed to install this repository. This is not completely necessary as you can also install the zip file for this 
repository and store it on a local drive manually. To install Git, follow 
[this guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

After you have successfully installed Git, you can run the following command in a terminal / Command Prompt etc:

```bash
$ git clone https://github.com/walledai/walledeval.git
```

This stores a copy in the folder `walledeval`. You can then navigate into it using `cd walledeval`.

#### Poetry

This project can be used easily via a tool known as Poetry. This allows you to easily reflect edits made in the original 
source code! To install `poetry`, you can also install it using `pip` by typing in the command as follows:

```bash
$ pip install poetry
```

Again, if you have any issues with `pip`, check out [here](#installing-from-pypi).

After this, you can use the following command to install this library:

```bash
$ poetry install
```


## Basic Usage

### LLMs (`walledeval.llm`)

WalledEval's LLM architecture aims to support various kinds of LLMs, which a current focus on Decoder-only and MoE architecures. These LLMs are used as **systems-under-test (SUTs)**, which allows generating question answers and prompt outputs.

#### LLM Types

Our LLM architecture supports two types of models: `INSTRUCT` and `BASE`. The distinction between these two model types is as follows:

| LLM Type | Function | Corresponding Number |
| -------- | -------- | -------------------- |
| `BASE` | Next-token predictor LLMs that support text completion but are not tuned for chatting and role-based conversation. | `0` |
| `INSTRUCT` | Instruction-tuned / Chat-tuned LLMs that can take in a chat format and generate text for an assistant. | `1` |

These types fall under the `walledeval.types.LLMType` enumeration class, and we support a `NEITHER` flag (with corresponding number `2`) to ensure the LLM does not discriminate between types.

#### Input Types

We have added support for several types of input formats in LLMs (with more on the way!) to make our library easily extensible and usable.

Our LLM architecture supports the following input types:

| Input Type | Format | Example |
| ---------- | ------ | ------- |
| `str` | `"text to ask LLM as user"` | `"Hi, how are you today?"` |
| `list[dict[str, str]]` | List of dictionary objects with the following keys: <ul><li> `"role"`: Either one of `"system"`, `"user"`, `"assistant"`. </li><li> `"content"`: Any string or alternative input supported by the model tokenizer. </li></ul> | `[ {"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": "Hi, how are you today?"} ]` |
| `list[walledeval.types.Message]` | Similar to above, except the dictionary object is wrapped within a custom Pydantic model class | `[ Message(role="system", content="You are a helpful assistant"), Message(role="user", content="Hi, how are you today?") ]` |

These are supported under an encompassing `walledeval.types.Messages` class. The supported LLMs convert these into recognizable formats for the LLM to generate based on. Certain class methods cannot support some of these formats due to their expected formats.


#### HuggingFace LLM Support

WalledEval supports a plethora of LLM models accessible through the HuggingFace Hub. This means that any model deployed on HuggingFace under the `text-generation` task can be loaded up as a SUT.

These LLMs can be accessed via the `walledeval.llm.HF_LLM` class. Here is a table to summarise the methods present in the `HF_LLM` class:

| Method | Purpose | Parameters |
| ------ | ------- | ---------- |
| `HF_LLM(id, system_prompt = "", type = LLMType.NEITHER)` | Initiates LLM from HuggingFace Hub | <ul><li> `id` (`str`): Idenitifer of LLM from HuggingFace Hub. For example, [`"meta-llama/Meta-Llama-3-8B"`](https://huggingface.co/meta-llama/Meta-Llama-3-8B). Ensure that the model falls within the task of `text-generation`</li> <li> `system_prompt` (`str`): Default System Prompt for LLM (note: this is overridden if a system prompt is provided by the user in the generation process). Defaults to an empty string. </li> <li> `type` (int or LLMType): Type of LLM to discriminate. Integer values should fall between 0 and 2 to signify the corresponding `LLMType` value. This is overridden by the `instruct` field in `HF_LLM.generate`. By default, this value is `LLMType.NEITHER`, which means that the user needs to specify during the `HF_LLM.generate` function or use the specific functions indented for use.</li></ul> |
| `HF_LLM.chat(text, max_new_tokens = 256, temperature = 0.0) -> str` | Uses a chat format (provided by the tokenizer) to get the LLM to complete a chat discussion) |<ul> <li>`text` (`Messages`): Input in either string or list format to generate LLM data. (See the above Input Types subsection for more info regarding the `Messages` type). If a system prompt is specified at the start, it is used in place of the previously specified System Prompt.</li> <li> `max_new_tokens` (`int`): Maximum tokens to be generated by the LLM. Per LLM, there is a different range of values for this variable. Defaults to 256. </li> <li> `temperature` (`float`): Temperature of LLM being queried. This variable is highly dependent on the actual LLM. Defaults to 0. </li> </ul> | 
| `HF_LLM.complete(text, max_new_tokens = 256, temperature = 0.0) -> str` | Uses LLM as a next-token predictor to generate a completion of a piece of text | <ul> <li>`text` (`str`): Input in **only** string format to generate LLM data. Unlike chat completion, this does not support a chat format as an input.</li> <li> `max_new_tokens` (`int`): Maximum tokens to be generated by the LLM. Per LLM, there is a different range of values for this variable. Defaults to 256. </li> <li> `temperature` (`float`): Temperature of LLM being queried. This variable is highly dependent on the actual LLM. Defaults to 0. </li> </ul> |
| `HF_LLM.generate(text, max_new_tokens = 256, temperature = 0.0, instruct = None) -> str` | Merges the `chat` and `complete` methods into a single method to simplify accessing the generation defaults. |<ul> <li>`text` (`Messages`): Input in either string or list format to generate LLM data. (See the above Input Types subsection for more info regarding the `Messages` type). If this is indeed a completion, any list input will throw a `ValueError`. If a system prompt is specified at the start, it is used in place of the previously specified System Prompt.</li> <li> `max_new_tokens` (`int`): Maximum tokens to be generated by the LLM. Per LLM, there is a different range of values for this variable. Defaults to 256. </li> <li> `temperature` (`float`): Temperature of LLM being queried. This variable is highly dependent on the actual LLM. Defaults to 0. </li> <li> `instruct` (`bool or None`): Optional flag to change behaviour of `generate` command. This overrides the input `type` parameter at instantiation. </li></ul> |

#### Other API Support

WalledEval also currently supports the following alternative LLM types:


| Class                                 | LLM Type                                                                         |
| --------------------------------------- | ---------------------------------------------------------------------------------- |
| `Claude(model_id, api_key, system_prompt = "", type = LLMType.NEITHER)` | Claude 3 (`Claude.haiku`, `Claude.sonnet` and `Claude.opus` class methods exist to initiate the most recent versions of each of these models)                                                                    |

#### Supporting your own LLMs

Assuming one cannot put up their models on HuggingFace Hub, they can also define their own classes to add support for their own LLMs using the abstract `llm.LLM` class. To extend this class, one needs to define an `__init___` method to override the default `LLM(name: str, system_prompt: str = "", type: int or LLMType)` method. One also needs to implement the abstract methods `complete(text: str, max_new_tokens: int = 256, temperature: float = 0.0) -> str` and `chat(text: Messages, max_new_tokens: int = 256, temperature: float = 0.0) -> str`.



### Judges (`walledeval.judge`)

Judges are used to identify if outputs are malignant. We currently support the judge `ClaudeJudge`, which uses Claude 3 Opus and a custom-defined taxonomy to test malignant outputs. It returns `False` if malignant (i.e. it didn't pass the test).

Usage is as follows:

```python
>>> from walledeval.judge import ClaudeJudge

>>> judge = ClaudeJudge("INSERT_API_KEY")
>>> judge.check("<insert output>")
# <boolean output>
```

A custom abstract `judge.Judge` class is also defined to support other possible judges, which takes in the judge identifier `name`, and an abstract method `check(text: str) -> bool`.

### Benchmarks (`walledeval.benchmark`)

Benchmarks are available to provide datasets to test both the LLM and Judges. We currently test the following benchmarks:


| Benchmark Name                         | Class  |
| ---------------------------------------- | -------- |
| [WMDP Benchmark](https://www.wmdp.ai/) | `WMDP` |

Usage is as follows:

```python
>>> from walledeval.benchmark import WMDP

>>> wmdp = WMDP()

>>> wmdp.test(llm, judge)
# <logs>
# generator[logs]
```

A custom abstract `benchmark.Benchmark` class is also defined for you to define your own benchmarks. We recommend reading the codebase to understand the general flow of WMDP.

<br><br>

<div style="border: 1px solid black; padding: 10px; display: inline-block;">
  <p align="center">
    <img width="350" alt="walleai_logo_shield" src="https://github.com/walledai/walledeval/assets/32847115/d8b1d14f-7071-448b-8997-2eeba4c2c8f6">
  </p>
</div>
