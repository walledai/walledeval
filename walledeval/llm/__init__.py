# walledeval/llm/__init__.py

import warnings

from walledeval.llm.core import LLM
from walledeval.llm.huggingface import hf_models, HF_LLM
from walledeval.llm.claude import Claude
from walledeval.llm.openai import OpenAI
from walledeval.llm.gemini import Gemini
from walledeval.llm.azure_openai import AzureOpenAI
from walledeval.llm.together import Together
from walledeval.llm.anyscale import Anyscale
from walledeval.llm.octoai import OctoAI
from walledeval.llm.groq import Groq

# We require users to install llama-cpp-python separately to use Llama
try:
    from walledeval.llm.llama import Llama
except (ImportError, OSError):
    warnings.warn("Llama could not be imported, library needs to be installed separately to use", ImportWarning, stacklevel=2)


__all__ = [
    "LLM",
    "hf_models", "HF_LLM",
    "Claude", "OpenAI",
    "Gemini",
    "AzureOpenAI",
    "Bedrock",
    "Together", "Anyscale",
    "OctoAI", "Groq",
]

if "Llama" in globals():
    __all__.append("Llama")