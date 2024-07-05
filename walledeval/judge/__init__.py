# walledeval/judge/__init__.py

import warnings

from walledeval.judge.core import Judge
from walledeval.judge.lionguard import LionGuardJudge
from walledeval.judge.mcq import MCQJudge
from walledeval.judge.llm import (
    LLMasaJudge,
    QuestionLLMasaJudge, SystemLLMasaJudge,
    MultiClassToxicityJudge,
    LlamaGuardJudge, LlamaGuardOutput
)
from walledeval.judge.string import StringMatchingJudge
from walledeval.judge.toxicity import (
    ToxicityModelJudge
)

# Windows does not support CodeShield so we use this as a bypass
try:
    from walledeval.judge.code import CodeShieldJudge
except ImportError:
    warnings.warn("CodeShieldJudge could not be imported, not supported on Windows OS", ImportWarning, stacklevel=2)

__all__ = [
    "Judge",
    "MCQJudge",
    "LLMasaJudge",
    "QuestionLLMasaJudge", "SystemLLMasaJudge",
    "MultiClassToxicityJudge",
    "LlamaGuardJudge", "LlamaGuardOutput",
    "ToxicityModelJudge", "LionGuardJudge",
    "StringMatchingJudge"
]

if "CodeShieldJudge" in globals():
    __all__.append("CodeShieldJudge")
