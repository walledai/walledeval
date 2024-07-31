# walledeval/judge/__init__.py

import warnings

from walledeval.judge.core import Judge
from walledeval.judge.lionguard import LionGuardJudge
from walledeval.judge.mcq import MCQJudge
from walledeval.judge.llm import (
    LLMasaJudge,
    QuestionLLMasaJudge,
    MultiClassToxicityJudge,
    LLMGuardJudge, LLMGuardOutput,
    LlamaGuardJudge
)
from walledeval.judge.string import StringMatchingJudge
from walledeval.judge.toxicity import (
    ToxicityModelJudge
)
from walledeval.judge.huggingface import (
    HFTextClassificationJudge,
    GPTFuzzJudge, UnitaryJudge,
    RobertaToxicityJudge,
    PromptGuardJudge
)

# Windows does not support CodeShield so we use this as a bypass
try:
    from walledeval.judge.code import CodeShieldJudge
except ImportError:
    warnings.warn("CodeShieldJudge could not be imported, not supported on Windows OS", ImportWarning, stacklevel=2)
except OSError:
    warnings.warn("CodeShieldJudge could not be imported, not supported on Windows OS", ImportWarning, stacklevel=2)

__all__ = [
    "Judge",
    "MCQJudge",
    "LLMasaJudge",
    "QuestionLLMasaJudge",
    "MultiClassToxicityJudge",
    "LLMGuardJudge", "LLMGuardOutput",
    "LlamaGuardJudge",
    "ToxicityModelJudge", "LionGuardJudge",
    "StringMatchingJudge",
    "HFTextClassificationJudge",
    "GPTFuzzJudge", "UnitaryJudge",
    "RobertaToxicityJudge",
    "PromptGuardJudge"
]

if "CodeShieldJudge" in globals():
    __all__.append("CodeShieldJudge")