# walledeval/attacks/mutators/__init__.py

import warnings

from walledeval.attacks.mutators.core import Mutator, CompositeMutator
from walledeval.attacks.mutators.rule import (
    DisemvowelMutator, LeetSpeakMutator
)
from walledeval.attacks.mutators.cipher import (
    CipherMutator, CaesarMutator,
    Base64Mutator, AsciiMutator,
    SelfDefineMutator, UnicodeMutator,
    UTF8Mutator, GBKMutator,
    MorseMutator
)

from walledeval.attacks.mutators.generative import GenerativeMutator
from walledeval.attacks.mutators.artprompt import MaskingMutator, CloakingMutator
    
try:
    from walledeval.attacks.mutators.translation import TranslationMutator
except ImportError:
    warnings.warn("TranslationMutator could not be imported, googletrans package not supported", ImportWarning, stacklevel=2)
except OSError:
    warnings.warn("TranslationMutator could not be imported, googletrans package not supported", ImportWarning, stacklevel=2)



__all__ = [
    "Mutator", "CompositeMutator",
    "DisemvowelMutator", "LeetSpeakMutator",
    "CipherMutator", "CaesarMutator",
    "Base64Mutator", "AsciiMutator",
    "GenerativeMutator", "MaskingMutator",
    "CloakingMutator", "SelfDefineMutator", 
    "UnicodeMutator", "UTF8Mutator",
    "GBKMutator", "MorseMutator",
    #"TranslationMutator"
]


if "TranslationMutator" in globals():
    __all__.append("TranslationMutator")