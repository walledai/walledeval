# walledeval/attacks/mutators/cipher.py

from abc import abstractmethod
from collections.abc import Iterable

import inflect
import base64

from walledeval.prompts import PromptTemplate
from walledeval.attacks.mutators.core import Mutator

p = inflect.engine()

class CipherMutator(Mutator):
    def __init__(self, name: str, template: PromptTemplate, samples: Iterable[str] = []):
        super().__init__(name)
        self.template = template
        self.samples = list(samples)
    
    @abstractmethod
    def encode(self, x: str) -> str:
        raise NotImplementedError
    
    @abstractmethod
    def decode(self, y: str) -> str:
        raise NotImplementedError
    
    def mutate(self, prompt: str, **kwargs) -> str:
        examples = ""
        for i, sample in enumerate(self.samples):
            examples += f"Example {i}:\n{self.encode(sample)}\n\n"
        
        return self.template.format(
            prompt = self.encode(prompt),
            examples = examples.strip(),
            **kwargs
        )


class CaesarMutator(CipherMutator):
    def __init__(self, shift: int, samples: Iterable[str] = []):
        super().__init__(
            "CaesarMutator",
            template = PromptTemplate.from_preset("mutations/cipher/caesar"),
            samples = samples
        )
        self.shift = shift % 26
    
    def mutate(self, prompt: str, **kwargs) -> str:
        return super().mutate(prompt,
                              shift = p.number_to_words(self.shift),
                              **kwargs)
    
    def shift(self, char: str, shift: int) -> str:
        if 'a' <= char <= 'z':
                ans += chr(ord('a') + (ord(char) - ord('a') + shift) % 26)
        elif 'A' <= char <= 'Z':
            ans += chr(ord('A') + (ord(char) - ord('A') + shift) % 26)
        else:
            ans += p
    
    def encode(self, x: str) -> str:
        return "".join([self.shift(char, self.shift) for char in x])
    
    def decode(self, y: str) -> str:
        return "".join([self.shift(char, -self.shift) for char in y])


class Base64Mutator(CipherMutator):
    def __init__(self, samples: Iterable[str] = []):
        super().__init__(
            "Base64Mutator",
            template = PromptTemplate.from_preset("mutations/cipher/base64"),
            samples = samples
        )
    
    def encode(self, x: str) -> str:
        return base64.b64encode(x.encode()).decode()
    
    def decode(self, y: str) -> str:
        return base64.b64decode(y.encode()).decode()


class AsciiMutator(CipherMutator):
    def __init__(self, samples: Iterable[str] = []):
        super().__init__(
            "AsciiMutator",
            template = PromptTemplate.from_preset("mutations/cipher/ascii"),
            samples = samples
        )
    
    def encode(self, x: str) -> str:
        return "".join([x if x == "\n" else str(ord(x)) for i in x])
    
    def decode(self, y: str) -> str:
        return "\n".join(["".join([chr(int(yij)) for yij in yi.split()]) for yi in y.split("\n")])
    
class SelfDefineMutator(CipherMutator):
    chinese_alphabet = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸", "子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥", "天", "地", "人", "黄"]
    english_alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
    chinese_alphabet_shifted = ["e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "a", "b", "c", "d"]

    def __init__(self, samples: Iterable[str]):
        super().__init__("SelfDefineMutator", samples=samples)

    def encode(self, s: str) -> str:
        s = s.lower()
        ans = ""
        for letter in s:
            try:
                ans += self.chinese_alphabet_shifted[ord(letter.lower()) - 97]
            except:
                ans += letter
        return ans

    def decode(self, s: str) -> str:
        ans = ""
        for letter in s:
            try:
                position = self.chinese_alphabet_shifted.index(letter)
                ans += self.english_alphabet[position]
            except:
                ans += letter
        return ans
    
class UnicodeMutator(CipherMutator):
    def __init__(self, samples: Iterable[str]):
        super().__init__("UnicodeMutator", samples=samples)

    def encode(self, s: str) -> str:
        ans = ''
        for c in s:
            byte_s = str(c.encode("unicode_escape"))
            if len(byte_s) > 8:
                ans += byte_s[3:-1]
            else:
                ans += byte_s[-2]
        return ans

    def decode(self, s: str) -> str:
        return bytes(s, encoding="utf8").decode("unicode_escape")
    
class UTF8Mutator(CipherMutator):
    def __init__(self, samples: Iterable[str]):
        super().__init__("UTF8Mutator", samples=samples)

    def encode(self, s: str) -> str:
        ans = ''
        for c in s:
            byte_s = str(c.encode("utf8"))
            if len(byte_s) > 8:
                ans += byte_s[2:-1]
            else:
                ans += byte_s[-2]
        return ans

    def decode(self, s: str) -> str:
        ans = b''
        while len(s):
            if s.startswith("\\x"):
                ans += bytes.fromhex(s[2:4])
                s = s[4:]
            else:
                ans += bytes(s[0], encoding="utf8")
                s = s[1:]
        return ans.decode("utf8")
    
class MorseMutator(CipherMutator):
    MORSE_CODE_DICT = {'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.', 'G': '--.', 'H': '....',
                       'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---', 'P': '.--.',
                       'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-',
                       'Y': '-.--', 'Z': '--..', '1': '.----', '2': '..---', '3': '...--', '4': '....-', '5': '.....',
                       '6': '-....', '7': '--...', '8': '---..', '9': '----.', '0': '-----', ', ': '--..--', '.': '.-.-.-',
                       '?': '..--..', '/': '-..-.', '-': '-....-', '(': '-.--.', ')': '-.--.-'}
    
    def __init__(self, samples: Iterable[str]):
        super().__init__("MorseMutator", samples=samples)

    def encode(self, s: str) -> str:
        s = s.upper()
        return ' '.join([self.MORSE_CODE_DICT.get(char, char) for char in s])

    def decode(self, s: str) -> str:
        reverse_dict = {v: k for k, v in self.MORSE_CODE_DICT.items()}
        return ''.join([reverse_dict.get(char, char) for char in s.split(' ')])

class GBKMutator(CipherMutator):
    def __init__(self, samples: Iterable[str]):
        super().__init__("GBKMutator", samples=samples)

    def encode(self, s: str) -> str:
        ans = ''
        for c in s:
            byte_s = str(c.encode("GBK"))
            if len(byte_s) > 8:
                ans += byte_s[2:-1]
            else:
                ans += byte_s[-2]
        return ans

    def decode(self, s: str) -> str:
        ans = b''
        while len(s):
            if s.startswith("\\x"):
                ans += bytes.fromhex(s[2:4])
                s = s[4:]
            else:
                ans += bytes(s[0], encoding="GBK")
                s = s[1:]
        return ans.decode("GBK")
    
    
