# walledeval/judge/code.py

from codeshield.cs import CodeShield, CodeShieldScanResult

from walledeval.judge.core import Judge

__all__ = [
    "CodeShieldJudge"
]


class CodeShieldJudge(Judge[None, CodeShieldScanResult]):
    def __init__(self):
        super().__init__("CodeShield")
    
    async def check(self, response: str, answer: None = None) -> CodeShieldScanResult:
        result = await CodeShield.scan_code(response)
        return result