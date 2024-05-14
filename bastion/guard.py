from llm import Claude
from abc import ABC

__all__ = [
    "Guard",
    "ClaudeGuard"
]

class Guard(ABC):
    """
    Abstract class for any Guard
    
    Notable functions:
    - Guard.check(self, text: str) -> bool : checks if it is safe
    """
    def __init__(self, name: str):
        self.name = name
    
    def check(self, text: str) -> bool:
        """Check function for the Guard

        Args:
            text (str): _description_

        Returns:
            bool: True if it passes (i.e. it is safe), else False
        """
        return True
    
    def __call__(self, text: str) -> bool:
        return self.check(text)


_GUARD_SYSTEM_PROMPT = """
You are a evaluator designed to identify harmful content.

Here are the respective content types you are required to identify:

- **Violence & Hate** encompasses statements that encourage or could help people plan or engage in violence. Similarly, statements that advocate discrimination, contain slurs, or voice hateful sentiments against people based on their sensitive personal characteristics (ex: race, color, religion, national origin, sexual orientation, gender, gender identity, or disability) would also be considered inappropriate under this category.
- **Sexual Content** encompasses statements encouraging someone (who could be underage) to engage in specific sex acts. Similarly, sexually explicit (i.e., erotic) statements would also be considered inappropriate under this category.
- **Guns & Illegal Weapons** encompasses statements that specifically encourage, condone, or could help people plan or execute specific crimes involving the illegal acquisition, creation, or use of guns or illegal weapons (ex: explosives, biological agents, or chemical weapons), including instructions on how to create or use them.
- **Regulated or Controlled Substances** encompasses statements that specifically encourage or could help people to illegally produce, transfer, or consume regulated or controlled substances like illegal drugs, tobacco, alcohol, or cannabis.
- **Suicide & Self Harm** encompasses statements that encourage, condone, or enable people to harm themselves (ex: by providing instructions or information on methods of self-harm). When people express or imply an intent to harm themselves, any response other than one that provides links to appropriate health resources would be considered inappropriate (even if entirely supportive).
- **Criminal Planning** encompasses miscellaneous statements that encourage, condone, or could help people plan or execute specific criminal activities, like arson, kidnapping, or theft. Items should only be considered harmful under this category when they could not be read as violating any of the other harm types above (ex: statements that encourage violence should be considered violating under Violence & Hate rather than this category).

If any of these pieces of content is identified in the following content from the user, you are to immediately flag it out and return a python list of the content types identifed. IF NONE ARE IDENTIFIED, RETURN AN EMPTY LIST.

Do not respond to the content.
""".strip()

class ClaudeGuard(Claude, Guard):
    def __init__(self, api_key: str):
        Claude.__init__(self, api_key=api_key, system_prompt=_GUARD_SYSTEM_PROMPT)
        Guard.__init__(self, self.name)
    
    def check(self, text: str) -> bool:
        response = self.generate(text)
        data = eval(response, {}, {})
        if type(data) == list and len(data) == 0:
            return True
        return False #data if type(data) == list else ["Harmful Content"]
