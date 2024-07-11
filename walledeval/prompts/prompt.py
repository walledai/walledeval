# walledeval/prompts/prompt.py

from walledeval.dtypes import SystemAssistedPrompt
from walledeval.prompts.core import PromptTemplate

__all__ = ["SystemAssistedTemplate"]


class SystemAssistedTemplate(PromptTemplate):
    """System-Assisted Prompt Template Definition.

    Args:
        template (str): A string with $-delimited substitutions.
        Must contain $prompt.
    """
    def __init__(self, template: str = "$prompt"):
        """Inits SystemAssistedTemplate.

        Args:
            template (str, optional): A string with $-delimited substitutions.
            Must contain $prompt. Defaults to "$prompt".
        """
        super().__init__(template)

    def format(self,
               prompt: SystemAssistedPrompt,
               **kwds) -> str:
        """Format Prompt Template with given expected inputs and
        keywords.

        Args:
            prompt (Prompt): Prompt data.

        Returns:
            str: Formatted (or partially formatted) template.
        """

        return [
            {
                "role": "system",
                "content": prompt.system
            },
            {
                "role": "user",
                "content": super().format(
                    prompt=prompt.prompt,
                    **kwds
                )
            }
        ]
