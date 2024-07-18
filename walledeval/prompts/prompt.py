# walledeval/prompts/prompt.py

from walledeval.types import Question, Prompt, SystemAssistedPrompt
from walledeval.prompts.core import BasePromptTemplate

__all__ = [
    "DefaultPromptTemplate", "QuestionTemplate",
    "SystemAssistedTemplate"
]


class DefaultPromptTemplate(BasePromptTemplate):
    """Default Prompt Template Definition.

    Args:
        template (str): A string with $-delimited substitutions.
        Must contain $prompt.
    """
    def __init__(self, template: str = "$prompt"):
        """Inits PromptTemplate.

        Args:
            template (str, optional): A string with $-delimited substitutions.
            Must contain $prompt. Defaults to "$prompt".
        """
        super().__init__(template)

    def format(self,
               prompt: Prompt,
               **kwds) -> str:
        """Format Prompt Template with given expected inputs and
        keywords.

        Args:
            prompt (Prompt): Prompt data.

        Returns:
            str: Formatted (or partially formatted) template.
        """

        return super().format(
            prompt=prompt.prompt,
            **kwds
        )


class QuestionTemplate(BasePromptTemplate):
    """Question-based Prompt Template Definition.

    Args:
        template (str): A string with $-delimited substitutions.
        Must contain $question.
    """
    def __init__(self, template: str = "$question"):
        """Inits QuestionTemplate.

        Args:
            template (str, optional): A string with $-delimited substitutions. 
            Defaults to "$question".
        """
        super().__init__(template)

    def format(self,
               question: Question,
               **kwds) -> str:
        """Format Question Template with given expected inputs and
        keywords.

        Args:
            question (Question): Question data.

        Returns:
            str: Formatted (or partially formatted) template.
        """

        return super().format(
            question=question.question,
            **kwds
        )


class SystemAssistedTemplate(BasePromptTemplate):
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