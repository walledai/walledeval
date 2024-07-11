# walledeval/prompts/core.py

from string import Template

from walledeval.dtypes import Question, Prompt

__all__ = ["BasePromptTemplate"]


class BasePromptTemplate(Template):
    """Basic Prompt Template Definition.

    Attributes:
        template (str): A string with $-delimited substitutions.
    """

    def __init__(self, template: str):
        """Inits BasePromptTemplate.

        Args:
            template (str): A string with $-delimited substitutions.
        """
        super().__init__(template)

    def format(self, **kwds) -> str:
        """Formats Prompt Template with Keywords specified.

        Returns:
            str: Formatted (or partially formatted) template.
        """
        return self.safe_substitute(**kwds)


class PromptTemplate(Template):
    """Prompt Template Definition.

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


class QuestionTemplate(Template):
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


