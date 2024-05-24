# walledeval/prompts/core.py

from string import Template

__all__ = ["PromptTemplate"]


class PromptTemplate(Template):
    """Basic Prompt Template Definition.

    Attributes:
        template (str): A string with $-delimited substitutions.
    """

    def __init__(self, template: str):
        """Inits PromptTemplate.

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
