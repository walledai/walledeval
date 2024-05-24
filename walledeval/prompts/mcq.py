# walledeval/prompts/mcq.py

from walledeval.constants import (
    DEFAULT_OPTIONS,
    DEFAULT_SAMPLE_QUESTION,
    DEFAULT_OPTION_FORMAT
)
from walledeval.prompts.core import PromptTemplate
from walledeval.types import MultipleChoiceQuestion

__all__ = [
    "MultipleChoiceTemplate",
    "FewShotMCQTemplate",
]

_INSPECT_TEMPLATE = """
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: $$LETTER' (without quotes) where LETTER is one of $options.

$question

$choices

ANSWER:""".strip()+" "

_DEFAULT_TEMPLATE = """
Answer the following multiple choice question. The entire content of your response should be confined to the option. Choose from $options.

$question

$choices

Answer:""".strip()+" "


class MultipleChoiceTemplate(PromptTemplate):
    """Multiple Choice Prompt Template.

    Attributes:
        template (str): A string with $-delimited substitutions.
        MUST contain $question and $choices.
        options (list[str]): MCQ options to input to model.
        option_format (str): Format string for these options.
    """

    def __init__(self,
                 template: str,
                 options: list[str] = DEFAULT_OPTIONS,
                 option_format: str = DEFAULT_OPTION_FORMAT):
        """Inits MultipleChoiceTemplate.

        Args:
            template (str): A string with $-delimited substitutions.
            MUST contain $question, $options and $choices.
            options (list[str], optional): MCQ options to input to model.
            Defaults to ["A", "B", "C", "D", ...].
            option_format (str, optional): Format string for these options.
            Defaults to "{}. ".
        """

        super().__init__(template)

        self.options = options
        self.option_format = option_format

    @classmethod
    def default(cls,
                options: list[str] = DEFAULT_OPTIONS,
                option_format: str = DEFAULT_OPTION_FORMAT):
        """Default Template for Multiple Choice.

        Args:
            options (list[str], optional): MCQ options to input to model.
            Defaults to ["A", "B", "C", "D", ...].
            option_format (str, optional): Format string for these options.
            Defaults to "{}. ".

        Returns:
            _type_: A MultipleChoiceTemplate object
        """
        return cls(
            _DEFAULT_TEMPLATE,
            options=options,
            option_format=option_format
        )

    @classmethod
    def inspect(cls,
                options: list[str] = DEFAULT_OPTIONS,
                option_format: str = DEFAULT_OPTION_FORMAT):
        """Inspect Template for Multiple Choice.

        Sourced from [here](https://github.com/UKGovernmentBEIS/inspect_ai/blob/main/src/inspect_ai/solver/_multiple_choice.py#L18).

        Args:
            options (list[str], optional): MCQ options to input to model.
            Defaults to ["A", "B", "C", "D", ...].
            option_format (str, optional): Format string for these options.
            Defaults to "{}. ".

        Returns:
            _type_: A MultipleChoiceTemplate object
        """
        return cls(
            _INSPECT_TEMPLATE,
            options=options,
            option_format=option_format
        )

    def format(self,
               question: MultipleChoiceQuestion,
               **kwds) -> str:
        """Format Multiple Choice Template with given expected inputs and
        keywords.

        Args:
            question (MultipleChoiceQuestion): MultipleChoiceQuestion data.

        Returns:
            str: Formatted (or partially formatted) template.
        """
        # Conjoin Choices and Options to make final choices section
        choices_str = "\n".join([
            self.option_format.format(option) + choice
            for option, choice in zip(self.options, question.choices)
        ])

        return super().format(
            question=question.question,
            choices=choices_str,
            options=self.options[:len(question.choices)],
            **kwds
        )


_DEFAULT_QUESTION_TEMPLATE = """
$question

$choices

Choose from $options.

Answer:""".strip()+" "

_DEFAULT_PRECEDING_INSTRUCTION = """
Answer the following multiple choice question. The entire content of your response should be confined to the options indicated.
"""


class FewShotMCQTemplate(MultipleChoiceTemplate):
    """Few-Shot Multiple Choice Prompting Template.

    Attributes:
        template (str): A string with $-delimited substitutions.
        MUST contain $question, $options and $choices.
        options (list[str]): MCQ options to input to model.
        option_format (str): Format string for these options.
        sample_questions: (list[MultipleChoiceQuestion]) for
        sample (str): Instructions and sample qns preceding template.
    """

    def __init__(self,
                 template: str,
                 options: list[str] = DEFAULT_OPTIONS,
                 option_format: str = DEFAULT_OPTION_FORMAT,
                 samples: list[MultipleChoiceQuestion] = [DEFAULT_SAMPLE_QUESTION],
                 preceding_instruction: str = "",
                 boxed_answer: bool = False,
                 **sample_kwds):
        """Inits FewShotMCQTemplate.

        Args:
            template (str): A string with $-delimited substitutions.
            MUST contain $question and $choices.
            options (list[str], optional): MCQ options to input to model.
            Defaults to ["A", "B", "C", "D", ...].
            option_format (str, optional): Format string for these options.
            Defaults to "{}. ".
            samples (list[MultipleChoiceQuestion], optional):
                List of sample questions.
            preceding_instruction (str, optional): Preceding Instruction.
            Defaults to "".
            boxed_answer (bool, optional): Whether or not to box up the sample.
            answers.
        """

        super().__init__(template, options, option_format)

        self.sample_questions = samples

        self.boxed_answer = boxed_answer

        self.sample = ("\n\n".join([preceding_instruction] + [
            MultipleChoiceTemplate.format(
                self,
                question=sample,
                **sample_kwds
            ) + (
                "\\boxed{"+self.options[sample.answer]+"}"
                if boxed_answer else
                self.options[sample.answer]
            )
            for sample in samples
        ])).lstrip()

    @classmethod
    def default(cls,
                options: list[str] = DEFAULT_OPTIONS,
                option_format: str = DEFAULT_OPTION_FORMAT,
                samples: list[MultipleChoiceQuestion] = [DEFAULT_SAMPLE_QUESTION],
                boxed_answer: bool = False):
        return cls(
            _DEFAULT_QUESTION_TEMPLATE,
            options, option_format, samples,
            _DEFAULT_PRECEDING_INSTRUCTION,
            boxed_answer
        )

    def format(self,
               question: MultipleChoiceQuestion,
               **kwds) -> str:
        """Format Few-Shot Multiple Choice Template with given expected inputs.
        and keywords.

        Args:
            question (MultipleChoiceQuestion): MultipleChoiceQuestion data.

        Returns:
            str: Formatted (or partially formatted) template.
        """
        return (
            self.sample + "\n\n" +
            super().format(question, **kwds) +
            ("\\boxed{" if self.boxed_answer else "")
        )
