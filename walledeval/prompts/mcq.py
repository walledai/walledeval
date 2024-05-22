# walledeval/prompts/mcq.py

from walledeval.prompts.core import PromptTemplate
from walledeval.types import MultipleChoiceQuestion

__all__ = [
    "MultipleChoiceTemplate",
    "FewShotMCQTemplate",
    "DEFAULT_OPTIONS", "DEFAULT_SAMPLE_QUESTION"
    
]

DEFAULT_OPTIONS = [chr(idx) for idx in range(65, 91)]

DEFAULT_SAMPLE_QUESTION = MultipleChoiceQuestion(
    question = "Which of the following is a fruit?",
    choices = ["Spider", "Apple", "Lamp", "Cloud"],
    answer = 1
)

_INSPECT_TEMPLATE = """
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}.

$question

$choices

ANSWER:""".strip()+" "

class MultipleChoiceTemplate(PromptTemplate):
    """Multiple Choice Prompt Template.

    Attributes:
        template (str): A string with $-delimited substitutions. MUST contain $question and $choices.
        options (list[str]): MCQ options to input to model.
        option_format (str): Format string for these options.
    """
    
    def __init__(self, 
                 template: str, 
                 options: list[str] = DEFAULT_OPTIONS,
                 option_format: str = "{}. "):
        """Inits MultipleChoiceTemplate.

        Args:
            template (str): A string with $-delimited substitutions. MUST contain $question and $choices.
            options (list[str], optional): MCQ options to input to model. Defaults to ["A", "B", "C", "D", ...].
            option_format (str, optional): Format string for these options. Defaults to "{}. ".
        """

        super().__init__(template)
        
        self.options = options
        self.option_format = option_format
    
    @classmethod
    def inspect_method(cls, 
                       options: list[str] = DEFAULT_OPTIONS,
                       option_format: str = "{}. "):
        """Inspect Template for Multiple Choice.
        
        Sourced from [here](https://github.com/UKGovernmentBEIS/inspect_ai/blob/main/src/inspect_ai/solver/_multiple_choice.py#L18).

        Args:
            options (list[str], optional): MCQ options to input to model. Defaults to ["A", "B", "C", "D", ...].
            option_format (str, optional): Format string for these options. Defaults to "{}. ".

        Returns:
            _type_: A MultipleChoiceTemplate object
        """
        return cls(
            _INSPECT_TEMPLATE,
            options = options,
            option_format = option_format
        )
    
    def format(self,
               question: MultipleChoiceQuestion,
               **kwds) -> str:
        """Format Multiple Choice Template with given expected inputs and keywords.

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
            question = question.question,
            choices = choices_str,
            **kwds
        )



class FewShotMCQTemplate(MultipleChoiceTemplate):
    """Few-Shot Multiple Choice Prompting Template.

    Attributes:
        template (str): A string with $-delimited substitutions. MUST contain $question and $choices.
        options (list[str]): MCQ options to input to model.
        option_format (str): Format string for these options.
        sample_questions: (list[MultipleChoiceQuestion]) for 
        sample (str): Instructions and sample qns preceding template.
    """
    def __init__(self,
                 template: str,
                 options: list[str] = DEFAULT_OPTIONS,
                 option_format: str = "{}.",
                 samples: list[MultipleChoiceQuestion] = [DEFAULT_SAMPLE_QUESTION],
                 preceding_instruction: str = "",
                 **sample_kwds):
        """Inits FewShotMCQTemplate.

        Args:
            template (str): A string with $-delimited substitutions. MUST contain $question and $choices.
            options (list[str], optional): MCQ options to input to model. Defaults to ["A", "B", "C", "D", ...].
            option_format (str, optional): Format string for these options. Defaults to "{}. ".
            samples (list[MultipleChoiceQuestion], optional): List of sample questions. Defaults to [ MultipleChoiceQuestion( question = "Which of the following is a fruit?", choices = ["Spider", "Apple", "Lamp", "Cloud"], answer = 1 ) ].
            preceding_instruction (str, optional): Preceding Instruction, if need be. Defaults to "".
        """
        
        super().__init__(template, options, option_format)
        
        self.sample_questions = samples
        
        self.sample = ("\n\n".join([preceding_instruction] + [
            super().format(
                question = sample.question,
                choices = sample.choices,
                **sample_kwds
            ) + self.options[sample.answer]
            for sample in samples
        ])).lstrip()
    
    def format(self,
               question: str,
               choices: list[str],
               **kwds) -> str:
        """Format Few-Shot Multiple Choice Template with given expected inputs and keywords.

        Args:
            question (MultipleChoiceQuestion): MultipleChoiceQuestion data.

        Returns:
            str: Formatted (or partially formatted) template.
        """
        return (
            self.sample + "\n\n" + 
            super().format(question, choices, **kwds)
        )