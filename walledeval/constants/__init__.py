# walledeval/constants/__init__.py

from walledeval.types import MultipleChoiceQuestion

DEFAULT_OPTIONS = [chr(idx) for idx in range(65, 91)]

DEFAULT_OPTION_FORMAT = "{}. "

DEFAULT_SAMPLE_QUESTION = MultipleChoiceQuestion(
    question="Which of the following is a fruit?",
    choices=["Spider", "Apple", "Lamp", "Cloud"],
    answer=1
)
