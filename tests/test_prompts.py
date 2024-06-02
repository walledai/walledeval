from walledeval.constants import DEFAULT_SAMPLE_QUESTION
from walledeval.prompts import MultipleChoiceTemplate, FewShotMCQTemplate


def test_mct():
    template = MultipleChoiceTemplate.default()

    fmt = template.format(DEFAULT_SAMPLE_QUESTION)

    ground_truth = (
        "Answer the following multiple choice question. "
        "The entire content of your response should be "
        "confined to the option. Choose from ['A', 'B', 'C', 'D'].\n\n"
        "Which of the following is a fruit?\n\nA. Spider\n"
        "B. Apple\nC. Lamp\nD. Cloud\n\nAnswer: "
    )

    assert fmt == ground_truth


def test_fs_0_unboxed():
    fsmcq0 = FewShotMCQTemplate.default(
        samples=[],
        boxed_answer=False
    )

    fmt = fsmcq0.format(DEFAULT_SAMPLE_QUESTION)

    ground_truth = (
        "Answer the following multiple choice question. "
        "The entire content of your response should be "
        "confined to the options indicated.\n\n\nWhich "
        "of the following is a fruit?\n\nA. Spider\n"
        "B. Apple\nC. Lamp\nD. Cloud\n\nChoose from "
        "['A', 'B', 'C', 'D'].\n\nAnswer: "
    )

    assert fmt == ground_truth


def test_fs_0_boxed():
    fsmcq0 = FewShotMCQTemplate.default(
        samples=[],
        boxed_answer=True
    )

    fmt = fsmcq0.format(DEFAULT_SAMPLE_QUESTION)

    ground_truth = (
        "Answer the following multiple choice question. "
        "The entire content of your response should be "
        "confined to the options indicated.\n\n\nWhich "
        "of the following is a fruit?\n\nA. Spider\n"
        "B. Apple\nC. Lamp\nD. Cloud\n\nChoose from "
        "['A', 'B', 'C', 'D'].\n\nAnswer: \\boxed{"
    )

    assert fmt == ground_truth


def test_fs_1_unboxed():
    fsmcq1 = FewShotMCQTemplate.default(
        samples=[DEFAULT_SAMPLE_QUESTION],
        boxed_answer=False
    )

    fmt = fsmcq1.format(DEFAULT_SAMPLE_QUESTION)

    ground_truth = (
        "Answer the following multiple choice question. "
        "The entire content of your response should be "
        "confined to the options indicated.\n\n\nWhich "
        "of the following is a fruit?\n\nA. Spider\n"
        "B. Apple\nC. Lamp\nD. Cloud\n\nChoose from "
        "['A', 'B', 'C', 'D'].\n\nAnswer: B\n\nWhich "
        "of the following is a fruit?\n\nA. Spider\n"
        "B. Apple\nC. Lamp\nD. Cloud\n\nChoose from "
        "['A', 'B', 'C', 'D'].\n\nAnswer: "
    )

    assert fmt == ground_truth


def test_fs_1_boxed():
    fsmcq1 = FewShotMCQTemplate.default(
        samples=[DEFAULT_SAMPLE_QUESTION],
        boxed_answer=True
    )

    fmt = fsmcq1.format(DEFAULT_SAMPLE_QUESTION)

    ground_truth = (
        "Answer the following multiple choice question. "
        "The entire content of your response should be "
        "confined to the options indicated.\n\n\nWhich "
        "of the following is a fruit?\n\nA. Spider\n"
        "B. Apple\nC. Lamp\nD. Cloud\n\nChoose from "
        "['A', 'B', 'C', 'D'].\n\nAnswer: \\boxed{B}\n\nWhich "
        "of the following is a fruit?\n\nA. Spider\n"
        "B. Apple\nC. Lamp\nD. Cloud\n\nChoose from "
        "['A', 'B', 'C', 'D'].\n\nAnswer: \\boxed{"
    )

    assert fmt == ground_truth
