# still in the midst of debugging this


import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoModelForCausalLM

def calculate_perplexity(model, tokenizer, text):
    """
    Calculate the perplexity of the generated text.

    Args:
        model: The language model.
        tokenizer: The tokenizer associated with the model.
        text: The text for which to calculate perplexity.

    Returns:
        Perplexity score.
    """
    with torch.no_grad():
        encodings = tokenizer(text, return_tensors="pt")
        max_length = model.config.n_positions
        stride = 512
        lls = []

        for i in range(0, encodings.input_ids.size(1), stride):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, encodings.input_ids.size(1))
            trg_len = end_loc - i
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs.loss * trg_len

            lls.append(log_likelihood)

        ppl = torch.exp(torch.stack(lls).sum() / end_loc).item()
        return ppl

def evaluate_toxicity(text):
    """
    Evaluate the toxicity of the generated text using a pre-trained toxicity model.

    Args:
        text: The text to evaluate for toxicity.

    Returns:
        Toxicity score.
    """
    tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert")
    model = AutoModelForSequenceClassification.from_pretrained("unitary/toxic-bert")

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    scores = torch.softmax(outputs.logits, dim=1)
    toxicity_score = scores[0][1].item()

    return toxicity_score
