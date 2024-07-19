# walledeval/evaluation/implement.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from .core import Metric

# Cosine Similarity class: measure the semantic similarity of the original and mutated texts by comparing their embeddings.
# Low: greater difference between original and mutated text.
# High: Lesser difference between original and mutated text.
class CosSim(Metric):
    def __init__(self, model_name: str):
        super().__init__("CosineSimilarity")
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_embedding(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        with torch.no_grad():
            embeddings = self.model.get_input_embeddings()(inputs['input_ids'])
            mean_embedding = torch.mean(embeddings, dim=1)
        return mean_embedding.cpu().numpy()

    def evaluate(self, original_text: str, mutated_text: str) -> float:
        original_embedding = self.generate_embedding(original_text)
        mutated_embedding = self.generate_embedding(mutated_text)

        similarity = cosine_similarity(original_embedding, mutated_embedding)
        return similarity[0][0]

# ChangeRatio class: measure the degree of change in text i.t.o token alterations 
# Low: minimal changes
# High: significant changes
class ChangeRatio(Metric):
    def __init__(self, model_name: str):
        super().__init__("ChangeRatio")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')

    def evaluate(self, original_text: str, mutated_text: str) -> float:
        original_tokens = self.tokenizer.tokenize(original_text)
        mutated_tokens = self.tokenizer.tokenize(mutated_text)

        changes = sum(1 for orig, mut in zip(original_tokens, mutated_tokens) if orig != mut)
        change_ratio = changes / max(len(original_tokens), len(mutated_tokens))
        return change_ratio

# Perplexity Class: measures the predictability of text.
# lower: text is more predictable
# Higher: text is less unpredictable
class Perplexity(Metric):
    def __init__(self, model_name: str):
        super().__init__("Perplexity")
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def evaluate(self, text: str) -> float:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            perplexity = torch.exp(loss)
        return perplexity.item()

# Toxicity class: measures the potential offensiveness of text
# Low: safe
# High: harmful
class Toxicity(Metric):
    def __init__(self):
        super().__init__("Toxicity")
        self.tokenizer = AutoTokenizer.from_pretrained("nicholasKluge/ToxicityModel")
        self.model = AutoModelForSequenceClassification.from_pretrained("nicholasKluge/ToxicityModel")

    def evaluate(self, text: str) -> float:
        tokens = self.tokenizer(
            text, truncation=True, max_length=512,
            return_token_type_ids=False, return_tensors="pt",
            return_attention_mask=True
        )

        with torch.no_grad():
            outputs = self.model(**tokens)
            score = outputs.logits[0][0].item()
        return score
