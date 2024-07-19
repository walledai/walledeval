# walledeval/evaluation/metrics.py

from .implement import CosSim, ChangeRatio, Perplexity, Toxicity

class Evaluator:
    def __init__(self, model_name):
        self.cos_sim = CosSim(model_name)
        self.change_ratio = ChangeRatio(model_name)
        self.perplexity = Perplexity(model_name)
        self.toxicity = Toxicity()

    def evaluate(self, original_text: str, mutated_text: str) -> dict:
        similarity = self.cos_sim.evaluate(original_text, mutated_text)
        change_ratio = self.change_ratio.evaluate(original_text, mutated_text)
        original_perplexity = self.perplexity.evaluate(original_text)
        mutated_perplexity = self.perplexity.evaluate(mutated_text)
        original_toxicity = self.toxicity.evaluate(original_text)
        mutated_toxicity = self.toxicity.evaluate(mutated_text)

        return {
            "similarity": similarity,
            "change_ratio": change_ratio,
            "original_perplexity": original_perplexity,
            "mutated_perplexity": mutated_perplexity,
            "original_toxicity": original_toxicity,
            "mutated_toxicity": mutated_toxicity
        }