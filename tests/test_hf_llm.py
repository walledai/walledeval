# walledeval/tests/test_hf_llm.py

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from walledeval.llm.huggingface import HF_LLM
from walledeval.types.message import Message
from walledeval.attacks.gcg import gcg_attack
from walledeval.attacks.wildteaming import WT
from walledeval.evaluation.metrics import Evaluator

# Example model to attack
model = HF_LLM(id='distilgpt2')

test_prompt = "The quick brown fox jumps over the lazy dog."

# GCG attack
mutated_text = gcg_attack(model_name='distilgpt2', text=test_prompt, max_iters=50, top_k=5)
print("GCG Mutated Text:", mutated_text)

# Generate output from the original model for comparison
original_output = model.generate(test_prompt)
print("Original Output:", original_output)

# Create messages for chat
messages = [
    Message(role="user", content="How are you today?")
]

# Generate chat output from the original model for comparison
chat_output = model.chat(messages)
print("Chat Output:", chat_output)

# Evaluate GCG attack
evaluator = Evaluator(model_name='distilgpt2')
gcg_results = evaluator.evaluate(test_prompt, mutated_text)
print("Evaluation Results for GCG attack:")
print(f"Cosine Similarity: {gcg_results['similarity']}")
print(f"Change Ratio: {gcg_results['change_ratio']}")
print(f"Original Perplexity: {gcg_results['original_perplexity']}")
print(f"Mutated Perplexity: {gcg_results['mutated_perplexity']}")
print(f"Original Toxicity: {gcg_results['original_toxicity']}")
print(f"Mutated Toxicity: {gcg_results['mutated_toxicity']}")

# Wild Teaming attack
wt = WT(model_name='distilgpt2')
wt_mutated_text = wt.compose_attack(test_prompt)
print("WT Mutated Text:", wt_mutated_text)

# Evaluate WT attack
wt_evaluation_results = wt.evaluate(test_prompt, wt_mutated_text)
print("Evaluation Results for WT attack:")
print(f"Cosine Similarity: {wt_evaluation_results['similarity']}")
print(f"Change Ratio: {wt_evaluation_results['change_ratio']}")
print(f"Original Perplexity: {wt_evaluation_results['original_perplexity']}")
print(f"Mutated Perplexity: {wt_evaluation_results['mutated_perplexity']}")
print(f"Original Toxicity: {wt_evaluation_results['original_toxicity']}")
print(f"Mutated Toxicity: {wt_evaluation_results['mutated_toxicity']}")
print(f"Is Attack Effective: {wt_evaluation_results['is_effective']}")
