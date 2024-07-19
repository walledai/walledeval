# # walledeval/tests/test_hf_llm.py

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from walledeval.llm.huggingface import HF_LLM
from walledeval.types.message import Message
from walledeval.attacks.gcg import gcg_attack
from walledeval.evaluation.metrics import Evaluator

# Example model to attack
model = HF_LLM(id='distilgpt2')

test_prompt = "The quick brown fox jumps over the lazy dog."

# GCG attack
mutated_text = gcg_attack(model_name='distilgpt2', text=test_prompt, max_iters=50, top_k=5)
print("Mutated Text:", mutated_text)

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

# Evaluate of attack
evaluator = Evaluator(model_name='distilgpt2')
results = evaluator.evaluate(test_prompt, mutated_text)
print("Evaluation Results:")
print(f"Cosine Similarity: {results['similarity']}")
print(f"Change Ratio: {results['change_ratio']}")
print(f"Original Perplexity: {results['original_perplexity']}")
print(f"Mutated Perplexity: {results['mutated_perplexity']}")
print(f"Original Toxicity: {results['original_toxicity']}")
print(f"Mutated Toxicity: {results['mutated_toxicity']}")
