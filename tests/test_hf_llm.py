import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from walledeval.llm.huggingface import HF_LLM
from walledeval.types.message import Message, Messages

# example model to use
model = HF_LLM(id='distilgpt2')

# test
test_prompt = "The quick brown fox jumps over the lazy dog."

mutated_output = model.generate(test_prompt)
print("Mutated Output:", mutated_output)

messages = Messages(messages=[
    Message(role="user", content="How are you today?")
])
chat_output = model.chat(messages)
print("Chat Output:", chat_output)
