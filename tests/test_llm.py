import os
import torch
from dotenv import load_dotenv
from bastion.llm import HF_LLM, Claude

load_dotenv(".env")

CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

# def test_tiny_llama():
#     llm = HF_LLM("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")
#     output = llm.generate("Hi, how are you?")
#     assert isinstance(output, str)


