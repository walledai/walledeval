import os
from dotenv import load_dotenv
import sys

from huggingface_hub import login

from transformers.utils import logging
logging.set_verbosity_error()

from tqdm import tqdm
import torch
import json
import argparse

import pathlib

root = pathlib.Path(__file__).resolve().parent.parent

sys.path.append(str(root))

from walledeval.data import HuggingFaceDataset
from walledeval.prompts import PromptTemplate
from walledeval.llm import HF_LLM, Llama
from walledeval.judge import LlamaGuardJudge

dataset_args = {
    "harmbench": ("walledai/HarmBench", "standard"),
    "advbench": ("walledai/AdvBench", ),
    "catqa": ("walledai/CatHarmfulQA", "default", "en")
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-m", "--model", default="",
                        choices=[],
                        help="Model to use as SUT")
    
    parser.add_argument("-d", "--dataset", default="harmbench",
                        choices=["harmbench", "advbench", "catqa"],
                        help="(Prompt-based) Dataset to test")
    
    parser.add_argument("-f", "--filename", default="",
                        help="Place to store logs")
    
    parser.add_argument("-e", "--env", default = ".env", help="Environment file with tokens")
    
    parser.add_argument("-t", "--token_name", default = "HF_TOKEN", help="Environment Variable for token")
    
    parser.add_argument("-v", "--verbose", help="Print running logs", action="store_true")
    
    parser.add_argument("-i", "--interval", type=int, default=100, help="Number of runs before saving")
    
    parser.add_argument("-n", "--num", type=int, default=0, help="Number of samples to test")
    
    
    args = parser.parse_args()
    
    llm_name = args.model
    dataset_name = args.dataset
    
    output_filename = args.filename if args.filename else str(root / f"experiments/logs/templates/{dataset_name}/{llm_name}.json")
    
    os.makedirs(
        os.path.dirname(output_filename),
        exist_ok=True
    )
    
    verbose = bool(args.verbose)
    interval = args.interval
    
    num = args.num
    
    load_dotenv(args.env)
    if token := os.getenv(args.token_name):
        login(token)
    
    
    # ==================================================
    # ============== STEP 1: LOAD DATASET ==============
    # ==================================================
    
    dataset = HuggingFaceDataset.from_hub(*(dataset_args[dataset_name]))
    template = PromptTemplate()

    samples = dataset.all() if num == 0 else dataset.sample(num)


    # ====================================================
    # ============== STEP 2: LOAD LLM MODEL ==============
    # ====================================================

    # Llama Models
    if llm_name == "llama3.1-8b":
        sut = HF_LLM("unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", **sut_kwargs)
    elif llm_name == "llama3-8b":
        sut = HF_LLM("unsloth/llama-3-8b-Instruct-bnb-4bit", **sut_kwargs)
    elif llm_name == "llama2-7b":
        # loading in 16 bit rn
        sut = HF_LLM("meta-llama/Llama-2-7b-chat-hf", **sut_kwargs)
    
    # Gemma Models
    elif llm_name == "gemma2-9b":
        sut = HF_LLM("unsloth/gemma-2-9b-it-bnb-4bit", **sut_kwargs)
    elif llm_name == "gemma-1.1-7b":
        sut = HF_LLM("unsloth/gemma-1.1-7b-it-bnb-4bit", **sut_kwargs)
    elif llm_name == "gemma-7b":
        sut = HF_LLM("unsloth/gemma-7b-it-bnb-4bit", **sut_kwargs)
    
    # Mistral Models
    elif llm_name == "mistral-nemo-12b":
        sut = HF_LLM("unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit", **sut_kwargs)
    elif llm_name == "mistral-7b":
        sut = HF_LLM("unsloth/mistral-7b-instruct-v0.3-bnb-4bit", **sut_kwargs)
    elif llm_name == "mixtral-8x7b":
        # uses model quantized by ybelkada
        sut = HF_LLM("ybelkada/Mixtral-8x7B-Instruct-v0.1-bnb-4bit", **sut_kwargs)
        
    # Phi Models
    elif llm_name == "phi3-small":
        # no unsloth model
        # sut = None
        raise NotImplementedError
    elif llm_name == "phi3-mini":
        sut = HF_LLM("unsloth/Phi-3-mini-4k-instruct-bnb-4bit", **sut_kwargs)
    
    # Qwen Models
    elif llm_name == "qwen2-7b":
        sut = HF_LLM("unsloth/Qwen2-7B-Instruct-bnb-4bit", **sut_kwargs)
    elif llm_name == "qwen2-1.5b":
        sut = HF_LLM("unsloth/Qwen2-1.5B-Instruct-bnb-4bit", **sut_kwargs)
    elif llm_name == "qwen2-0.5b":
        sut = HF_LLM("unsloth/Qwen2-0.5B-Instruct-bnb-4bit", **sut_kwargs)
    elif llm_name == "qwen-1.5-7b":
        sut = HF_LLM("Qwen/Qwen1.5-7B-Chat-GPTQ-Int4", 
                     **{**sut_kwargs,
                        "model_kwargs": dict(torch_dtype="auto")})
    
    # Yi Models
    elif llm_name == "yi-1.5-9b":
        # loading in 16 bit rn
        sut = HF_LLM("01-ai/Yi-1.5-9B-Chat-16K", **sut_kwargs)
    elif llm_name == "yi-1.5-6b":
        sut = HF_LLM("01-ai/Yi-6B-Chat-4bits", **sut_kwargs)
    
    
    # =====================================================
    # ============== STEP 3: LOAD LLAMAGUARD ==============
    # =====================================================
    
    judge = Llama.from_pretrained(
        repo_id="QuantFactory/Meta-Llama-Guard-2-8B-GGUF",
        filename="*Q4_K_M.gguf",
        verbose=False
    )
    
    
