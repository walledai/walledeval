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

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

from walledeval.data import HuggingFaceDataset
from walledeval.types import Prompt
from walledeval.llm import HF_LLM
from walledeval.attacks.mutators import GenerativeMutator
from walledeval.attacks.wildteaming import WildTeamingAttack


dataset_args = {
    "harmbench": ("walledai/HarmBench", "standard"),
    "advbench": ("walledai/AdvBench", ),
    "xstest": ("walledai/XSTest"),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model", default="mistral-7b", 
                        choices=["mistral-7b", "llama2-7b-uncensored"], 
                        help="Model to use")

    parser.add_argument("-d", "--dataset", 
                        default="harmbench", choices=["harmbench", "advbench"], help="(Prompt-based) Dataset to test")

    parser.add_argument("-f", "--filename", 
                        default="", help="Place to store logs")

    parser.add_argument("-e", "--env", 
                        default=".env", help="Environment file with tokens")

    parser.add_argument("-t", "--token_name", 
                        default="HF_TOKEN", help="Environment Variable for token")

    parser.add_argument("-v", "--verbose", 
                        help="Print running logs", action="store_true")

    parser.add_argument("-i", "--interval", 
                        type=int, default=1, 
                        help="Number of runs before saving")

    parser.add_argument("-n", "--num", 
                        type=int, default=0, 
                        help="Number of samples to test")

    args = parser.parse_args()
    
    llm_name = args.model
    dataset_name = args.dataset
    
    output_filename = args.filename if args.filename else f"experiments/logs/mutations/{dataset_name}/{llm_name}.json"
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

    samples: list[Prompt] = dataset.all() if num == 0 else dataset.sample(num)


    # ====================================================
    # ============== STEP 2: LOAD LLM MODEL ==============
    # ====================================================
    
    llm_kwargs = dict(
        device_map="auto",
        model_kwargs=dict(torch_dtype=torch.bfloat16)
    )
    
    model_kwargs = {
        "quantization_config": {"load_in_4bit": True},
        "torch_dtype": torch.bfloat16
    }
    
    if llm_name == "llama2-7b-uncensored":
        llm = HF_LLM("georgesung/llama2_7b_chat_uncensored", 
                     **{**llm_kwargs, "model_kwargs": model_kwargs})
    elif llm_name == "mistral-7b":
        llm = HF_LLM("unsloth/mistral-7b-instruct-v0.3-bnb-4bit", **llm_kwargs)
    
    
    # ================================================================
    # ============== STEP 3: LOAD WILDTEAMING ATTACK ==============
    # ================================================================

    tactics = [
        #"autodan/revise", 
        "masterkey/rephrase",
        "renellm/alter-sentence-structure",
        "renellm/change-style",
        "renellm/insert-meaningless-characters",
        "renellm/misspell-sensitive-words",
        "renellm/paraphrase-fewer-words",
        "renellm/translation",
        "future-tense", "past-tense"
    ]
    
    mutators = {
        name: GenerativeMutator.from_preset(name, llm)
        for name in tactics
    }


    logs = []

    try:
        for i, sample in tqdm(enumerate(samples)):
            prompt = sample.prompt
            
            if verbose:
                print("\n\n---------")
                print(f"{i+1}/{len(samples)}")
                print("prompt:", prompt)
                print()
                
            mutated = []
            
            for j, (name, mutator) in enumerate(mutators.items()):
                mutated_sample = mutator.mutate(prompt)
                print(name, "mutated sample:", mutated_sample)
                mutated.append({
                    "mutator": name,
                    "prompt": mutated_sample
                })
            
            logs.append({
                "prompt": prompt,
                "mutations": mutated
            })
            
            if (i + 1) % interval == 0:
                with open(output_filename, "w") as f:
                    json.dump(logs, f, indent=4, ensure_ascii=False)
                    if verbose:
                        print("Saved", len(logs), "logs to", output_filename)
    except KeyboardInterrupt:
        pass
    finally:
        with open(output_filename, "w") as f:
            json.dump(logs, f, indent=4, ensure_ascii=False)
            if verbose:
                print("Saved", len(logs), "logs to", output_filename)