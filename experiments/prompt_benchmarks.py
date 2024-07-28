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

dataset_args = {
    "harmbench": ("walledai/HarmBench", "standard"),
    "advbench": ("walledai/AdvBench"),
    "aya-redteaming": ("walledai/AyaRedTeaming", "default", "english"),
    "xstest": ("walledai/XSTest"),
    
}



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-m", "--model", default="llama3.1-8b",
                        choices=["llama3.1-8b", "llama3-8b", "llama2-7b",
                                 "gemma2-9b", 
                                 "mistral-nemo-12b", "mistral-7b", "mixtral-8x7b",
                                 "qwen2-7b", "qwen2-1.5b", "qwen2-0.5b"],
                        help="Model to use as SUT")
    
    parser.add_argument("-d", "--dataset", default="harmbench",
                        choices=["harmbench", "advbench", "aya-redteaming"],
                        help="(Prompt-based) Dataset to test")
    
    parser.add_argument("-f", "--filename", default="",
                        help="Place to store logs")
    
    parser.add_argument("-e", "--env", default = ".env", help="Environment file with tokens")
    
    parser.add_argument("-t", "--token_name", default = "HF_TOKEN", help="Environment Variable for token")
    
    
    args = parser.parse_args()
    
    llm_name = args.model
    dataset_name = args.dataset
    output_filename = args.filename if args.filename else f"experiments/logs/{dataset_name}/{llm_name}.json"
    
    load_dotenv(args.env)
    if token := os.getenv(args.token_name):
        login(token)
        
    sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
    
    from walledeval.data import HuggingFaceDataset
    from walledeval.prompts import PromptTemplate
    from walledeval.llm import HF_LLM
    from walledeval.judge import LlamaGuardJudge


    dataset = HuggingFaceDataset.from_hub(*(dataset_args[dataset_name]))
    template = PromptTemplate()

    # if dataset_name == "HarmBench":
    #     # only the 200 rows
    #     dataset = HuggingFaceDataset.from_hub("walledai/HarmBench", "standard")
    #     template = PromptTemplate()
    # elif dataset_name == "AdvBench":
    #     dataset = HuggingFaceDataset.from_hub("walledai/AdvBench")
    #     template = PromptTemplate()

    samples = dataset.all()


    sut_kwargs = dict(
        type = 1,
        device_map="auto",
        model_kwargs=dict(torch_dtype=torch.bfloat16)
    )
    
    model_kwargs = {
        "quantization_config": {"load_in_4bit": True}
    }

    if llm_name == "llama3.1-8b":
        sut = HF_LLM("unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", **sut_kwargs)
    elif llm_name == "llama3-8b":
        sut = HF_LLM("unsloth/llama-3-8b-Instruct-bnb-4bit", **sut_kwargs)
    elif llm_name == "llama2-7b": # no unsloth model
        sut = HF_LLM("meta-llama/Llama-2-7b-chat-hf", **sut_kwargs, 
                     model_kwargs = {**model_kwargs, **(sut_kwargs["model_kwargs"])})
    # elif llm_name == "llama2-13b": # no unsloth model
    #     sut = HF_LLM("meta-llama/Llama-2-13b-chat-hf", **sut_kwargs, 
    #                  model_kwargs = {**model_kwargs, **(sut_kwargs["model_kwargs"])})
    
    elif llm_name == "gemma2-9b":
        sut = HF_LLM("unsloth/gemma-2-9b-it-bnb-4bit", **sut_kwargs)
        
        
    elif llm_name == "mistral-nemo-12b":
        sut = HF_LLM("unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit", **sut_kwargs)
    elif llm_name == "mistral-7b":
        sut = HF_LLM("unsloth/mistral-7b-instruct-v0.3-bnb-4bit", **sut_kwargs)
    elif llm_name == "mixtral-8x7b":
        sut = HF_LLM("ybelkada/Mixtral-8x7B-Instruct-v0.1-bnb-4bit", **sut_kwargs)
        
        
        
    elif llm_name == "phi3-small": # no unsloth model
        sut = None
        
        
        
    elif llm_name == "qwen2-7b": # no unsloth model
        sut = HF_LLM("unsloth/Qwen2-7B-Instruct-bnb-4bit", **sut_kwargs)
    elif llm_name == "qwen2-1.5b": # no unsloth model
        sut = HF_LLM("unsloth/Qwen2-1.5B-Instruct-bnb-4bit", **sut_kwargs)
    elif llm_name == "qwen2-0.5b": # no unsloth model
        sut = HF_LLM("unsloth/Qwen2-0.5B-Instruct-bnb-4bit", **sut_kwargs)
        
    judge = LlamaGuardJudge(2, model_kwargs=model_kwargs, device_map="auto")


    logs = []

    for i, sample in tqdm(enumerate(samples)):
        prompt = template(sample)
        response = sut(prompt)

        if dataset_name == "XSTest":
            judge_output, score = judge(response, int(sample.label == "safe"))
        else:
            judge_output, score = judge(response)
        logs.append({
            "sample": sample.prompt, # all are by default prompt datasets
            "prompt": prompt,
            "response": response,
            "judge_output": judge_output,
            "score": score # True if safe, False if unsafe
        })
    
    with open(output_filename, "w+") as f:
        json.dump(logs, f)



