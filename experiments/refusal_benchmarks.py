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

from pydantic import BaseModel

import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

from walledeval.data import HuggingFaceDataset
from walledeval.prompts import PromptTemplate
from walledeval.llm import HF_LLM
from walledeval.judge import MCQJudge


dataset_args = {
    "xstest": ("walledai/XSTest", ),
}

class RefusalMCQ(BaseModel):
    prompt: str
    label: str

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-m", "--model", default="llama3.1-8b",
                        choices=["llama3.1-8b", "llama3-8b", "llama2-7b",
                                 "gemma2-9b", "gemma-1.1-7b", "gemma-7b",
                                 "mistral-nemo-12b", "mistral-7b", "mixtral-8x7b",
                                 "phi3-mini",
                                 "qwen2-7b", "qwen2-1.5b", "qwen2-0.5b", "qwen-1.5-7b",
                                 "yi-1.5-9b", "yi-1.5-6b"],
                        help="Model to use as SUT")
    
    # parser.add_argument("-d", "--dataset", default="xstest",
    #                     choices=["xstest"],
    #                     help="(Prompt-based) Dataset to test")
    
    parser.add_argument("-f", "--filename", default="",
                        help="Place to store logs")
    
    parser.add_argument("-e", "--env", default = ".env", help="Environment file with tokens")
    
    parser.add_argument("-t", "--token_name", default = "HF_TOKEN", help="Environment Variable for token")
    
    parser.add_argument("-v", "--verbose", help="Print running logs", action="store_true")
    
    parser.add_argument("-i", "--interval", type=int, default=100, help="Number of runs before saving")
    
    parser.add_argument("-n", "--num", type=int, default=0, help="Number of samples to test")
    
    
    args = parser.parse_args()
    
    llm_name = args.model
    # dataset_name = args.dataset
    
    output_filename = args.filename if args.filename else f"experiments/logs/xstest/{llm_name}.json"
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
    
    dataset = HuggingFaceDataset[RefusalMCQ].from_hub(
        "walledai/XSTest"
        # *(dataset_args[dataset_name])
    )
    template = PromptTemplate.from_preset("mcq/refusal")

    samples = dataset.all() if num == 0 else dataset.sample(num)


    # ====================================================
    # ============== STEP 2: LOAD LLM MODEL ==============
    # ====================================================

    sut_kwargs = dict(
        type = 1,
        device_map="auto",
        model_kwargs=dict(torch_dtype=torch.bfloat16)
    )
    
    model_kwargs = {
        "quantization_config": {"load_in_4bit": True}
    }

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
    

    # ====================================================
    # ============== STEP 3: LOAD MCQ JUDGE ==============
    # ====================================================
    
    judge = MCQJudge(["A", "B"], unknown_answer=0)


    # ================================================================
    # ============== STEP 3: TEST MODEL AGAINST DATASET ==============
    # ================================================================
    
    running_score = 0
    logs = []

    try:
        for i, sample in tqdm(enumerate(samples)):
            prompt = template(sample)
            response = sut(prompt)

            judge_output, score = judge(response, int(sample.label == "safe"))
            
            logs.append({
                "sample": sample.prompt,
                "label": sample.label,
                "prompt": prompt,
                "response": response,
                "judge_output": "safe" if judge_output.predicted else "unsafe",
                "score": score # True if judge_output matches label
            })
            
            if score:
                running_score += 1
            
            if verbose:
                print("\n\n---------")
                print(f"{i+1}/{len(samples)}")
                print("sample:", sample.prompt)
                print("expected label", sample.label)
                print("response:", response)
                print("judge label:", "safe" if judge_output.predicted else "unsafe")
                print("score:", score)
                print("running score:", round(running_score / (i+1), 3))

            if (i+1) % interval == 0:
                with open(output_filename, "w") as f:
                    json.dump(logs, f, indent=4, ensure_ascii=False)
                
                if verbose:
                    print("Saved", len(logs), "logs to", output_filename)
    except KeyboardInterrupt:
        pass
    finally:
        if verbose:
            print("\n\n\n---------")
            print("Final score:", round(running_score/len(logs), 3))
        
        with open(output_filename, "w") as f:
            json.dump(logs, f, indent=4, ensure_ascii=False)
            if verbose:
                print("Saved", len(logs), "logs to", output_filename)
