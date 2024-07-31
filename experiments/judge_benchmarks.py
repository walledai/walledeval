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
from walledeval.judge import (
    LlamaGuardJudge, WalledGuardJudge, LionGuardJudge,
    ToxicityModelJudge, PromptGuardJudge, UnitaryJudge
)


dataset_args = {
    "xstest": ("walledai/XSTest", ),
}

for language in ["arabic", "english", "filipino", "french", "hindi", "russian", "serbian", "spanish"]:
    dataset_args[f"aya-{language[:2]}"] = ("walledai/AyaRedTeaming", "default", language)

class RefusalMCQ(BaseModel):
    prompt: str
    label: str

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-j", "--judge", default="llamaguard3",
                        choices=["llamaguard3", "llamaguard2", "llamaguard",
                                 "walledguard", "lionguard", "promptguard",
                                 "toxicitymodel", "toxic-bert", 
                                 "multilingual-toxic-xlm-roberta",
                                 "unbiased-toxic-roberta"],
                        help="Model to use as SUT")
    
    parser.add_argument("-d", "--dataset", default="xstest",
                        choices=dataset_args.keys(),
                        help="(Prompt-based) Dataset to test")
    
    parser.add_argument("-f", "--filename", default="",
                        help="Place to store logs")
    
    parser.add_argument("-e", "--env", default = ".env", help="Environment file with tokens")
    
    parser.add_argument("-t", "--token_name", default = "HF_TOKEN", help="Environment Variable for token")
    
    parser.add_argument("-v", "--verbose", help="Print running logs", action="store_true")
    
    parser.add_argument("-i", "--interval", type=int, default=100, help="Number of runs before saving")
    
    parser.add_argument("-n", "--num", type=int, default=0, help="Number of samples to test")
    
    
    args = parser.parse_args()
    
    judge_name = args.judge
    dataset_name = args.dataset
    
    output_filename = args.filename if args.filename else f"experiments/logs/judge-eval/{dataset_name}/{judge_name}.json"
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
    
    if dataset_name in ["xstest"]:
        dataset = HuggingFaceDataset[RefusalMCQ].from_hub(
            *(dataset_args[dataset_name])
        )
    else:
        dataset = HuggingFaceDataset.from_hub(
            *(dataset_args[dataset_name])
        )
    
    template = PromptTemplate()

    samples = dataset.all() if num == 0 else dataset.sample(num)


    # ======================================================
    # ============== STEP 2: LOAD JUDGE MODEL ==============
    # ======================================================

    model_kwargs = {
        "quantization_config": {"load_in_4bit": True}
    }

    if judge_name.startswith("llamaguard"):
        judge = LlamaGuardJudge(int(judge_name[-1]), model_kwargs=model_kwargs, device_map="auto")
    
    elif judge_name == "walledguard":
        judge = WalledGuardJudge(model_kwargs=model_kwargs, device_map="auto")
    
    elif judge_name == "lionguard":
        judge = LionGuardJudge.from_preset("binary")
    
    elif judge_name == "promptguard":
        judge = PromptGuardJudge()
    
    elif judge_name == "toxicitymodel":
        judge = ToxicityModelJudge()
    
    elif judge_name in [
        "toxic-bert", "unbiased-toxic-roberta", 
        "multilingual-toxic-xlm-roberta",
    ]:
        judge = UnitaryJudge(judge_name)


    # ================================================================
    # ============== STEP 3: TEST MODEL AGAINST DATASET ==============
    # ================================================================
    
    running_score = 0
    logs = []

    try:
        for i, sample in tqdm(enumerate(samples)):
            prompt = template(sample)
            output, pred = judge(prompt, max_new_tokens=10)

            if isinstance(sample, RefusalMCQ):
                label = sample.label == "safe"
            else:
                label = False

            score = pred == label
            
            logs.append({
                "prompt": prompt,
                "label": label,
                "output": output,
                "predicted": pred,
                "score": score
            })
            
            if score:
                running_score += 1
            
            if verbose:
                print("\n\n---------")
                print(f"{i+1}/{len(samples)}")
                print("prompt:", prompt)
                print("label:", label)
                print("output:", output)
                print("predicted:", pred)
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
