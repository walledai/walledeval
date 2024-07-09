# walledeval/judge/lionguard.py

import yaml
import torch
import numpy as np
from pathlib import Path

from transformers import (
    AutoTokenizer, AutoModel,
    AutoModelForSequenceClassification
)
from huggingface_hub import hf_hub_download

from walledeval.judge.core import Judge
from walledeval.judge.classifiers import Ridge, XGBoost

__all__ = [
    "LionGuardJudge"
]


class LionGuardJudge(Judge[None, float, bool]):
    def __init__(self, name: str, type: str,
                 config_file: str,
                 tokenizer: str, embedding_model: str,
                 max_length: int = 512, batch_size: int = 32):
        super().__init__(name)
        
        self.type = type

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.embedding_model = AutoModel.from_pretrained(embedding_model)
        self.max_length = max_length
        self.batch_size = batch_size
        
        self.model_path = hf_hub_download(repo_id=name, filename=config_file)
        
        if self.type == "xgboost":
            self.classifier = XGBoost.from_config(self.model_path)
        elif self.type == "ridge":
            self.classifier = Ridge.from_config(self.model_path)
        else:
            raise NotImplementedError(f"Model type '{self.type}' not implement yet.")
    
    @classmethod
    def from_preset(cls, name: str = "beta"):
        yaml_fp = Path(__file__).resolve().parent / f"presets/{name}.yaml"
        yaml_text = yaml_fp.read_text(encoding="utf-8")
        config = yaml.safe_load(yaml_text)
        
        return cls(
            name = config.get("model_id", ""),
            type = config.get("model_type", "ridge"),
            config_file = config.get("config_file", "config.json"),
            tokenizer = config.get("tokenizer", "BAAI/bge-large-en-v1.5"),
            embedding_model = config.get("embedding_model", "BAAI/bge-large-en-v1.5"),
            max_length = int(config.get("max_length", 512)),
            batch_size = int(config.get("batch_size", 32))
        )

    def embed(self, prompt: str):
        # TODO: Implement Batching
        # num_batches = int(np.ceil(len(data) / self.batch_size))
        # output = []
        
        # for i in range(num_batches):
        #     sentences
        encoded_input = self.tokenizer(
            [prompt], 
            max_length=self.max_length,
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        )
        
        with torch.no_grad():
            model_output = self.embedding_model(**encoded_input)
            embeddings = model_output[0][:, 0]
        
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        output = np.array(embeddings.cpu().numpy())
        # returns (1, embed_dim)
        return output


    def check(self, response: str, answer: None = None) -> float:
        embeddings = self.embed(response)
        preds = self.classifier.predict(embeddings)
        return preds[0]

    def score(self, output: float) -> bool:
        return not round(output)
