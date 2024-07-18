from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class GCGMutator(ABC):
    def __init__(self, model_name):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()

    @abstractmethod
    def mutate(self, inputs, **kwargs):
        # use mutation logic for GCG
        raise NotImplementedError

    def __call__(self, inputs, **kwargs):
        return self.mutate(inputs, **kwargs)

class MyGCGMutator(GCGMutator):
    def mutate(self, prompt, **kwargs):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        
        # forward pass to compute gradients
        outputs = self.model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss
        loss.backward()

        # Example gradient access and manipulation
        grad = inputs['input_ids'].grad.data
        token_id = grad.abs().argmax()  # find token with highest gradient
        inputs['input_ids'][0, token_id] += 1  # naive mutation approach: increment token ID

        mutated_prompt = self.tokenizer.decode(inputs['input_ids'][0])
        return mutated_prompt
