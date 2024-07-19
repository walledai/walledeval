# walledeval/attacks/mutators.py

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class GCG:
    def __init__(self, model_name):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()

        # Ensure all parameters require gradients
        for param in self.model.parameters():
            param.requires_grad = True

    def mutate(self, prompt, max_iters=50, top_k=5, step_size=0.5):
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        
        for iteration in range(max_iters):
            print(f"Iteration {iteration + 1}/{max_iters}")
            
            # Zero gradients
            self.model.zero_grad()

            # Get embeddings for input_ids
            embeddings = self.model.get_input_embeddings()(inputs['input_ids'])
            embeddings = embeddings.clone().detach().requires_grad_(True)

            # Forward pass to compute gradients
            outputs = self.model(inputs_embeds=embeddings, labels=inputs['input_ids'])
            loss = outputs.loss
            loss.backward()

            # Ensure gradients are not None
            if embeddings.grad is None:
                raise RuntimeError("Gradients are None. Ensure the model supports gradient calculation.")

            # Access and manipulate the gradient
            grad = embeddings.grad
            top_k_tokens = grad.abs().sum(dim=-1).topk(top_k).indices.squeeze()
            token_id = top_k_tokens[torch.randint(0, top_k, (1,)).item()]  # Randomly pick from top-k

            # Log the token being mutated and its gradient
            print(f"Mutating token at position {token_id.item()} with gradient {grad[0, token_id].sum().item()}")

            # Naive mutation approach: modify embeddings directly
            with torch.no_grad():
                update = torch.sign(grad[0, token_id]) * step_size  # Adjust based on gradient direction
                embeddings[0, token_id] += update

            # Re-tokenize modified embeddings
            with torch.no_grad():
                modified_input_ids = torch.argmax(self.model.get_output_embeddings()(embeddings), dim=-1)
                inputs['input_ids'] = modified_input_ids

            # Log modified input IDs
            print(f"Modified input IDs: {inputs['input_ids']}")

        mutated_prompt = self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
        print(f"Mutated prompt: {mutated_prompt}")
        return mutated_prompt

    def __call__(self, inputs, **kwargs):
        return self.mutate(inputs, **kwargs)
