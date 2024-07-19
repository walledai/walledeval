from walledeval.attacks.mutators import GCG

def calculate_gradients(model, tokenizer, inputs, labels):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
    labels = tokenizer(labels, return_tensors="pt", padding=True, truncation=True)

    input_ids = inputs["input_ids"]
    labels_ids = labels["input_ids"]

    min_length = min(input_ids.shape[1], labels_ids.shape[1])
    input_ids = input_ids[:, :min_length]
    labels_ids = labels_ids[:, :min_length]

    outputs = model(input_ids=input_ids, labels=labels_ids)
    loss = outputs.loss
    loss.backward()

    return input_ids.grad

def modify_input_based_on_gradients(input_ids, gradients):
    return input_ids  

def gcg_attack(model_name, text, max_iters=10, top_k=5):
    mutator = GCG(model_name)
    mutated_text = mutator(text, max_iters=max_iters, top_k=top_k)
    return mutated_text
