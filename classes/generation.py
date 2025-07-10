import torch


class TextTokenConversion():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode(self, text):
        encoded = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})

        return torch.tensor(encoded).unsqueeze(0)
    
    def decode(self, ids):
        return self.tokenizer.decode(ids.squeeze(0).tolist())


class generator():
    def __init__(self, model, encoder):
        self.model = model
        self.encoder = encoder

        self.context_size = self.model.position_emb.weight.shape[0]

    
    def topk_sample(self, logits, topk):
        top_logits, _ = torch.topk(logits, k=topk)

        return torch.where(
            condition=logits < top_logits[:, -1],
            input=torch.tensor(float("-inf")).to(logits.device),
            other=logits
        )
    
    def temperature_scalling(self, logits, temperature):
        return logits / temperature

    
    def generate_text_simple(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            context = idx[:, -self.context_size:]

            with torch.no_grad():
                logits = self.model(context)
            prob = torch.softmax(logits[:, -1, :], dim=-1)
            new_idx = torch.argmax(prob, dim=-1, keepdim=True)

            idx = torch.cat((idx, new_idx), dim=1)

        return idx
    

    def generate_text_temperature(self, idx, max_new_tokens, **kwargs):
        for _ in range(max_new_tokens):
            context = idx[:, -self.context_size:]

            with torch.no_grad():
                logits = self.model(context)
            
            logits = self.temperature_scalling(logits, kwargs["temperature"])
            
            prob = torch.softmax(logits[:, -1, :], dim=-1)

            # this allows the network to have some unpredictability, by chosing the next token based on their probability
            new_idx = torch.multinomial(prob, num_samples=1)

            idx = torch.cat((idx, new_idx), dim=1)

        return idx
    

    def generate_text_topk(self, idx, max_new_tokens, **kwargs):
        for _ in range(max_new_tokens):
            context = idx[:, -self.context_size:]

            with torch.no_grad():
                logits = self.model(context)
            
            masked_logits = self.topk_sample(logits, kwargs["topk"])
            
            prob = torch.softmax(masked_logits[:, -1, :], dim=-1)

            # this allows the network to have some unpredictability, by chosing the next token based on their probability
            new_idx = torch.multinomial(prob, num_samples=1)

            idx = torch.cat((idx, new_idx), dim=1)

        return idx
    
    
    def generate_text(self, idx, max_new_tokens, temperature=0.0, topk=None, eos_id=None):
        for _ in range(max_new_tokens):
            context = idx[:, -self.context_size:]

            with torch.no_grad():
                logits = self.model(context)
            
            # get only the last logit set
            logits = logits[:, -1, :]

            if not topk is None:
                logits = self.topk_sample(logits, topk)

            if temperature > 0:
                logits = self.temperature_scalling(logits, temperature)

                prob = torch.softmax(logits, dim=-1)
                new_idx = torch.multinomial(prob, num_samples=1)
            else:
                prob = torch.softmax(logits, dim=-1)
                new_idx = torch.argmax(prob, dim=-1, keepdim=True)

            # Indicates when the llm should stop generating
            if new_idx == eos_id:
                break

            idx = torch.cat((idx, new_idx), dim=1)

        return idx
    
    def generate_text_input_text(self, text, max_new_tokens, temperature=0.0, topk=None, eos_id=None):
        idx = self.encoder.encode(text)

        for _ in range(max_new_tokens):
            context = idx[:, -self.context_size:]

            with torch.no_grad():
                logits = self.model(context)
            
            # get only the last logit set
            logits = logits[:, -1, :]

            if not topk is None:
                logits = self.topk_sample(logits, topk)

            if temperature > 0:
                logits = self.temperature_scalling(logits, temperature)

                prob = torch.softmax(logits, dim=-1)
                new_idx = torch.multinomial(prob, num_samples=1)
            else:
                prob = torch.softmax(logits, dim=-1)
                new_idx = torch.argmax(prob, dim=-1, keepdim=True)

            # Indicates when the llm should stop generating
            if new_idx == eos_id:
                break

            idx = torch.cat((idx, new_idx), dim=1)

        return idx


    def generate_and_print(self, start_context, num_gen_tokens, method, device, **kwargs):
        if method == "simple":
            func = self.generate_text_simple
        elif method == "temperature":
            func = self.generate_text_temperature
        else:
            ValueError("No method with name {method} is found")

        self.model.eval()

        encoded = self.encoder.encode(start_context).to(device)

        token_ids = func(
            idx=encoded,
            max_new_tokens=num_gen_tokens,
            **kwargs
        )

        decoded_text = self.encoder.decode(token_ids)
        print(decoded_text.replace("\n", " "))

        self.model.train()
