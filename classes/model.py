import torch
import torch.nn as nn
import numpy as np

from classes.attention import MultiHeadAttention
from classes.activation import GELU

from tools.gpt_downloader import download_and_load_gpt2


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Size mismatch: {left.shape} - {right.shape}")
    
    return torch.nn.Parameter(torch.tensor(right))


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.token_emb = nn.Embedding(cfg['vocab_size'], cfg["emb_dim"])
        self.position_emb = nn.Embedding(cfg['context_length'], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg['n_layers'])]
        )

        self.final_norm = LayerNormalization(cfg["emb_dim"])
        self.output_layer = nn.Linear(cfg['emb_dim'], cfg["vocab_size"], bias=False)

        self.gpt_model_name = cfg["gpt_model_name"]
        self.gpt_num_params = cfg["gpt_num_params"]
    
    def forward(self, in_indexes):
        bacth_size, sequence_shape = in_indexes.shape

        tok_embeds = self.token_emb(in_indexes)
        pos_embeds = self.position_emb(torch.arange(sequence_shape, device=in_indexes.device))

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)

        logits = self.output_layer(x)
        return logits
    
    def save_weights(self, filepath):
        torch.save(self.state_dict(), filepath)
    
    def load_weights(self, filepath, device):
        self.load_state_dict(torch.load(filepath, map_location=device))

    def load_weights_gpt(self):
        settings, params = download_and_load_gpt2(model_size=self.gpt_num_params, models_dir="gpt2")

        self.position_emb.weight = assign(self.position_emb.weight, params['wpe'])
        self.token_emb.weight = assign(self.token_emb.weight, params['wte'])
        
        for b in range(len(params["blocks"])):
            q_w, k_w, v_w = np.split(
                (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
            self.trf_blocks[b].attention.W_queries.weight = assign(
                self.trf_blocks[b].attention.W_queries.weight, q_w.T)
            self.trf_blocks[b].attention.W_keys.weight = assign(
                self.trf_blocks[b].attention.W_keys.weight, k_w.T)
            self.trf_blocks[b].attention.W_values.weight = assign(
                self.trf_blocks[b].attention.W_values.weight, v_w.T)

            q_b, k_b, v_b = np.split(
                (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
            self.trf_blocks[b].attention.W_queries.bias = assign(
                self.trf_blocks[b].attention.W_queries.bias, q_b)
            self.trf_blocks[b].attention.W_keys.bias = assign(
                self.trf_blocks[b].attention.W_keys.bias, k_b)
            self.trf_blocks[b].attention.W_values.bias = assign(
                self.trf_blocks[b].attention.W_values.bias, v_b)

            self.trf_blocks[b].attention.out_proj.weight = assign(
                self.trf_blocks[b].attention.out_proj.weight, 
                params["blocks"][b]["attn"]["c_proj"]["w"].T)
            self.trf_blocks[b].attention.out_proj.bias = assign(
                self.trf_blocks[b].attention.out_proj.bias, 
                params["blocks"][b]["attn"]["c_proj"]["b"])

            self.trf_blocks[b].fforward.layers[0].weight = assign(
                self.trf_blocks[b].fforward.layers[0].weight, 
                params["blocks"][b]["mlp"]["c_fc"]["w"].T)
            self.trf_blocks[b].fforward.layers[0].bias = assign(
                self.trf_blocks[b].fforward.layers[0].bias, 
                params["blocks"][b]["mlp"]["c_fc"]["b"])
            self.trf_blocks[b].fforward.layers[2].weight = assign(
                self.trf_blocks[b].fforward.layers[2].weight, 
                params["blocks"][b]["mlp"]["c_proj"]["w"].T)
            self.trf_blocks[b].fforward.layers[2].bias = assign(
                self.trf_blocks[b].fforward.layers[2].bias, 
                params["blocks"][b]["mlp"]["c_proj"]["b"])

            self.trf_blocks[b].norm_1.scale = assign(
                self.trf_blocks[b].norm_1.scale, 
                params["blocks"][b]["ln_1"]["g"])
            self.trf_blocks[b].norm_1.shift = assign(
                self.trf_blocks[b].norm_1.shift, 
                params["blocks"][b]["ln_1"]["b"])
            self.trf_blocks[b].norm_2.scale = assign(
                self.trf_blocks[b].norm_2.scale, 
                params["blocks"][b]["ln_2"]["g"])
            self.trf_blocks[b].norm_2.shift = assign(
                self.trf_blocks[b].norm_2.shift, 
                params["blocks"][b]["ln_2"]["b"])

        self.final_norm.scale = assign(self.final_norm.scale, params["g"])
        self.final_norm.shift = assign(self.final_norm.shift, params["b"])
        self.output_layer.weight = assign(self.output_layer.weight, params["wte"])


class LayerNormalization(nn.Module):
    def __init__(self, length):
        super().__init__()

        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(length))
        self.shift = nn.Parameter(torch.zeros(length))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift    


# Feed Forward (the basic building block for the transformer block)
class FeedForward(nn.Module):
    def __init__(self, cfg, expansion_rate=4):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], expansion_rate * cfg["emb_dim"]),
            GELU(),
            nn.Linear(expansion_rate * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.attention = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.fforward = FeedForward(cfg)
        self.norm_1 = LayerNormalization(cfg["emb_dim"])
        self.norm_2 = LayerNormalization(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        short = x
        x = self.norm_1(x)
        x = self.attention(x)
        x = self.drop_shortcut(x)
        x  = x + short

        short = x
        x = self.norm_2(x)
        x = self.fforward(x)
        x = self.drop_shortcut(x)
        x = x + short
        return x
