import torch
import torch.nn as nn

from classes.attention import MultiHeadAttention
from classes.activation import GELU


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

    def backward(self):
        pass
    

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
