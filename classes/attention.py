import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_in, d_out, context_length, dropout=0.0, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.d_head = d_out // num_heads

        self.W_queries = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_keys = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_values = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)

        # causal mask
        self.register_buffer("causal_mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

        # dropout mask
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # we split the matrices by adding a new dimension (d_head)
        queries = self.W_queries(x)
        queries = queries.view(b, num_tokens, self.num_heads, self.d_head)
        queries = queries.transpose(1, 2)

        keys = self.W_keys(x)
        keys = keys.view(b, num_tokens, self.num_heads, self.d_head)
        keys = keys.transpose(1, 2)

        values = self.W_values(x)
        values = values.view(b, num_tokens, self.num_heads, self.d_head)
        values = values.transpose(1, 2)

        att_scores = queries @ keys.transpose(2, 3)
        att_scores.masked_fill_(
            self.causal_mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )

        att_scores = torch.softmax(att_scores / keys.shape[-1]**0.5, dim=-1)

        att_scores = self.dropout(att_scores)

        context_vector = (att_scores @ values).transpose(1,2) 

        # Combine the results from each head
        context_vector = context_vector.contiguous().view(b, num_tokens, self.d_out) # "equivalent" to reshape()
        context_vector = self.out_proj(context_vector)
        return context_vector