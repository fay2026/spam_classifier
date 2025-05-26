import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg["n_heads"]
        self.emb_dim = cfg["emb_dim"]
        self.head_dim = self.emb_dim // self.n_heads
        assert self.head_dim * self.n_heads == self.emb_dim, "emb_dim must be divisible by n_heads"
        
        self.qkv = nn.Linear(self.emb_dim, 3 * self.emb_dim, bias=False)
        self.proj = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        self.attn_drop = nn.Dropout(cfg["drop_rate"])
        self.resid_drop = nn.Dropout(cfg["drop_rate"])
        
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(cfg["context_length"], cfg["context_length"]))
                           .view(1, 1, cfg["context_length"], cfg["context_length"]))
        
    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality
        
        # calculate query, key, values for all heads in batch
        qkv = self.qkv(x).split(self.emb_dim, dim=2)
        q, k, v = map(lambda t: t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2), qkv)
        
        # causal self-attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        
        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
            nn.Dropout(cfg["drop_rate"]),
        )
    
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg["emb_dim"])
        self.attn = MultiHeadAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg["emb_dim"])
        self.ff = FeedForward(cfg)
        
    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # Pre-norm architecture
        x = x + self.ff(self.ln2(x))
        return x
