"""Decoder-only Transformer with optimized PyTorch attention kernels."""
import torch
from torch import nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, heads, dropout, context):
        super().__init__()
        if d_model % heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.heads = heads
        self.dim = d_model // heads
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch, tokens, channels = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (batch, tokens, self.heads, self.dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        # PyTorch selects the fastest available CPU/GPU attention kernel.
        attended = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, tokens, channels)
        return self.out(attended)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config.d_model, config.num_heads,
                                        config.dropout, config.context_length)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff), nn.GELU(),
            nn.Linear(config.d_ff, config.d_model), nn.Dropout(config.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ff(self.ln2(x))


class MiniTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok = nn.Embedding(config.vocab_size, config.d_model)
        self.pos = nn.Embedding(config.context_length, config.d_model)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.num_layers))
        self.ln = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, idx, targets=None):
        _, tokens = idx.shape
        if tokens > self.config.context_length:
            raise ValueError("sequence exceeds context_length")
        positions = torch.arange(tokens, device=idx.device)
        x = self.tok(idx) + self.pos(positions).unsqueeze(0)
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1), ignore_index=-100)
        return logits, loss

    @staticmethod
    def parameter_report(model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable,
                "parameter_MB_fp32": total * 4 / 1e6,
                "adam_MB_estimate": total * 12 / 1e6,
                "training_MB_estimate": total * 16 / 1e6}
