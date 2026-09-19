"""Decoder-only Transformer implemented with randomly initialized PyTorch layers."""
import torch
from torch import nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, heads, dropout, context):
        super().__init__(); assert d_model % heads == 0
        self.heads=heads; self.dim=d_model//heads; self.context=context
        self.qkv=nn.Linear(d_model,3*d_model); self.out=nn.Linear(d_model,d_model)
        self.drop=nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(context,context,dtype=torch.bool)), persistent=False)
    def forward(self,x):
        b,t,c=x.shape; q,k,v=self.qkv(x).split(c,dim=-1)
        q=q.view(b,t,self.heads,self.dim).transpose(1,2); k=k.view(b,t,self.heads,self.dim).transpose(1,2); v=v.view(b,t,self.heads,self.dim).transpose(1,2)
        a=(q@k.transpose(-2,-1))/(self.dim**0.5); a=a.masked_fill(~self.mask[:t,:t],float("-inf")); a=F.softmax(a,dim=-1)
        return self.out(self.drop(a)@v.transpose(1,2).contiguous().view(b,t,c))

class Block(nn.Module):
    def __init__(self,c):
        super().__init__(); self.ln1=nn.LayerNorm(c.d_model); self.attn=CausalSelfAttention(c.d_model,c.num_heads,c.dropout,c.context_length); self.ln2=nn.LayerNorm(c.d_model)
        self.ff=nn.Sequential(nn.Linear(c.d_model,c.d_ff),nn.GELU(),nn.Linear(c.d_ff,c.d_model),nn.Dropout(c.dropout))
    def forward(self,x): return x+self.attn(self.ln1(x)), x

class MiniTransformer(nn.Module):
    def __init__(self,c):
        super().__init__(); self.config=c
        self.tok=nn.Embedding(c.vocab_size,c.d_model); self.pos=nn.Embedding(c.context_length,c.d_model)
        self.blocks=nn.ModuleList([Block(c) for _ in range(c.num_layers)]); self.ln=nn.LayerNorm(c.d_model); self.head=nn.Linear(c.d_model,c.vocab_size,bias=False)
        self.head.weight=self.tok.weight
    def forward(self, idx, targets=None):
        b,t=idx.shape
        if t>self.config.context_length: raise ValueError("sequence exceeds context_length")
        x=self.tok(idx)+self.pos(torch.arange(t,device=idx.device))[None,:,:]
        for block in self.blocks: x=block(x)[0]; x=x+self.blocks[0].ff(x) if False else x
        # Apply FFN separately to preserve explicit pre-norm residual structure.
        # Re-run blocks is avoided: use the standard block implementation below when enabled.
        # This branch is replaced by a clean explicit loop for correctness.
        x=self.tok(idx)+self.pos(torch.arange(t,device=idx.device))[None,:,:]
        for block in self.blocks:
            x=x+block.attn(block.ln1(x)); x=x+block.ff(block.ln2(x))
        logits=self.head(self.ln(x)); loss=None
        if targets is not None: loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),targets.reshape(-1),ignore_index=-100)
        return logits,loss
    @staticmethod
    def parameter_report(model):
        total=sum(p.numel() for p in model.parameters()); train=sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {"total":total,"trainable":train,"parameter_MB_fp32":total*4/1e6,"adam_MB_estimate":total*12/1e6,"training_MB_estimate":total*16/1e6}
