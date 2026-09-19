"""Configurable CPU-first settings for a from-scratch decoder Transformer."""
from dataclasses import dataclass, asdict
import json

@dataclass
class Config:
    vocab_size: int = 512
    d_model: int = 128
    num_layers: int = 2
    num_heads: int = 4
    d_ff: int = 512
    context_length: int = 128
    batch_size: int = 8
    learning_rate: float = 3e-4
    dropout: float = 0.1
    epochs: int = 20
    grad_accumulation: int = 1
    num_threads: int = 2
    num_workers: int = 0
    device: str = "auto"
    seed: int = 1337

CONFIGS = {
    "tiny": Config(d_model=96, num_layers=2, num_heads=4, d_ff=384, context_length=96, batch_size=8),
    "mini": Config(),
    "large_mini": Config(d_model=160, num_layers=3, num_heads=5, d_ff=640, context_length=256, batch_size=4),
}

def get_config(name="mini"):
    if name not in CONFIGS:
        raise ValueError(f"Unknown config {name}; choose {list(CONFIGS)}")
    return CONFIGS[name]

def save_config(cfg, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

def load_config(path):
    with open(path, encoding="utf-8") as f:
        return Config(**json.load(f))
