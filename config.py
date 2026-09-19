"""Configurable settings for TinyAI's decoder Transformer."""
from dataclasses import dataclass, asdict
import json


@dataclass
class Config:
    vocab_size: int = 512
    d_model: int = 192
    num_layers: int = 4
    num_heads: int = 6
    d_ff: int = 768
    context_length: int = 256
    batch_size: int = 4
    learning_rate: float = 1e-4
    dropout: float = 0.05
    epochs: int = 500
    grad_accumulation: int = 1
    num_threads: int = 2
    num_workers: int = 0
    device: str = "auto"
    seed: int = 1337


CONFIGS = {
    "tiny": Config(d_model=96, num_layers=2, num_heads=4, d_ff=384, context_length=128, batch_size=8),
    "mini": Config(),
    "large_mini": Config(d_model=256, num_layers=6, num_heads=8, d_ff=1024, context_length=256, batch_size=2),
}


def get_config(name="mini"):
    if name not in CONFIGS:
        raise ValueError(f"Unknown config {name}; choose {list(CONFIGS)}")
    return Config(**asdict(CONFIGS[name]))


def save_config(cfg, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return Config(**json.load(f))
