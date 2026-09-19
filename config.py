"""CPU-friendly configurations for TinyAI."""
from dataclasses import asdict, dataclass
import json


@dataclass
class Config:
    vocab_size: int = 512
    d_model: int = 384
    num_layers: int = 6
    num_heads: int = 6
    d_ff: int = 1536
    context_length: int = 256
    batch_size: int = 2
    learning_rate: float = 2e-4
    min_learning_rate: float = 2e-5
    dropout: float = 0.05
    epochs: int = 500
    grad_accumulation: int = 8
    num_threads: int = 2
    num_workers: int = 0
    device: str = "auto"
    seed: int = 1337


CONFIGS = {
    "cpu": Config(),
    "tiny": Config(d_model=192, num_layers=4, num_heads=6, d_ff=768,
                   context_length=128, batch_size=4, grad_accumulation=4),
    "mini": Config(),
    "large_mini": Config(d_model=512, num_layers=8, num_heads=8, d_ff=2048,
                          context_length=256, batch_size=1, grad_accumulation=16),
}


def get_config(name="cpu"):
    if name not in CONFIGS:
        raise ValueError(f"Unknown config {name}; choose {list(CONFIGS)}")
    return Config(**asdict(CONFIGS[name]))


def save_config(cfg, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return Config(**json.load(f))
