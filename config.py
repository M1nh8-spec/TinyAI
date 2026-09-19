"""Portable hardware-aware configurations for TinyAI."""
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
    # Safe baseline for old dual-core CPUs and 8GB RAM.
    "cpu": Config(),
    "tiny": Config(d_model=192, num_layers=4, num_heads=6, d_ff=768,
                   context_length=128, batch_size=4, grad_accumulation=4),
    "mini": Config(),
    "large_mini": Config(d_model=512, num_layers=8, num_heads=8, d_ff=2048,
                          context_length=256, batch_size=1, grad_accumulation=16),
}


def _ram_gb():
    try:
        import ctypes
        class Memory(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                        ("available", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong),
                        ("free", ctypes.c_ulonglong), ("total_page", ctypes.c_ulonglong),
                        ("free_page", ctypes.c_ulonglong), ("total_virtual", ctypes.c_ulonglong),
                        ("free_virtual", ctypes.c_ulonglong), ("free_extended", ctypes.c_ulonglong)]
        m = Memory()
        m.length = ctypes.sizeof(Memory)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return m.total / (1024 ** 3)
    except (AttributeError, OSError, TypeError):
        pass
    try:
        pages = __import__("os").sysconf("SC_PHYS_PAGES")
        size = __import__("os").sysconf("SC_PAGE_SIZE")
        return pages * size / (1024 ** 3)
    except (AttributeError, ValueError, OSError):
        return 8.0


def hardware_config():
    """Select a conservative model automatically on any supported computer."""
    import os
    cores = os.cpu_count() or 2
    ram = _ram_gb()
    if ram < 6 or cores <= 2:
        name = "tiny"
    elif ram >= 16 and cores >= 8:
        name = "large_mini"
    else:
        name = "cpu"
    cfg = Config(**asdict(CONFIGS[name]))
    cfg.num_threads = max(1, min(cores, 8))
    if ram < 10:
        cfg.batch_size = 1
        cfg.grad_accumulation = max(cfg.grad_accumulation, 8)
    return name, cfg, cores, ram


def get_config(name="auto"):
    if name == "auto":
        return hardware_config()[1]
    if name not in CONFIGS:
        raise ValueError(f"Unknown config {name}; choose auto or {list(CONFIGS)}")
    return Config(**asdict(CONFIGS[name]))


def save_config(cfg, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return Config(**json.load(f))
