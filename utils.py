import os, random, torch

def seed_all(seed): random.seed(seed); torch.manual_seed(seed)
def choose_device(name): return torch.device("cuda" if name=="cuda" or (name=="auto" and torch.cuda.is_available()) else "cpu")
def set_cpu_threads(n):
    if n: torch.set_num_threads(n)
def perplexity(loss): return float(torch.exp(torch.tensor(loss))) if loss < 20 else float("inf")
def rss_mb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    except (ImportError, AttributeError): return None
