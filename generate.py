import argparse
import torch
from config import get_config
from tokenizer import CharTokenizer
from model import MiniTransformer
from utils import choose_device


def sample(logits, temperature=0.8, top_k=20, top_p=0.9, recent_ids=None,
           repetition_penalty=1.15, frequency_penalty=0.02):
    """Sample from logits with lightweight anti-repetition penalties."""
    logits = logits.clone() / max(temperature, 1e-5)
    if recent_ids is not None and repetition_penalty != 1.0:
        for token_id in set(int(i) for i in recent_ids):
            if logits[token_id] > 0:
                logits[token_id] /= repetition_penalty
            else:
                logits[token_id] *= repetition_penalty
        counts = torch.bincount(recent_ids, minlength=logits.numel()).to(logits.device)
        logits -= frequency_penalty * counts
    if top_k:
        values, _ = torch.topk(logits, min(top_k, logits.numel()))
        logits[logits < values[-1]] = -float("inf")
    probs = torch.softmax(logits, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_ids = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=0)
        remove = cumulative - sorted_probs >= top_p
        sorted_probs[remove] = 0
        sorted_probs /= sorted_probs.sum()
        return sorted_ids[torch.multinomial(sorted_probs, 1)]
    return torch.multinomial(probs, 1)


def generate(model, tok, prompt, device, max_new_tokens=80, temperature=.8,
             top_k=20, top_p=.9, repetition_penalty=1.15,
             frequency_penalty=.02):
    ids = torch.tensor([tok.encode(prompt, add_bos=True)], device=device, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = ids[:, -model.config.context_length:]
            logits, _ = model(context)
            recent = context[0, -min(context.size(1), 96):]
            nxt = sample(logits[0, -1], temperature, top_k, top_p, recent,
                         repetition_penalty, frequency_penalty)
            ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
            if nxt.item() == tok.eos_id:
                break
    return tok.decode(ids[0].tolist(), skip_special=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--prompt")
    args = ap.parse_args()
    device = choose_device("auto")
    ck = torch.load(args.checkpoint, map_location=device)
    config = get_config("mini")
    config.__dict__.update(ck["config"])
    tok = CharTokenizer(ck["tokenizer"])
    model = MiniTransformer(config).to(device)
    model.load_state_dict(ck["model"])
    prompt = args.prompt
    while prompt is None or prompt.lower() not in {"quit", "exit"}:
        prompt = prompt if prompt is not None else input("you> ")
        print(generate(model, tok, prompt, device))
        prompt = None


if __name__ == "__main__":
    main()
