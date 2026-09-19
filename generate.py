import argparse
import re
import torch
from config import get_config
from tokenizer import CharTokenizer
from model import MiniTransformer
from utils import choose_device


def sample(logits, temperature=0.45, top_k=8, top_p=0.8, recent_ids=None,
           repetition_penalty=1.25, frequency_penalty=0.08, greedy=True):
    """Choose a stable next character and suppress degenerate repetitions."""
    logits = logits.clone() / max(temperature, 1e-5)
    if recent_ids is not None:
        counts = torch.bincount(recent_ids, minlength=logits.numel()).to(logits.device)
        logits -= frequency_penalty * counts
        for token_id in set(int(i) for i in recent_ids.tolist()):
            if logits[token_id] > 0:
                logits[token_id] /= repetition_penalty
            else:
                logits[token_id] *= repetition_penalty

    # A tiny character model is more readable with deterministic decoding.
    if greedy:
        return torch.argmax(logits).view(1)

    if top_k:
        values, _ = torch.topk(logits, min(top_k, logits.numel()))
        logits[logits < values[-1]] = -float("inf")
    probs = torch.softmax(logits, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_ids = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=0)
        remove = cumulative - sorted_probs >= top_p
        sorted_probs[remove] = 0
        sorted_probs /= sorted_probs.sum().clamp_min(1e-12)
        return sorted_ids[torch.multinomial(sorted_probs, 1)]
    return torch.multinomial(probs, 1)


def _would_repeat(ids, candidate):
    """Reject a third identical character in a row."""
    return ids.numel() >= 2 and int(ids[-1]) == candidate and int(ids[-2]) == candidate


def generate(model, tok, prompt, device, max_new_tokens=80, temperature=.45,
             top_k=8, top_p=.8, repetition_penalty=1.25,
             frequency_penalty=.08, greedy=True):
    ids = torch.tensor([tok.encode(prompt, add_bos=True)], device=device, dtype=torch.long)
    original_length = ids.size(1)
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = ids[:, -model.config.context_length:]
            logits, _ = model(context)
            recent = context[0, -min(context.size(1), 64):]
            next_logits = logits[0, -1].clone()
            candidate = sample(next_logits, temperature, top_k, top_p, recent,
                               repetition_penalty, frequency_penalty, greedy)
            token_id = int(candidate.item())
            if _would_repeat(ids[0], token_id):
                next_logits[token_id] = -float("inf")
                candidate = sample(next_logits, temperature, top_k, top_p, recent,
                                   repetition_penalty, frequency_penalty, greedy)
                token_id = int(candidate.item())
            ids = torch.cat([ids, candidate.view(1, 1)], dim=1)
            if token_id == tok.eos_id:
                break
    text = tok.decode(ids[0, original_length:].tolist(), skip_special=True)
    # Keep generated answers readable if a weak checkpoint emits punctuation runs.
    text = re.sub(r"(.)\1{3,}", r"\1\1", text)
    text = re.sub(r"([.!?,])\1{2,}", r"\1", text)
    return text.strip()


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
