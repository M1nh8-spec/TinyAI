"""Train TinyAI with AdamW, accumulation, clipping, and cosine decay."""
import argparse
import math
import os
import torch
from torch.utils.data import DataLoader

from config import get_config
from tokenizer import CharTokenizer
from dataset import load_records, split_records, ConversationDataset
from model import MiniTransformer
from checkpoint import save_checkpoint, load_checkpoint
from utils import seed_all, choose_device, set_cpu_threads


def run_epoch(model, loader, optimizer, device, accumulation, train=True):
    model.train(train)
    total, count = 0.0, 0
    optimizer.zero_grad(set_to_none=True)
    for step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            _, loss = model(x, y)
            if train:
                (loss / accumulation).backward()
                if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
        total += loss.item()
        count += 1
    return total / max(count, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cpu")
    ap.add_argument("--data", default="data/conversations.json")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--device")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--resume")
    args = ap.parse_args()

    c = get_config(args.config)
    c.epochs = args.epochs or c.epochs
    seed_all(c.seed)
    set_cpu_threads(c.num_threads)
    device = choose_device(args.device or c.device)
    records = load_records(args.data)
    texts = [t["text"] for r in records for t in r["conversation"]]
    tok = CharTokenizer().fit(texts + ["<BOS> <EOS> <USER> <ASSISTANT> <EN>"])
    c.vocab_size = len(tok.stoi)
    tr, va = split_records(records, seed=c.seed)
    train_loader = DataLoader(ConversationDataset(tr, tok, c.context_length), c.batch_size, shuffle=True)
    valid_loader = DataLoader(ConversationDataset(va, tok, c.context_length), c.batch_size, shuffle=False)
    model = MiniTransformer(c).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c.learning_rate, weight_decay=0.1, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, c.epochs), eta_min=c.min_learning_rate)
    start = 0
    if args.resume:
        start = load_checkpoint(args.resume, model, optimizer, device).get("epoch", 0) + 1
    print("device", device, "parameters", MiniTransformer.parameter_report(model))
    print("optimizer AdamW, gradient accumulation", c.grad_accumulation)
    os.makedirs(args.out, exist_ok=True)
    best = float("inf")
    for epoch in range(start, c.epochs):
        train_loss = run_epoch(model, train_loader, optimizer, device, c.grad_accumulation, True)
        with torch.no_grad():
            valid_loss = run_epoch(model, valid_loader, optimizer, device, c.grad_accumulation, False)
        lr = optimizer.param_groups[0]["lr"]
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={valid_loss:.4f} lr={lr:.8f}")
        metrics = {"train_loss": train_loss, "val_loss": valid_loss}
        save_checkpoint(os.path.join(args.out, "last.pt"), model, optimizer, epoch, epoch * len(train_loader), tok, c, metrics)
        if valid_loss < best:
            best = valid_loss
            save_checkpoint(os.path.join(args.out, "best.pt"), model, optimizer, epoch, epoch * len(train_loader), tok, c, metrics)
        scheduler.step()


if __name__ == "__main__":
    main()
