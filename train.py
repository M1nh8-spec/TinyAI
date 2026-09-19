"""Train TinyAI from scratch with minibatch stochastic gradient descent."""
import argparse
import os
import torch
from torch.utils.data import DataLoader

from config import get_config
from tokenizer import CharTokenizer
from dataset import load_records, split_records, ConversationDataset
from model import MiniTransformer
from checkpoint import save_checkpoint, load_checkpoint
from utils import seed_all, choose_device, set_cpu_threads


def run_epoch(model, loader, optimizer, device, train=True):
    model.train(train)
    total = 0.0
    count = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            _, loss = model(x, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # Keep unusually large gradient updates from destabilizing training.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total += loss.item()
        count += 1
    return total / max(count, 1)


def main():
    ap = argparse.ArgumentParser(description="Train TinyAI with SGD.")
    ap.add_argument("--config", default="mini")
    ap.add_argument("--data", default="data/conversations.json")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--device")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--resume")
    args = ap.parse_args()

    config = get_config(args.config)
    config.epochs = args.epochs or config.epochs
    seed_all(config.seed)
    set_cpu_threads(config.num_threads)
    device = choose_device(args.device or config.device)

    records = load_records(args.data)
    tokenizer_text = [
        turn["text"]
        for record in records
        for turn in record["conversation"]
    ] + ["<BOS> <EOS> <USER> <ASSISTANT> <EN>"]
    tok = CharTokenizer().fit(tokenizer_text)
    config.vocab_size = len(tok.stoi)

    train_records, valid_records = split_records(records, seed=config.seed)
    train_data = ConversationDataset(train_records, tok, config.context_length)
    valid_data = ConversationDataset(valid_records, tok, config.context_length)
    train_loader = DataLoader(
        train_data, config.batch_size, shuffle=True, num_workers=config.num_workers
    )
    valid_loader = DataLoader(
        valid_data, config.batch_size, shuffle=False, num_workers=config.num_workers
    )

    model = MiniTransformer(config).to(device)
    # This is minibatch stochastic gradient descent rather than Adam/AdamW.
    # Momentum makes SGD converge more smoothly without changing the model.
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True,
    )
    start = 0
    if args.resume:
        start = load_checkpoint(args.resume, model, optimizer, device).get("epoch", 0) + 1

    print("device", device, "parameters", MiniTransformer.parameter_report(model))
    print("optimizer SGD momentum=0.9 learning_rate", config.learning_rate)
    os.makedirs(args.out, exist_ok=True)
    best = float("inf")

    for epoch in range(start, config.epochs):
        train_loss = run_epoch(model, train_loader, optimizer, device, True)
        valid_loss = run_epoch(model, valid_loader, optimizer, device, False)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={valid_loss:.4f} lr={config.learning_rate}"
        )
        metrics = {"train_loss": train_loss, "val_loss": valid_loss}
        save_checkpoint(
            os.path.join(args.out, "last.pt"),
            model, optimizer, epoch, epoch * len(train_loader), tok, config, metrics
        )
        if valid_loss < best:
            best = valid_loss
            save_checkpoint(
                os.path.join(args.out, "best.pt"),
                model, optimizer, epoch, epoch * len(train_loader), tok, config, metrics
            )


if __name__ == "__main__":
    main()
