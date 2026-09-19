"""One-command terminal launcher for TinyAI."""
import argparse
import os
import subprocess
import sys
import venv

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(PROJECT_DIR, "lib")
REQUIREMENTS = os.path.join(PROJECT_DIR, "requirements.txt")
DEFAULT_CONFIG = "mini"
DEFAULT_EPOCHS = 40
DEFAULT_CHECKPOINT = os.path.join(PROJECT_DIR, "checkpoints", "best.pt")


def venv_python():
    return os.path.join(VENV_DIR, "Scripts", "python.exe") if os.name == "nt" else os.path.join(VENV_DIR, "bin", "python")


def ensure_local_environment():
    executable = venv_python()
    if os.path.abspath(sys.prefix) == os.path.abspath(VENV_DIR):
        return
    if not os.path.isfile(executable):
        print(f"Creating local virtual environment: {VENV_DIR}", flush=True)
        venv.EnvBuilder(with_pip=True, clear=False).create(VENV_DIR)
    print("Checking/installing dependencies in lib ...", flush=True)
    result = subprocess.run([executable, "-m", "pip", "install", "-r", REQUIREMENTS], cwd=PROJECT_DIR)
    if result.returncode:
        raise SystemExit(result.returncode)
    result = subprocess.run([executable, os.path.abspath(__file__), *sys.argv[1:]], cwd=PROJECT_DIR)
    raise SystemExit(result.returncode)


ensure_local_environment()

import torch
from config import get_config
from generate import generate
from model import MiniTransformer
from tokenizer import CharTokenizer


def choose_device(args):
    if args.cuda:
        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested, but no CUDA device is available.")
        return torch.device("cuda")
    if args.cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_if_needed(device):
    if os.path.isfile(DEFAULT_CHECKPOINT):
        return
    print(f"No checkpoint found. Training the default {DEFAULT_CONFIG} model on {device}...", flush=True)
    result = subprocess.run([
        sys.executable, "train.py", "--config", DEFAULT_CONFIG,
        "--epochs", str(DEFAULT_EPOCHS), "--device", str(device),
        "--out", os.path.dirname(DEFAULT_CHECKPOINT)
    ], cwd=PROJECT_DIR)
    if result.returncode or not os.path.isfile(DEFAULT_CHECKPOINT):
        raise SystemExit("Training failed or did not create the expected checkpoint.")


def load_model(device):
    checkpoint = torch.load(DEFAULT_CHECKPOINT, map_location=device)
    config = get_config(DEFAULT_CONFIG)
    config.__dict__.update(checkpoint["config"])
    tok = CharTokenizer(checkpoint["tokenizer"])
    model = MiniTransformer(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, tok


def main():
    parser = argparse.ArgumentParser(description="Train if needed, then run TinyAI.")
    devices = parser.add_mutually_exclusive_group()
    devices.add_argument("--cpu", action="store_true")
    devices.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    device = choose_device(args)
    train_if_needed(device)
    model, tok = load_model(device)
    print(f"TinyAI ready on {device}. Type !stop to exit.", flush=True)
    history = []
    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return
        if prompt == "!stop":
            print("Goodbye.")
            return
        if not prompt:
            continue
        history.append(f"<USER> {prompt} <ASSISTANT>")
        # Keep the entire recent dialogue inside the model's context budget.
        max_chars = max(256, model.config.context_length * 3)
        context = " ".join(history)[-max_chars:]
        answer = generate(model, tok, context, device, max_new_tokens=120,
                          temperature=.75, top_k=25, top_p=.9,
                          repetition_penalty=1.18, frequency_penalty=.025)
        if answer.startswith(context):
            answer = answer[len(context):]
        answer = answer.split("<USER>", 1)[0].strip()
        print(f"AI: {answer or '[The model generated an empty continuation.]'}")
        history.append(answer)


if __name__ == "__main__":
    main()
