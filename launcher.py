"""One-command terminal launcher for TinyAI.

Run:
    py launcher.py

The launcher creates/uses the local virtual environment named ``lib``,
installs requirements.txt into it, trains the default model if needed, and
starts terminal chat. Use --cpu or --cuda to choose the device.
"""
import argparse
import os
import subprocess
import sys
import venv

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(PROJECT_DIR, "lib")
REQUIREMENTS = os.path.join(PROJECT_DIR, "requirements.txt")


def venv_python():
    if os.name == "nt":
        return os.path.join(VENV_DIR, "Scripts", "python.exe")
    return os.path.join(VENV_DIR, "bin", "python")


def ensure_local_environment():
    """Set up lib once, then run exactly one launcher process inside it.

    subprocess.run is used instead of os.execv because Windows can leave the
    parent process alive during an exec-style restart, creating two training
    processes and allowing terminal input to be mixed with startup output.
    """
    executable = venv_python()
    in_local_venv = os.path.abspath(sys.prefix) == os.path.abspath(VENV_DIR)
    if in_local_venv:
        return

    if not os.path.isfile(executable):
        print(f"Creating local virtual environment: {VENV_DIR}", flush=True)
        venv.EnvBuilder(with_pip=True, clear=False).create(VENV_DIR)

    print("Checking/installing dependencies in lib ...", flush=True)
    result = subprocess.run(
        [executable, "-m", "pip", "install", "-r", REQUIREMENTS],
        cwd=PROJECT_DIR,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(result.returncode)

    result = subprocess.run(
        [executable, os.path.abspath(__file__), *sys.argv[1:]],
        cwd=PROJECT_DIR,
        check=False,
    )
    raise SystemExit(result.returncode)


ensure_local_environment()

import torch  # noqa: E402

from config import get_config  # noqa: E402
from generate import generate  # noqa: E402
from model import MiniTransformer  # noqa: E402
from tokenizer import CharTokenizer  # noqa: E402

DEFAULT_CONFIG = "mini"
DEFAULT_EPOCHS = 20
DEFAULT_CHECKPOINT = os.path.join(PROJECT_DIR, "checkpoints", "best.pt")
DEFAULT_MAX_NEW_TOKENS = 120
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_K = 20
DEFAULT_TOP_P = 0.9


def train_if_needed(checkpoint_path, device):
    if os.path.isfile(checkpoint_path):
        return
    print(f"No checkpoint found. Training the default {DEFAULT_CONFIG} model on {device}...", flush=True)
    result = subprocess.run(
        [
            sys.executable, "train.py", "--config", DEFAULT_CONFIG,
            "--epochs", str(DEFAULT_EPOCHS), "--device", str(device),
            "--out", os.path.dirname(checkpoint_path),
        ],
        cwd=PROJECT_DIR,
        check=False,
    )
    if result.returncode != 0 or not os.path.isfile(checkpoint_path):
        raise SystemExit("Training failed or did not create the expected checkpoint.")


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = get_config(DEFAULT_CONFIG)
    config.__dict__.update(checkpoint["config"])
    tokenizer = CharTokenizer(checkpoint["tokenizer"])
    model = MiniTransformer(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, tokenizer


def choose_device(args):
    if args.cuda:
        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested, but no CUDA device is available.")
        return torch.device("cuda")
    if args.cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(description="Set up, train if needed, and run TinyAI.")
    devices = parser.add_mutually_exclusive_group()
    devices.add_argument("--cpu", action="store_true", help="force CPU training and inference")
    devices.add_argument("--cuda", action="store_true", help="use CUDA training and inference")
    args = parser.parse_args()

    device = choose_device(args)
    train_if_needed(DEFAULT_CHECKPOINT, device)
    model, tokenizer = load_model(DEFAULT_CHECKPOINT, device)
    print(f"TinyAI ready on {device}. Type !stop to exit.", flush=True)

    history = []
    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if prompt == "!stop":
            print("Goodbye.")
            break
        if not prompt:
            continue

        history.append(f"<USER> {prompt} <ASSISTANT>")
        context = " ".join(history)
        answer = generate(
            model, tokenizer, context, device,
            max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
            temperature=DEFAULT_TEMPERATURE,
            top_k=DEFAULT_TOP_K,
            top_p=DEFAULT_TOP_P,
        )
        if answer.startswith(context):
            answer = answer[len(context):]
        answer = answer.strip() or "[The model generated an empty continuation.]"
        print(f"AI: {answer}")
        history.append(answer)


if __name__ == "__main__":
    main()
