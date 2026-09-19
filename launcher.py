"""Single-file graphical launcher for the trained neural chatbot.

Run:
    python launcher.py

The only non-neural command is !stop, which closes the interface. Every other
message is passed through tokenization, the Transformer, and sampling.
"""
import argparse
import os
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext

import torch

from config import get_config
from model import MiniTransformer
from tokenizer import CharTokenizer
from generate import generate


class ChatWindow:
    def __init__(self, root, model, tokenizer, device, settings):
        self.root = root
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.settings = settings
        self.busy = False

        root.title("TinyAI — Mini-Transformer")
        root.geometry("760x560")
        root.minsize(520, 380)

        self.transcript = scrolledtext.ScrolledText(
            root, wrap=tk.WORD, state="disabled", font=("Arial", 11)
        )
        self.transcript.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 6))

        bottom = tk.Frame(root)
        bottom.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.entry = tk.Entry(bottom, font=("Arial", 11))
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.entry.bind("<Return>", self.submit)
        self.entry.focus_set()
        self.send_button = tk.Button(bottom, text="Send", width=10, command=self.submit)
        self.send_button.pack(side=tk.LEFT, padx=(6, 0))

        self.write("TinyAI is ready. Type a message; type !stop to exit.\n", "system")

    def write(self, text, tag=None):
        self.transcript.configure(state="normal")
        self.transcript.insert(tk.END, text, tag)
        self.transcript.configure(state="disabled")
        self.transcript.see(tk.END)

    def submit(self, _event=None):
        if self.busy:
            return "break"
        prompt = self.entry.get().strip()
        if not prompt:
            return "break"
        self.entry.delete(0, tk.END)
        if prompt == "!stop":
            self.root.destroy()
            return "break"

        self.write(f"You: {prompt}\n", "user")
        self.busy = True
        self.entry.configure(state="disabled")
        self.send_button.configure(state="disabled")
        threading.Thread(target=self.respond, args=(prompt,), daemon=True).start()
        return "break"

    def respond(self, prompt):
        try:
            # The prompt is serialized as context. No response dictionary or
            # retrieval is used; generate() samples from Transformer logits.
            formatted = f"<USER> {prompt} <ASSISTANT>"
            answer = generate(
                self.model,
                self.tokenizer,
                formatted,
                self.device,
                max_new_tokens=self.settings["max_new_tokens"],
                temperature=self.settings["temperature"],
                top_k=self.settings["top_k"],
                top_p=self.settings["top_p"],
            )
            # Remove only the visible prompt prefix; the generated text itself
            # remains entirely model-produced.
            if answer.startswith(formatted):
                answer = answer[len(formatted):]
            answer = answer.strip()
            if not answer:
                answer = "[The model generated an empty continuation.]"
            self.root.after(0, self.finish, answer)
        except Exception as exc:
            self.root.after(0, self.fail, str(exc))

    def finish(self, answer):
        self.write(f"AI: {answer}\n\n", "assistant")
        self.busy = False
        self.entry.configure(state="normal")
        self.send_button.configure(state="normal")
        self.entry.focus_set()

    def fail(self, error):
        self.write(f"Launcher error: {error}\n\n", "system")
        self.busy = False
        self.entry.configure(state="normal")
        self.send_button.configure(state="normal")
        self.entry.focus_set()


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = get_config("mini")
    config.__dict__.update(checkpoint["config"])
    tokenizer = CharTokenizer(checkpoint["tokenizer"])
    model = MiniTransformer(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Launch the trained TinyAI chatbot GUI")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        raise SystemExit(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Train first, for example: python train.py --config tiny --epochs 30"
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable. Run with --device cpu.")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    model, tokenizer = load_model(args.checkpoint, device)
    root = tk.Tk()
    root.option_add("*Font", "Arial 11")
    ChatWindow(root, model, tokenizer, device, vars(args))
    root.mainloop()


if __name__ == "__main__":
    main()
