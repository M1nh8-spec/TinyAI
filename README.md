# TinyAI: from-scratch bilingual Mini-Transformer

TinyAI is a small educational **decoder-only causal Transformer** for English and Vietnamese. It uses a project-built Unicode character tokenizer and randomly initialized PyTorch weights. The model is trained only on the project's data; it does not use pretrained models, downloaded weights, external inference services, or response retrieval.

This is a research/learning project, not a ChatGPT replacement. The included dataset is intentionally small, so the model's conversational and translation quality will be limited.

## Requirements

- Python 3.10 or newer
- PyTorch 2.1 or newer
- A CPU is supported and is the default target
- CUDA is optional when a compatible PyTorch installation and NVIDIA GPU are available

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run TinyAI

For normal users, run one command:

```bash
python launcher.py
```

The launcher has training settings built into `launcher.py`. It automatically checks for `checkpoints/best.pt`. If the checkpoint does not exist, it runs the default `train.py` pipeline using the Mini configuration for the default number of epochs, then starts the terminal chatbot. No separate training command or configuration command is required.

The only command-line choice is the compute device:

```bash
python launcher.py --cpu
python launcher.py --cuda
```

With no switch, TinyAI uses CUDA when it is available and otherwise uses the CPU. `--cuda` stops with a clear error if CUDA is unavailable. `--cpu` forces both automatic training and inference to use the CPU.

After startup:

```text
TinyAI ready on cpu. Type !stop to exit.
You: 
```

Type a message and press Enter. The generated response is printed, followed by another `You:` prompt. Type `!stop` to exit. `Ctrl+C` and end-of-file also end the session cleanly.

The first launch may take time because the model must train before it can answer. Later launches reuse `checkpoints/best.pt` and do not train again unless that file is removed.

## Run the development checks

Developers can run the tiny forward/backward benchmark directly:

```bash
python benchmark.py --steps 30
```

This is optional and is not needed for normal users. It prints measurements from the machine where it is actually run. No benchmark values are claimed in this README because hardware and execution time depend on the user's computer.

## Dataset

The sample data is hand-written demonstration data in:

- `data/conversations.json`: English and Vietnamese multi-turn conversations
- `data/translations.json`: complete-sentence Vietnamese-English translation examples
- `data/names.json`: sample names for dataset expansion

The dataset includes greetings, introductions, help, arithmetic, colors, sky, family, computers, programming, daily activities, and good morning/night examples. Add legally usable records to `data/conversations.json` to expand it. Keep the same structure:

```json
{
  "language": "vi",
  "conversation": [
    {"role": "user", "text": "Tên bạn là gì?"},
    {"role": "assistant", "text": "Mình là một chatbot mini."}
  ]
}
```

Conversation turns are serialized with markers such as `<BOS>`, `<USER>`, `<ASSISTANT>`, language markers, and `<EOS>`. Conversation history is provided to the Transformer as token context; it is not stored as a keyword-to-answer dictionary.

The current dataset is small and mostly hand-written. It is not a large real-world corpus, and synthetic or demonstration data should not be interpreted as equivalent to broad language training data.

## Architecture

`model.py` implements token embeddings, learned positional embeddings, multi-head self-attention, a lower-triangular causal attention mask, pre-normalized residual connections, GELU feed-forward networks, layer normalization, a tied output projection, logits, and cross-entropy loss.

At inference time, the model generates autoregressively: it tokenizes the prompt, predicts logits for the next token, applies sampling, appends the sampled token, and repeats. Responses are not selected from the dataset and are not chosen by keyword or intent rules. The only special command is `!stop`, which exits the terminal loop.

The tokenizer in `tokenizer.py` is built from the project data and works at Unicode character level. This keeps Vietnamese tone marks and characters such as `ă`, `â`, `ê`, `ô`, `ơ`, and `ư` intact. Character tokenization is simple and robust, but less compact than subword tokenization, so context is consumed more quickly.

## Honest status and limitations

The project implements a real randomly initialized Transformer, CPU-first PyTorch training, Vietnamese and English sample data, Unicode tokenization, cross-entropy training, validation/checkpoints, autoregressive sampling, and the one-command terminal launcher.

Actual loss values, training speed, peak RAM, CPU duration, generalization quality, and translation quality depend on the target machine and the training run. They are **not tested or fabricated here**. Because the sample dataset is small, the model may memorize examples, produce malformed text, repeat tokens, or fail on unseen prompts. Successful training and a decreasing loss do not prove broad language understanding.
