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

## Quick start: one command

Run the terminal chatbot:

```bash
python launcher.py
```

`launcher.py` automatically checks for `checkpoints/best.pt`. If it does not exist, it runs the normal training pipeline first using the default Mini configuration and 20 epochs. After training or loading the checkpoint, it starts a continuous terminal conversation:

```text
TinyAI ready on cpu. Type !stop to exit.
You: 
```

Type a message and press Enter. The generated response is printed, followed by another `You:` prompt. Type `!stop` to exit. `Ctrl+C` and end-of-file also end the session cleanly.

The default automatic training can take time on a low-power CPU. To change the number of automatic training epochs:

```bash
python launcher.py --train-epochs 30
```

To force CPU execution:

```bash
python launcher.py --device cpu
```

To request CUDA when available:

```bash
python launcher.py --device cuda
```

## Run the benchmark first

Before a larger training run, test forward propagation, backward propagation, parameter updates, and overfitting behavior on a tiny synthetic batch:

```bash
python benchmark.py --steps 30
```

The script prints measurements from the machine where it is actually run. No benchmark values are claimed in this README because hardware and execution time depend on the user's computer.

## Manual training

You can train without the launcher:

```bash
python train.py --config tiny --epochs 30
python train.py --config mini --epochs 20
```

The available configurations are:

- `tiny`: smaller and faster for testing
- `mini`: default CPU-oriented configuration
- `large_mini`: larger experiment; it may be slow on an older CPU

Checkpoints are written to:

```text
checkpoints/last.pt
checkpoints/best.pt
```

Resume an interrupted run:

```bash
python train.py --config mini --resume checkpoints/last.pt
```

Training prints epoch, training loss, validation loss, and learning rate. The checkpoint stores model weights, optimizer state, epoch, step, configuration, tokenizer vocabulary, and metrics.

## Other generation and evaluation commands

Generate one response from a prompt:

```bash
python generate.py --checkpoint checkpoints/best.pt --prompt "<USER> Could you tell me your name? <ASSISTANT>"
```

Start the older interactive generation script:

```bash
python generate.py --checkpoint checkpoints/best.pt
```

Evaluate validation loss and perplexity:

```bash
python evaluate.py --checkpoint checkpoints/best.pt
```

Generation supports temperature, top-k, top-p, and maximum generated tokens through the `generate()` function and its command-line options.

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

`model.py` implements:

- token embeddings;
- learned positional embeddings;
- multi-head self-attention;
- a lower-triangular causal attention mask;
- pre-normalized residual connections;
- GELU feed-forward networks;
- layer normalization;
- tied output projection;
- logits and cross-entropy loss.

At inference time, the model generates autoregressively: it tokenizes the prompt, predicts logits for the next token, applies sampling, appends the sampled token, and repeats. Responses are not selected from the dataset and are not chosen by keyword or intent rules. The only special command is `!stop`, which exits the terminal loop.

The tokenizer in `tokenizer.py` is built from the project data and works at Unicode character level. This keeps Vietnamese tone marks and characters such as `ă`, `â`, `ê`, `ô`, `ơ`, and `ư` intact. Character tokenization is simple and robust, but less compact than subword tokenization, so context is consumed more quickly.

## Parameter and memory estimates

The model configuration exposes vocabulary size, model width, number of layers, number of heads, feed-forward width, context length, batch size, learning rate, dropout, CPU threads, and DataLoader workers.

Training prints the actual total and trainable parameter counts using `MiniTransformer.parameter_report()`, along with estimates for parameter, Adam optimizer, and training memory. These are estimates based on tensor sizes, not measurements of peak process RAM.

Use modest CPU settings on an older computer:

```text
num_threads = 2
num_workers = 0
```

Reduce batch size, context length, model width, or layer count if training is too slow or memory-heavy.

## Honest status and limitations

### Implemented

- From-scratch randomly initialized Transformer
- CPU-first PyTorch training
- Vietnamese and English sample data
- Unicode character tokenizer
- Cross-entropy training with backpropagation
- Validation split and checkpoint/resume support
- Autoregressive sampling
- Terminal launcher with automatic first training run
- Tiny forward/backward benchmark

### Not claimed without running the code

Actual loss values, training speed, peak RAM, CPU duration, generalization quality, and translation quality depend on the target machine and the training run. They are **not tested or fabricated here**. Run `benchmark.py`, `train.py`, and `evaluate.py` locally to obtain those results.

Because the sample dataset is small, the model may memorize examples, produce malformed text, repeat tokens, or fail on unseen prompts. Successful training and a decreasing loss do not prove broad language understanding.
