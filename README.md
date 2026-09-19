# TinyAI: from-scratch bilingual Mini-Transformer

This repository is a small educational **decoder-only causal Transformer** for English and Vietnamese. It uses a project-built Unicode character tokenizer and randomly initialized PyTorch weights. It is CPU-first and does not call external inference services or load pretrained artifacts.

## Install

Use Python 3.10+ and a virtual environment:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
```

## Run the checks first

```bash
python benchmark.py --steps 30
```
This performs real forward/backward updates and prints the measured timings on the machine where it runs. The repository does not claim results that have not been executed.

## Train

```bash
python train.py --config tiny --epochs 30
# or the default mini configuration
python train.py --config mini --epochs 20
```
Checkpoints are written to `checkpoints/last.pt` and `checkpoints/best.pt`. Resume with:

```bash
python train.py --config mini --resume checkpoints/last.pt
```

The sample data is intentionally small, hand-written demonstration data. Add legally usable records to `data/conversations.json`; the same pipeline supports larger datasets. `data/translations.json` is included as translation material and can be converted into conversation records for training. No claims are made that this sample resembles a large corpus.

## Chat and evaluation

```bash
python generate.py --checkpoint checkpoints/best.pt --prompt "<USER> Could you tell me your name? <ASSISTANT>"
python generate.py --checkpoint checkpoints/best.pt
python evaluate.py --checkpoint checkpoints/best.pt
```
Generation feeds every sampled token back into the model and supports temperature, top-k, and top-p through the `generate()` function. Conversation history is represented by the serialized token context, not a dictionary memory. The response is not selected from the dataset.

## Architecture and sizing

`model.py` contains token and positional embeddings, pre-normalized residual blocks, multi-head self-attention, a lower-triangular causal mask, GELU feed-forward layers, layer normalization, tied output projection, logits, and cross-entropy loss. All parameters are initialized by PyTorch's normal layer initialization.

`Config` exposes vocabulary size, model width, layer/head counts, feed-forward width, context, batch size, learning rate, dropout, CPU threads, and workers. `tiny`, `mini`, and `large_mini` are selectable; actual counts and memory estimates are printed at training startup by `parameter_report()`. Memory numbers are estimates based on parameter counts, not fabricated measurements.

The character tokenizer preserves Vietnamese diacritics and arbitrary Unicode without a downloaded vocabulary. Character tokenization is simple and robust but less compact than subwords, so context is consumed faster.

## Honest limitations

This sample is too small to produce reliable open-ended conversation or translation. It is suitable for verifying tokenization, tensor shapes, causal masking, loss, backpropagation, checkpointing, and autoregressive generation. Unseen-prompt quality, generalization, training duration, RAM use, and i5-3230M throughput must be measured on the target computer; they are **not tested here**. A tiny synthetic/hand-written dataset cannot establish broad language competence.

For CPU use, keep `num_threads` modest and `num_workers=0`. CUDA is selected only when explicitly requested or available in `auto` mode; no downloaded model or weight is involved.
