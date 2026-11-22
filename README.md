# DumGPT2-124M

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-ee4c2c.svg)](https://pytorch.org/)
[![Model](https://img.shields.io/badge/model-GPT--2%20124M-6f42c1.svg)](#training-roadmap)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-ershubhamkrsahu-yellow.svg)](https://huggingface.co/ershubhamkrsahu)

A from-scratch GPT-2 124M language model: FineWeb-Edu preprocessing, single- and dual-GPU training, checkpointing, metrics, HellaSwag evaluation, and plots.

**The base model is trained by me from scratch.** It starts from random initialization and is pretrained on FineWeb-Edu — no OpenAI GPT-2 weights are loaded or fine-tuned. "GPT-2 124M" refers to the architecture and scale, not to a pretrained checkpoint.

## Contents

- [Training roadmap](#training-roadmap)
- [Models on Hugging Face](#models-on-hugging-face)
- [Hugging Face artifacts](#hugging-face-artifacts)
- [Repository layout](#repository-layout)
- [Setup](#setup)
- [Data preparation](#data-preparation)
- [Training](#training)
- [Monitoring and outputs](#monitoring-and-outputs)
- [Evaluation](#evaluation)
- [Interactive notebook & training runs](#interactive-notebook--training-runs)
- [Interactive UI & free hosting](#interactive-ui--free-hosting)
- [Reproducibility notes](#reproducibility-notes)
- [Acknowledgements](#acknowledgements)

## Training roadmap

Three models are planned, all released on Hugging Face:

1. **Base** — a 124M model trained from scratch by me on FineWeb-Edu.
2. **RLVR** — the base model further trained with reinforcement learning with verifiable rewards, starting from the [`math-500`](https://huggingface.co/datasets/ershubhamkrsahu/math-500) dataset.
3. **Distillation** — the base model distilled from a stronger teacher model.

## Models on Hugging Face

| Model | Stage | Status | Hugging Face |
| --- | --- | --- | --- |
| DumGPT-124M | Base, trained from scratch (1k-step release) | Available | [ershubhamkrsahu/DumGPT-124M](https://huggingface.co/ershubhamkrsahu/DumGPT-124M) |
| DumGPT2-124M-RLVR | RLVR fine-tune | Planned | [ershubhamkrsahu/DumGPT2-124M-RLVR](https://huggingface.co/ershubhamkrsahu/DumGPT2-124M-RLVR) |
| DumGPT2-124M-Distill | Distilled | Planned | [ershubhamkrsahu/DumGPT2-124M-Distill](https://huggingface.co/ershubhamkrsahu/DumGPT2-124M-Distill) |

Links stay in this table once the weights are uploaded, so it remains the canonical index for the model cards.

## Hugging Face artifacts

- Profile: [huggingface.co/ershubhamkrsahu](https://huggingface.co/ershubhamkrsahu)
- Base weights repository: [ershubhamkrsahu/DumGPT-124M](https://huggingface.co/ershubhamkrsahu/DumGPT-124M) (`pytorch_model.bin`, `dumgpt2-base.pt`, `outputs/`)
- RLVR math dataset: [ershubhamkrsahu/math-500](https://huggingface.co/datasets/ershubhamkrsahu/math-500)
- Interactive demo Space: [ershubhamkrsahu/dumgpt-studio](https://huggingface.co/spaces/ershubhamkrsahu/dumgpt-studio)

### Upstream references

- GPT-2 architecture and original weights: [openai-community/gpt2](https://huggingface.co/openai-community/gpt2)
- Pretraining data: [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (`sample-10BT`)
- Evaluation data: [rowanz/hellaswag](https://github.com/rowanz/hellaswag)

## Repository layout

```text
DumGPT2_124M/
├── app/                 # Gradio Web UI for Hugging Face Spaces and local inference
│   ├── app.py           # Gradio app with streaming text generation & parameter controls
│   ├── model.py         # Self-contained GPT-2 inference engine & HF Hub weight loader
│   ├── requirements.txt # Space runtime dependencies
│   └── README.md        # Hugging Face Space metadata card
├── images/              # Training curves and metrics visualization plots
│   ├── train_loss.png   # Training loss progression (10.8 -> 6.48)
│   ├── val_loss.png     # Validation loss curve (eval on held-out shard)
│   ├── learning_rate.png# Cosine decay schedule with warmup
│   ├── grad_norm.png    # Gradient norm stability with clipping at 1.0
│   └── tokens_per_sec.png # Throughput over training steps
├── notebooks/
│   └── DumGPT_124M.ipynb # Interactive training, validation, sampling & one-click HF upload
├── fineweb_edu.py       # Tokenize FineWeb-Edu into uint16 binary shards
├── train_gpt2.py        # From-scratch GPT-2 124M training (AMP, DDP, checkpoints)
├── hellaswag.py         # HellaSwag download, rendering, and scoring
├── tinyshakespeare.py   # Downloads Tiny Shakespeare for small local experiments
├── utils/
│   └── check_gpu.py     # CUDA, FP16, and BF16 sanity check
├── my_view.md           # Developer journal and architectural learnings
├── pyproject.toml       # Dependencies and uv configuration
└── uv.lock              # Pinned dependency tree
```

## Setup

Requires Python 3.12 (pinned in `.python-version`), [uv](https://docs.astral.sh/uv/), and a CUDA-capable GPU for training. A CUDA-enabled PyTorch build is selected through `pyproject.toml`; see [pytorch.org](https://pytorch.org/get-started/locally/) if you need a different CUDA version.

```powershell
git clone https://github.com/ershubhamkrsahu/DumGPT2_124M.git
cd DumGPT2_124M
uv venv --python 3.12
.venv\Scripts\activate
uv sync
```

On Linux or macOS, activate with `source .venv/bin/activate`.

Verify the environment before starting a long run:

```powershell
python utils/check_gpu.py
```

Training uses BF16 when the GPU supports it, otherwise FP16 with a gradient scaler. CPU training runs in FP32.

## Data preparation

`fineweb_edu.py` streams `HuggingFaceFW/fineweb-edu` (`sample-10BT`), tokenizes each document with the GPT-2 `tiktoken` encoding, and packs the IDs into `uint16` shards. Shard `000000` is the validation set; every later shard is training data.

Always run a small preprocessing pass first to confirm the pipeline works:

```powershell
python fineweb_edu.py --max-documents 1000 --shard-size 1000000
```

Then choose a token budget:

```powershell
# ~700M tokens
python fineweb_edu.py --max-tokens 700000000

# Full sample-10BT (~10B tokens, ~20 GB of shards)
python fineweb_edu.py
```

Output:

```text
edu_fineweb10B/
├── edu_fineweb_val_000000.bin
├── edu_fineweb_train_000001.bin
├── edu_fineweb_train_000002.bin
└── ...
```

Each 100M-token shard is about 200 MB. Existing shards are never overwritten unless `--overwrite` is passed.

| Option | Default | Description |
| --- | --- | --- |
| `--output-dir` | `edu_fineweb10B` | Directory for the binary shards |
| `--shard-size` | `100000000` | Tokens per shard |
| `--workers` | up to 8 | Parallel tokenization workers |
| `--max-documents` | all | Stop after this many documents |
| `--max-tokens` | all | Stop after this many tokens |
| `--overwrite` | off | Replace existing shards |

## Training

### Quick smoke test

```powershell
python train_gpt2.py --max-steps 10 --sequence-length 256 --total-batch-size 1024
```

### Standard run

```powershell
python train_gpt2.py
```

The model is randomly initialized and trained from scratch. The default schedule is sized for a 4 GB laptop GPU: one micro-batch of `1 x 512` tokens, gradient accumulation up to an 8,192-token effective batch, 19,073 steps, and a cosine learning-rate decay.

| Option | Default | Description |
| --- | --- | --- |
| `--data-dir` | `edu_fineweb10B` | Directory with the token shards |
| `--max-steps` | `19073` | Total optimization steps |
| `--batch-size` | `1` | Micro-batch size per GPU |
| `--sequence-length` | `512` | Tokens per training sample |
| `--total-batch-size` | `8192` | Effective batch size in tokens |
| `--resume` | none | Checkpoint to continue from |
| `--hellaswag-interval` | `0` | HellaSwag eval period; `0` disables it |
| `--hellaswag-max-examples` | `500` | Examples per HellaSwag evaluation |

The effective batch is `batch_size x sequence_length x grad_accum x world_size`. With the defaults above, gradient accumulation is 16 micro-steps. Note that 19,073 steps at 8,192 tokens per step covers roughly 156M tokens — a local starting budget, not the full 10B-token dataset. Increase the effective batch or the step count for a full pretraining run.

### Resume

```powershell
python train_gpt2.py --resume checkpoints/latest.pt
```

Checkpoints contain the step, model, optimizer, scaler, and model config. They are written to a temporary file and atomically renamed, so an interrupted save does not corrupt `latest.pt`.

### Multiple GPUs

```powershell
torchrun --standalone --nproc_per_node=2 train_gpt2.py
```

This uses PyTorch DDP with one process per GPU. Gradients are synchronized only on the final micro-step of each accumulation cycle.

## Monitoring and outputs

Every run writes to `outputs/` and `checkpoints/`:

| Path | Contents |
| --- | --- |
| `checkpoints/latest.pt` | Resumable checkpoint, saved every 500 steps |
| `outputs/metrics.csv` | Per-step training and evaluation metrics |
| `outputs/plots/*.png` | Graphs generated at the end of training |

`metrics.csv` columns: `step`, `train_loss`, `val_loss`, `learning_rate`, `grad_norm`, `step_ms`, `tokens_per_sec`, `hellaswag_accuracy`. Validation loss is measured every 250 steps over 10 batches. Every 500 steps the model also generates a 30-token sample from the prompt `Once upon a time, in a land far away,` so progress can be checked by eye.

At the end of training, `outputs/plots/` contains loss curves, learning rate, gradient norm, throughput, and HellaSwag accuracy (when available).

## Evaluation

HellaSwag is disabled by default because it slows down short runs. Enable periodic evaluation with:

```powershell
python train_gpt2.py --hellaswag-interval 250 --hellaswag-max-examples 500
```

`hellaswag.py` downloads the validation split once, renders each question as four candidate completions, and scores them by token loss. The cache lives in `hellaswag/` and is ignored by git.

## Interactive notebook & training runs

The full pretraining and export pipeline is packaged in [`notebooks/DumGPT_124M.ipynb`](notebooks/DumGPT_124M.ipynb) for interactive local or cloud execution (Jupyter, Kaggle, Colab):

- **Data streaming & tokenization**: Streams `FineWeb-Edu`, encodes via `tiktoken`, and writes memory-mapped `uint16` shards.
- **Pretraining execution**: Runs the 124M model with Flash Attention (SDPA), fused AdamW, cosine LR schedule ($6 \times 10^{-4} \to 6 \times 10^{-5}$), and gradient accumulation.
- **Resuming & Checkpoints**: Automatically checks for `checkpoints/latest.pt` and resumes step, model state, optimizer moments, and scaler state.
- **Evaluation & Sample generation**: Evaluates validation loss and generates live text samples every 250 steps.
- **Hugging Face Hub export**: Packages `pytorch_model.bin`, `config.json`, `README.md`, resumable checkpoint `checkpoints/dumgpt2-base.pt`, and `outputs/` (metrics and plots) directly to Hugging Face (`ershubhamkrsahu/DumGPT-124M`).

### Milestone: 1,000-Step Pretraining Run

- **Start**: Initialized from random Gaussian weights with cross-entropy loss ~10.8 ($\ln(50,304)$).
- **Step 200**: Resumed with training loss ~6.85, validation loss 7.04.
- **Step 1,000**: Final training loss ~6.48, validation loss 6.4468, throughput ~7.8k tokens/sec on GPU.
- **Generations**: Progressed from random noise to coherent grammar and English sentence structures.

### Training metrics & curves

| Training Loss | Validation Loss |
| :---: | :---: |
| ![Training Loss](images/train_loss.png) | ![Validation Loss](images/val_loss.png) |

| Learning Rate Schedule | Gradient Norm |
| :---: | :---: |
| ![Learning Rate](images/learning_rate.png) | ![Gradient Norm](images/grad_norm.png) |

| Training Throughput (Tokens / sec) |
| :---: |
| ![Tokens per second](images/tokens_per_sec.png) |

## Interactive UI & free hosting

An interactive web UI powered by [Gradio](https://gradio.app/) is available in the [`app/`](app/) directory:

- **Features**: Interactive text completion playground with streaming tokens, sliders for Temperature ($0.1 - 2.0$), Top-K ($1 - 100$), Max Tokens ($10 - 256$), and Seed, plus model specs and training curves.
- **Weight Loading**: Automatically pulls `pytorch_model.bin` from [Hugging Face `ershubhamkrsahu/DumGPT-124M`](https://huggingface.co/ershubhamkrsahu/DumGPT-124M).

### Running locally

```powershell
pip install gradio tiktoken huggingface_hub torch
python app/app.py
```
Open `http://127.0.0.1:7860` in your browser.

### Hosting 100% free on Hugging Face Spaces

1. Create a new Space on [huggingface.co/new-space](https://huggingface.co/new-space) with the name `dumgpt-studio`.
2. Select **Gradio** as the Space SDK and choose the **Free CPU (2 vCPU, 16 GB RAM)** hardware tier.
3. Push or upload the contents of the [`app/`](app/) directory (`app.py`, `model.py`, `requirements.txt`, `README.md`).
4. Hugging Face Spaces builds the container and serves your model 24/7 at no cost.

## Reproducibility notes

- Seeds are fixed at `1337 + rank` for Python, NumPy, and PyTorch.
- `uv.lock` pins the full dependency tree; `pyproject.toml` pins `torch==2.14.1`.
- Mixed precision is chosen per device: BF16 when supported, otherwise FP16 with a scaler, and FP32 on CPU.
- Gradient checkpointing is enabled by default to reduce VRAM use at the cost of some speed.
- `my_view.md` records the reasoning behind each phase of the build.

## Acknowledgements

The pipeline follows the structure of Andrej Karpathy's GPT-2 training scripts ([nanoGPT](https://github.com/karpathy/nanoGPT), [llm.c](https://github.com/karpathy/llm.c)). The training data and tokenizer come from Hugging Face.
