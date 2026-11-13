# DumGPT2-124M

A from-scratch GPT-2 124M build: FineWeb-Edu preprocessing, single- and dual-GPU training, checkpointing, metrics, HellaSwag evaluation, and plots.

Three main Python scripts, kept close to the original project structure.

## Training roadmap

I plan to train three models, all released on Hugging Face:

1. **Base** — the 124M model pretrained from scratch on FineWeb-Edu.
2. **RLVR** — the base model further trained with reinforcement learning with verifiable rewards (math/reasoning tasks).
3. **Distillation** — the base model distilled from a stronger teacher model.

## Models on Hugging Face

| Model | Stage | Status | Hugging Face |
| --- | --- | --- | --- |
| DumGPT2-124M | Base (pretrained) | Coming soon | [shubhamisok/DumGPT2-124M](https://huggingface.co/shubhamisok/DumGPT2-124M) |
| DumGPT2-124M-RLVR | RLVR fine-tune | Planned | [shubhamisok/DumGPT2-124M-RLVR](https://huggingface.co/shubhamisok/DumGPT2-124M-RLVR) |
| DumGPT2-124M-Distill | Distilled | Planned | [shubhamisok/DumGPT2-124M-Distill](https://huggingface.co/shubhamisok/DumGPT2-124M-Distill) |

Links stay here once the weights are uploaded, so this table is the canonical place for the model cards.

## Other Hugging Face artifacts

- Profile: [huggingface.co/shubhamisok](https://huggingface.co/shubhamisok)
- Current weights repo: [shubhamisok/dumllm-models](https://huggingface.co/shubhamisok/dumllm-models)
- RLVR math dataset: [shubhamisok/math-500](https://huggingface.co/datasets/shubhamisok/math-500)
- Demo Space: [shubhamisok/dumllm-studio](https://huggingface.co/spaces/shubhamisok/dumllm-studio)

## Upstream resources

- GPT-2 weights used for correctness checks: [openai-community/gpt2](https://huggingface.co/openai-community/gpt2)
- Pretraining data: [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- Evaluation data: [rowanz/hellaswag](https://github.com/rowanz/hellaswag)

## Install
```powershell
uv venv --python 3.12
.venv\Scripts\activate
uv sync
```

Install a CUDA-enabled PyTorch build from https://pytorch.org/get-started/locally/ if needed.

## Preprocess a small test first
```powershell
python fineweb_edu.py --max-documents 1000 --shard-size 1000000
```

## Preprocess a limited token budget
```powershell
python fineweb_edu.py --max-tokens 700000000
```

## Train
```powershell
python train_gpt2.py --max-steps 10 --sequence-length 256 --total-batch-size 1024
```

## Resume
```powershell
python train_gpt2.py --resume checkpoints/latest.pt
```

## Two GPUs
```powershell
torchrun --standalone --nproc_per_node=2 train_gpt2.py
```

Training metrics are written to `outputs/metrics.csv`. Graphs are generated automatically at the end of training in `outputs/plots/`.
