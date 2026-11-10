# DumGPT2-124M

Three main Python scripts, kept close to the original project structure.

## Install
```powershell
uv venv --python 3.11
.venv\Scripts\activate
uv pip install -r requirements.txt
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
