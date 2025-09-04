# Finetuning Bootcamp (7 Days)

This starter pack contains **code skeletons with TODOs** you will complete as you learn fine‑tuning from fundamentals to advanced parameter‑efficient methods.

## How to use
- You can work locally (Python 3.10+) or on **Google Colab Free** (recommended for GPU).
- Create a new Colab notebook, upload a folder's files, and run through the README for that day, or clone/download this whole pack.
- We deliberately **do not give final code**. You will fill in the TODOs and test.

## Install (local / Colab)
```bash
# minimal shared tools (install per day as needed)
pip install -U numpy matplotlib scikit-learn torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -U datasets transformers peft accelerate bitsandbytes tensorboard
```

> If CUDA isn't available, drop the `--index-url`. On CPU, training will be slower; use the smallest models and fewer steps.

## Structure
- `day1_numpy_mlp/` — Neural net from scratch with NumPy on a tiny binary classification dataset.
- `day2_pytorch_cnn/` — PyTorch training loop on MNIST (vision). Build the loop yourself.
- `day3_transformers_textclass/` — Text classification with Hugging Face `datasets` + `transformers`.
- `day4_lora_peft/` — LoRA fine‑tuning for a small causal LM **without** paid APIs.
- `day5_hparam_eval/` — Hyperparameter search & robust evaluation (confusion matrix, PR curve).
- `day6_transfer_vision/` — Transfer learning by **fine‑tuning only the head vs full model** on CIFAR‑10.
- `day7_capstone/` — Your mini‑project + report template.

Each folder has a README with objectives, checklists, and hints.
