# EchoVLM — Parameter-Efficient Fine-Tuning of a Vision-Language Model for Medical VQA

MSc dissertation project (Computer Science — Artificial Intelligence & Machine Vision, University of East London).

This repository contains the fine-tuning pipeline used to adapt **PaliGemma**, a 3B-parameter
vision-language model, to **medical Visual Question Answering (VQA)** using **QLoRA**
(quantized low-rank adaptation) on a single consumer-grade GPU.

## Why QLoRA

Full fine-tuning of a 3B-parameter VLM needs far more VRAM than a single T4 (16GB) provides.
QLoRA makes this tractable by:

- Loading the base model in **4-bit NormalFloat (NF4)** precision via `bitsandbytes`, cutting
  the memory footprint of the frozen backbone by roughly 4x relative to fp16.
- Training small **LoRA adapters** (rank-decomposed update matrices) injected into the
  attention and projection layers, instead of updating all 3B parameters.
- Using **gradient accumulation** to reach an effective batch size well beyond what fits in
  memory in one step, and **gradient checkpointing** to trade compute for activation memory.

Together this keeps peak memory low enough to fine-tune on a single T4 without exhausting
GPU memory, while still adapting the model's visual-reasoning behaviour to the medical domain.

## Pipeline overview

```
raw VQA pairs (image, question, answer)
        │
        ▼
  preprocessing / tokenization  (src/preprocess.py)
        │
        ▼
  4-bit quantized PaliGemma  +  LoRA adapters   (src/train_qlora.py)
        │
        ▼
  causal LM fine-tuning loop (gradient accumulation + checkpointing)
        │
        ▼
  evaluation on held-out VQA pairs  (src/evaluate.py)
```

## Repository structure

```
configs/qlora_config.yaml   Hyperparameters: LoRA rank/alpha/dropout, target modules,
                             quantization settings, training schedule
src/preprocess.py           Loads image/question/answer triples and formats them into the
                             causal-LM prompt/target layout PaliGemma expects
src/train_qlora.py          Builds the 4-bit quantized model, attaches PEFT/LoRA adapters,
                             and runs the fine-tuning loop
src/evaluate.py             Runs the fine-tuned adapter against a held-out split and reports
                             exact-match / token-level accuracy
requirements.txt            Pinned dependencies
```

## Tech stack

`PyTorch` · `Hugging Face Transformers` · `PEFT` · `bitsandbytes` · `QLoRA` · `PaliGemma`

## Hardware target

Designed and tuned for a single NVIDIA T4 (16GB) — the configuration in
`configs/qlora_config.yaml` (4-bit NF4 quantization, LoRA rank 16, gradient accumulation,
gradient checkpointing) reflects that constraint rather than a multi-GPU setup.

## Status

Core fine-tuning pipeline is implemented and documented here as a reference implementation
of the approach used in the dissertation. Trained checkpoints and the full medical VQA
dataset are not redistributed in this repo for licensing reasons.

## Author

Krish Patel — [linkedin.com/in/itzkrishpatel](https://linkedin.com/in/itzkrishpatel)
