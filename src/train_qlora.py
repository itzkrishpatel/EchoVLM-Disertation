"""
train_qlora.py
---------------
QLoRA fine-tuning entry point: loads PaliGemma in 4-bit precision, attaches LoRA
adapters via PEFT, and runs the causal-LM training loop on a medical VQA dataset.

Usage:
    python src/train_qlora.py --config configs/qlora_config.yaml
"""

import argparse

import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    BitsAndBytesConfig,
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    Trainer,
    TrainingArguments,
)

from preprocess import VQADataset, build_collate_fn


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model_and_processor(cfg: dict):
    quant_cfg = cfg["quantization"]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg["load_in_4bit"],
        bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, quant_cfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
    )

    model_id = cfg["model"]["base_model_id"]
    processor = PaliGemmaProcessor.from_pretrained(model_id)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # Freeze the quantized backbone and make it gradient-checkpointing friendly before
    # attaching LoRA adapters — required for QLoRA to train stably in 4-bit.
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=cfg["training"]["gradient_checkpointing"]
    )

    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        target_modules=lora_cfg["target_modules"],
        task_type=lora_cfg["task_type"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, processor


def build_training_args(cfg: dict) -> TrainingArguments:
    t = cfg["training"]
    return TrainingArguments(
        output_dir=t["output_dir"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        gradient_checkpointing=t["gradient_checkpointing"],
        num_train_epochs=t["num_train_epochs"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        optim=t["optim"],
        logging_steps=t["logging_steps"],
        save_strategy=t["save_strategy"],
        eval_strategy=t["eval_strategy"],
        bf16=t["bf16"],
        seed=t["seed"],
        report_to="none",
    )


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for PaliGemma medical VQA")
    parser.add_argument("--config", type=str, default="configs/qlora_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model, processor = build_model_and_processor(cfg)

    data_cfg = cfg["data"]
    train_ds = VQADataset(
        manifest_path=data_cfg["train_split"],
        image_dir=data_cfg["image_dir"],
        processor=processor,
        max_length=cfg["training"]["max_seq_length"],
    )
    eval_ds = VQADataset(
        manifest_path=data_cfg["eval_split"],
        image_dir=data_cfg["image_dir"],
        processor=processor,
        max_length=cfg["training"]["max_seq_length"],
    )

    trainer = Trainer(
        model=model,
        args=build_training_args(cfg),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=build_collate_fn(processor),
    )

    trainer.train()
    trainer.save_model(cfg["training"]["output_dir"])
    processor.save_pretrained(cfg["training"]["output_dir"])


if __name__ == "__main__":
    main()
