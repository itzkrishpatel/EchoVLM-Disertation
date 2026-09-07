"""
evaluate.py
-----------
Runs a fine-tuned QLoRA adapter against a held-out medical VQA split and reports
exact-match and token-level accuracy.

Usage:
    python src/evaluate.py --adapter_dir checkpoints/echovlm-qlora --eval_manifest data/eval.jsonl
"""

import argparse
import json
import os

import torch
from peft import PeftModel
from transformers import BitsAndBytesConfig, PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from PIL import Image


def load_model(base_model_id: str, adapter_dir: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = PaliGemmaForConditionalGeneration.from_pretrained(
        base_model_id, quantization_config=bnb_config, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    processor = PaliGemmaProcessor.from_pretrained(adapter_dir)
    return model, processor


def exact_match(pred: str, gold: str) -> bool:
    return pred.strip().lower() == gold.strip().lower()


def token_overlap_f1(pred: str, gold: str) -> float:
    pred_tokens = pred.strip().lower().split()
    gold_tokens = gold.strip().lower().split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


@torch.no_grad()
def run_eval(model, processor, manifest_path: str, image_dir: str, max_new_tokens: int = 32):
    exact_matches, f1_scores = [], []

    with open(manifest_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    for row in rows:
        image = Image.open(os.path.join(image_dir, row["image_path"])).convert("RGB")
        prompt = f"answer en {row['question'].strip()}"
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        prediction = processor.decode(output_ids[0], skip_special_tokens=True)
        # PaliGemma echoes the prompt before the answer; strip it off.
        prediction = prediction[len(prompt):].strip()

        exact_matches.append(exact_match(prediction, row["answer"]))
        f1_scores.append(token_overlap_f1(prediction, row["answer"]))

    return {
        "n_examples": len(rows),
        "exact_match": sum(exact_matches) / len(exact_matches),
        "token_f1": sum(f1_scores) / len(f1_scores),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a QLoRA-tuned PaliGemma adapter on medical VQA")
    parser.add_argument("--base_model_id", type=str, default="google/paligemma-3b-pt-224")
    parser.add_argument("--adapter_dir", type=str, required=True)
    parser.add_argument("--eval_manifest", type=str, required=True)
    parser.add_argument("--image_dir", type=str, default="data/images")
    args = parser.parse_args()

    model, processor = load_model(args.base_model_id, args.adapter_dir)
    metrics = run_eval(model, processor, args.eval_manifest, args.image_dir)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
