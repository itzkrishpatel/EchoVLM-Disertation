"""
preprocess.py
-------------
Loads (image, question, answer) triples for medical VQA and formats them into the
prompt/target layout PaliGemma expects during causal-LM fine-tuning.

Expected input: a JSONL file where each line looks like

    {"image_path": "images/scan_0001.png", "question": "Is there a fracture visible?", "answer": "No visible fracture."}

Usage:
    from preprocess import VQADataset, build_collate_fn
"""

import json
import os
from dataclasses import dataclass
from typing import Any

from PIL import Image
from torch.utils.data import Dataset


@dataclass
class VQAExample:
    image_path: str
    question: str
    answer: str


class VQADataset(Dataset):
    """Reads a JSONL manifest of medical VQA triples and returns processor-ready inputs."""

    def __init__(self, manifest_path: str, image_dir: str, processor, max_length: int = 512):
        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length
        self.examples: list[VQAExample] = list(self._load(manifest_path))

    @staticmethod
    def _load(manifest_path: str):
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                yield VQAExample(
                    image_path=row["image_path"],
                    question=row["question"].strip(),
                    answer=row["answer"].strip(),
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        image = Image.open(os.path.join(self.image_dir, ex.image_path)).convert("RGB")

        # PaliGemma expects a short imperative prompt describing the task, followed by the
        # question; the answer is the generation target during fine-tuning.
        prompt = f"answer en {ex.question}"

        inputs = self.processor(
            text=prompt,
            images=image,
            suffix=ex.answer,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        return {k: v.squeeze(0) for k, v in inputs.items()}


def build_collate_fn(processor):
    """Returns a collate_fn that left-pads variable-length batches for the processor-s tokenizer."""

    pad_token_id = processor.tokenizer.pad_token_id

    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        keys = batch[0].keys()
        out: dict[str, Any] = {}
        for key in keys:
            values = [item[key] for item in batch]
            if key == "input_ids":
                out[key] = torch.nn.utils.rnn.pad_sequence(
                    values, batch_first=True, padding_value=pad_token_id
                )
            elif key == "attention_mask":
                out[key] = torch.nn.utils.rnn.pad_sequence(values, batch_first=True, padding_value=0)
            else:
                out[key] = torch.stack(values)
        return out

    return collate_fn
