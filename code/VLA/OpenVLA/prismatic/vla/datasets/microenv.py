"""
microenv.py

Dataset utilities for loading converted MicroEnv VLA demonstrations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Type

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import IGNORE_INDEX


class MicroEnvVLADataset(Dataset):
    """Map-style dataset for converted MicroEnv demonstration manifests."""

    def __init__(
        self,
        dataset_dir: Path,
        split: str,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
        use_proprio: bool = True,
        image_aug: bool = False,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn
        self.use_proprio = use_proprio
        self.image_aug = image_aug

        manifest_path = self.dataset_dir / f"{split}_samples.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(f"MicroEnv manifest not found: {manifest_path}")

        with manifest_path.open("r") as f:
            self.samples = [json.loads(line) for line in f if line.strip()]

        stats_path = self.dataset_dir / "dataset_statistics.json"
        if not stats_path.exists():
            raise FileNotFoundError(f"Dataset statistics file not found: {stats_path}")
        with stats_path.open("r") as f:
            self.dataset_statistics = json.load(f)

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_image_path(self, image_path: str) -> Path:
        path = Path(image_path)
        if path.is_absolute():
            return path
        dataset_relative = self.dataset_dir / path
        if dataset_relative.exists():
            return dataset_relative
        for parent in [self.dataset_dir, *self.dataset_dir.parents]:
            candidate = parent / path
            if candidate.exists():
                return candidate
        return dataset_relative

    def _load_image(self, image_path: str) -> Image.Image:
        img = Image.open(self._resolve_image_path(image_path)).convert("RGB")
        return img

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        actions = np.asarray(sample["actions"], dtype=np.float32)

        prompt_builder = self.prompt_builder_fn("openvla")
        action_chunk_string = "".join(self.action_tokenizer(actions))
        action_chunk_len = len(action_chunk_string)

        conversation = [
            {"from": "human", "value": f"What action should the robot take to {sample['instruction'].lower()}?"},
            {"from": "gpt", "value": action_chunk_string},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)
        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX

        pixel_values = self.image_transform(self._load_image(sample["image_path"]))

        output = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "labels": labels,
            "dataset_name": sample["dataset_name"],
            "actions": actions,
        }
        if self.use_proprio:
            output["proprio"] = np.asarray(sample["proprio"], dtype=np.float32)
        return output
