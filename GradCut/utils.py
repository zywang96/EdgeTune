import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from datasets import load_dataset


_CONFIG_CACHE = {}


def load_global_config(config_path: str = "config.yaml"):
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), config_path)
    config_path = os.path.abspath(config_path)

    if config_path not in _CONFIG_CACHE:
        with open(config_path, "r", encoding="utf-8") as f:
            _CONFIG_CACHE[config_path] = yaml.safe_load(f) or {}
    return _CONFIG_CACHE[config_path]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_qwen_prompt(example, instruction_text=None):
    sentence = example['sentence']
    label = example['label']
    label_str = 'Positive' if label == 1 else 'Negative'
    instruction_text = instruction_text or 'You are an AI assistant that only answers negative/positive.'
    return {
        'text': (
            f'### Instruction:\n{instruction_text}\n\n'
            f'### Input:\nSentence: {sentence}\nQuestion: Is this sentence positive or negative?\n\n'
            f'### Response:\n{label_str}'
        )
    }


def prepare_datasets(config=None):
    config = config or {}
    dataset_cfg = config.get("dataset", {})
    dataset_name = dataset_cfg.get("name", "glue")
    dataset_subset = dataset_cfg.get("subset", "sst2")
    instruction_text = dataset_cfg.get("instruction_text", "You are an AI assistant that only answers negative/positive.")

    dataset = load_dataset(dataset_name, dataset_subset)
    train_dataset = dataset['train']

    columns_to_remove = dataset['train'].column_names
    train_dataset = train_dataset.map(
        lambda example: format_qwen_prompt(example, instruction_text=instruction_text),
        remove_columns=columns_to_remove,
    )
    return train_dataset


class SharedLoRAModule(nn.Module):
    def __init__(self, in_features, out_features, r, lora_alpha, lora_dropout):
        super().__init__()
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.scaling = lora_alpha / r
        self.dropout = nn.Dropout(lora_dropout)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x, vec_A=None, vec_B=None):
        A = self.lora_A.weight
        B = self.lora_B.weight

        if vec_A is not None:
            A = A * vec_A.unsqueeze(-1)
        if vec_B is not None:
            B = B * vec_B.unsqueeze(-1)

        x = self.dropout(x.to(self.lora_A.weight.dtype))
        x = F.linear(x, A)
        x = F.linear(x, B) * self.scaling
        return x
