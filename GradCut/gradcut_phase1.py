import numpy as np
import torch
import sys
import random
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, AutoConfig
from trl import SFTTrainer
import torch.nn.functional as F
import math
import os
from collections import defaultdict
import argparse
from huggingface_hub import login
from utils import SharedLoRAModule, load_global_config, prepare_datasets, set_seed


parser = argparse.ArgumentParser()
parser.add_argument("--step", type=int, required=True, help="step id")
parser.add_argument("--config", type=str, default="config.yaml", help="path to config yaml")
args = parser.parse_args()
CONFIG = load_global_config(args.config)



def estimate_cost(module_name, in_dim, out_dim, r, model):
    # Extract the layer's depth
    current_depth = int(module_name.split(".")[CONFIG["global"]["layer_index_position_in_name"]])
    return CONFIG["global"]["layer_depth"] - current_depth


class SharedDynamicLoRALinear(nn.Module):
    def __init__(self, base_layer, shared_lora_module, module_name):
        super().__init__()
        self.base_layer = base_layer
        self.shared_lora = shared_lora_module
        self.module_name = module_name
        self.full_activation = True
        r = self.shared_lora.lora_A.out_features
        out_dim = self.shared_lora.lora_B.out_features

        self.vec_A = nn.Parameter(torch.ones(r))
        self.vec_B = nn.Parameter(torch.ones(out_dim))
        self.f = True

    def forward(self, x):
        base_out = self.base_layer(x)
        torch_result_dtype = base_out.dtype
        if getattr(self, 'full_activation', False):
            lora_out = self.shared_lora(x, vec_A=self.vec_A, vec_B=self.vec_B)
            return base_out + lora_out.to(torch_result_dtype)

        else:
            if self.f:
                lora_out = self.shared_lora(x, vec_A=self.vec_A, vec_B=self.vec_B)
                return base_out + lora_out.to(torch_result_dtype)
            else:
                return base_out


class CustomSharedDynamicLoraModel(nn.Module):
    def __init__(self, base_model, r=64, lora_alpha=128, lora_dropout=0.1, target_modules=None, config=None):
        super().__init__()
        self.model = base_model
        self.config = config or CONFIG
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or self.config["phase1"]["model"]["target_modules"]
        self.step = 0
        self.prune_step = self.config["phase1"]["pruning"]["prune_step"]
        self._setup_lora()

        self.full_activation = False
        self.type_to_keep = self.config["phase1"]["pruning"]["type_to_keep"]
        self.prune_interval = self.config["phase1"]["pruning"]["prune_interval"]
        self.max_prune_steps = int(np.ceil((self.config["global"]["layer_depth"] - min(self.type_to_keep.values())) / self.prune_step) * self.prune_interval) + 1
        self.flag = 0

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if self.training:
            if self.step < self.config["phase1"]["pruning"]["initial_full_activation_steps"]:
                self.full_activation = True
            elif self.flag == 0:
                self.full_activation = False
                for module in self.model.modules():
                    if isinstance(module, SharedDynamicLoRALinear):
                        module.full_activation = self.full_activation
                self.flag = 1

            if self.step >= self.config["phase1"]["pruning"]["initial_full_activation_steps"] and self.step % self.prune_interval == 0 and self.step < self.max_prune_steps:
                type_to_active = {}
                torch.cuda.empty_cache()

                min_score_dict = defaultdict(lambda: float('inf'))
                max_score_dict = defaultdict(lambda: float('-inf'))
                min_cost_dict = defaultdict(lambda: float('inf'))
                max_cost_dict = defaultdict(lambda: float('-inf'))
                for name, module in self.model.named_modules():
                    if isinstance(module, SharedDynamicLoRALinear) and module.f:
                        A = module.shared_lora.lora_A
                        B = module.shared_lora.lora_B
                        in_dim = A.in_features
                        out_dim = B.out_features
                        r = A.out_features

                        layer_type = module.module_name.split(".")[-1]
                        score = (torch.norm(module.vec_A) + torch.norm(module.vec_B)).item()
                        cost = estimate_cost(name, in_dim, out_dim, r, self.model)
                        min_score_dict[layer_type] = min(score, min_score_dict[layer_type])
                        max_score_dict[layer_type] = max(score, max_score_dict[layer_type])
                        min_cost_dict[layer_type] = min(cost, min_cost_dict[layer_type])
                        max_cost_dict[layer_type] = max(cost, max_cost_dict[layer_type])
                        type_to_active.setdefault(layer_type, []).append((score, cost, module))

                

                for layer_type, modules in type_to_active.items():
                    keep = self.type_to_keep.get(layer_type, 0)
                    if len(modules) > keep:
                        modules.sort(key=lambda x: (self.config["phase1"]["pruning"]["importance_weight"] * (x[0] - min_score_dict[layer_type]) / (max_score_dict[layer_type] - min_score_dict[layer_type]) - self.config["phase1"]["pruning"]["cost_weight"] * (x[1] - min_cost_dict[layer_type]) / (max_cost_dict[layer_type] - min_cost_dict[layer_type])))
                        to_prune = modules[:self.prune_step]
                        for _, _, m in to_prune:
                            m.vec_A.requires_grad_(False)
                            m.vec_B.requires_grad_(False)
                            m.f = False
                            if hasattr(m.base_layer, "bias") and m.base_layer.bias is not None:
                                m.base_layer.bias.requires_grad_(False)

            self.step += 1

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        return outputs

    def _setup_lora(self):
        def manually_get_submodule(model, module_name):
            parts = module_name.split(".")
            parent = model
            for attr in parts[:-1]:
                parent = getattr(parent, attr)
            return parent, parts[-1]

        target_shapes = {}
        for name, module in self.model.named_modules():
            last_part = name.split(".")[-1]
            if last_part in self.target_modules and isinstance(module, nn.Linear):
                shape = (module.in_features, module.out_features)
                target_shapes[last_part] = shape

        shared_modules = nn.ModuleDict()
        for t, (in_dim, out_dim) in target_shapes.items():
            shared_modules[t] = SharedLoRAModule(
                in_features=in_dim,
                out_features=out_dim,
                r=self.r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
            )

        replaced_count = 0
        for name, module in list(self.model.named_modules()):
            last_part = name.split(".")[-1]
            if last_part in self.target_modules and isinstance(module, nn.Linear):
                parent, attr = manually_get_submodule(self.model, name)
                wrapped = SharedDynamicLoRALinear(module, shared_modules[last_part], name)
                parent._modules[attr] = wrapped
                replaced_count += 1

        self.shared_modules = shared_modules
        


def get_phase1_paths(curr_step):
    if curr_step == 0:
        model_name = CONFIG["global"]["base_model_name"]
        phase1_output_dir = CONFIG["paths"]["patch0"]["phase1_output_dir"]
    else:
        model_name = CONFIG["paths"]["templates"]["model_name"].format(step=curr_step)
        phase1_output_dir = CONFIG["paths"]["templates"]["phase1_output_dir"].format(step=curr_step)
    return model_name, phase1_output_dir


def load_phase1_model_and_tokenizer(model_name):
    #Adjust architecture-specific options such as sliding_window and attn_implementation for different model families. Add or remove options as needed.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    config.sliding_window = CONFIG["global"]["sliding_window"]
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        device_map=CONFIG["global"]["device_map"],
        attn_implementation=CONFIG["global"]["attn_implementation"],
    )

    tokenizer.padding_side = CONFIG["global"]["tokenizer_padding_side"]
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def build_phase1_trainer(model, tokenizer, train_dataset):
    trainer_cfg = CONFIG["phase1"]["trainer"]
    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=trainer_cfg["max_seq_length"],
        dataset_num_proc=trainer_cfg["dataset_num_proc"],
        args=TrainingArguments(
            per_device_train_batch_size=trainer_cfg["per_device_train_batch_size"],
            save_steps=trainer_cfg["save_steps"],
            max_steps=model.max_prune_steps,
            save_strategy=trainer_cfg["save_strategy"],
            learning_rate=trainer_cfg["learning_rate"],
            logging_steps=trainer_cfg["logging_steps"],
            optim=trainer_cfg["optim"],
            lr_scheduler_type=trainer_cfg["lr_scheduler_type"],
            seed=trainer_cfg["seed"],
            save_total_limit=trainer_cfg["save_total_limit"],
            report_to=trainer_cfg["report_to"],
            output_dir=".hf_tmp",
        ),
    )


def save_phase1_outputs(model, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    vecs = {}
    for name, module in model.model.named_modules():
        if isinstance(module, SharedDynamicLoRALinear) and module.f:
            vecs[name] = {
                "vec_A": module.vec_A.detach().cpu(),
                "vec_B": module.vec_B.detach().cpu(),
                "in_dim": module.shared_lora.lora_A.in_features,
                "out_dim": module.shared_lora.lora_B.out_features,
            }
    torch.save(vecs, os.path.join(output_dir, "vecs.pt"))
    torch.save(model.shared_modules.state_dict(), os.path.join(output_dir, "shared_lora_dict.pt"))


def main():
    set_seed(CONFIG["global"]["seed"])
    train_dataset = prepare_datasets(CONFIG)

    curr_step = args.step
    model_name, phase1_output_dir = get_phase1_paths(curr_step)
    base_model, tokenizer = load_phase1_model_and_tokenizer(model_name)

    model_cfg = CONFIG["phase1"]["model"]
    model = CustomSharedDynamicLoraModel(base_model, r=model_cfg["r"], lora_alpha=model_cfg["lora_alpha"], lora_dropout=model_cfg["lora_dropout"], target_modules=model_cfg["target_modules"], config=CONFIG)
    trainer = build_phase1_trainer(model, tokenizer, train_dataset)
    trainer.train()
    save_phase1_outputs(model, phase1_output_dir)


if __name__ == "__main__":
    main()
