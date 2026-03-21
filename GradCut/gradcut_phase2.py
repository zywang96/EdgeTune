import numpy as np
import torch
import torch.nn as nn
import os
from collections import defaultdict
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, AutoConfig
from trl import SFTTrainer
from utils import SharedLoRAModule, load_global_config, prepare_datasets, set_seed


parser = argparse.ArgumentParser()
parser.add_argument("--step", type=int, required=True, help="step id")
parser.add_argument("--config", type=str, default="config.yaml", help="path to config yaml")
args = parser.parse_args()
CONFIG = load_global_config(args.config)


def get_phase2_paths(curr_step):
    if curr_step == 0:
        model_name = CONFIG["global"]["base_model_name"]
        save_dir = CONFIG["paths"]["patch0"]["phase1_output_dir"]
        phase2_output_dir = CONFIG["paths"]["patch0"]["phase2_output_dir"]
    else:
        model_name = CONFIG["paths"]["templates"]["model_name"].format(step=curr_step)
        save_dir = CONFIG["paths"]["templates"]["phase1_output_dir"].format(step=curr_step)
        phase2_output_dir = CONFIG["paths"]["templates"]["phase2_output_dir"].format(step=curr_step)
    return model_name, save_dir, phase2_output_dir


def load_phase2_artifacts(model_name, save_dir):
    vecs = torch.load(f"{save_dir}/vecs.pt")
    shared_lora_state = torch.load(f"{save_dir}/shared_lora_dict.pt")

    config = AutoConfig.from_pretrained(model_name)
    config.sliding_window = CONFIG["global"]["sliding_window"]
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        device_map=CONFIG["global"]["device_map"],
        attn_implementation=CONFIG["global"]["attn_implementation"],
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = CONFIG["global"]["tokenizer_padding_side"]
    return vecs, shared_lora_state, base_model, tokenizer


def build_ttl_and_shapes(vecs):
    ttl_list = []
    for name, info in vecs.items():
        ttl_list.append(CONFIG["global"]["layer_depth"] - int(float(name.split('.')[CONFIG["global"]["layer_index_position_in_name"]])))

    type_to_shape = {}
    for name, info in vecs.items():
        layer_type = name.split(".")[-1]
        if layer_type not in type_to_shape:
            type_to_shape[layer_type] = (info["in_dim"], info["out_dim"])
    return ttl_list, type_to_shape


def build_shared_lora_dict(type_to_shape, shared_lora_state):
    shared_lora_dict = nn.ModuleDict()
    for t, (in_dim, out_dim) in type_to_shape.items():
        shared_cfg = CONFIG["phase1"]["model"]
        shared_lora_dict[t] = SharedLoRAModule(in_dim, out_dim, r=shared_cfg["r"], lora_alpha=shared_cfg["lora_alpha"], lora_dropout=shared_cfg["lora_dropout"])
    shared_lora_dict.load_state_dict(shared_lora_state)
    return shared_lora_dict


class LoRALinear(nn.Module):
    def __init__(self, base_layer, lora_module, vec_A, vec_B):
        super().__init__()
        self.base = base_layer
        self.lora = lora_module
        self.vec_A = nn.Parameter(vec_A)
        self.vec_B = nn.Parameter(vec_B)

    def forward(self, x):
        out = self.base(x)
        lora_out = self.lora(x, vec_A=self.vec_A, vec_B=self.vec_B)
        return out + lora_out.to(out.dtype)


class FinalLoRAModel(nn.Module):
    def __init__(self, base_model, shared_lora_dict, vecs, ttl_list, r, T=2.0, config=None):
        super().__init__()
        self.model = base_model
        self.shared_lora_dict = shared_lora_dict
        self.config = config or CONFIG
        self.target_modules = self.config["phase2"]["final_model"]["target_modules"]

        self.ignore_list = []
        min_ttl = min(ttl_list)
        max_ttl = max(ttl_list)
        new_ttl_list = (np.array(ttl_list) - min_ttl) / (max_ttl - min_ttl)

        id_mean = np.mean(new_ttl_list)
        id_std = np.std(new_ttl_list)
        for lid in new_ttl_list:
            if lid > id_mean + self.config["phase2"]["final_model"]["ignore_threshold_std_multiplier"] * id_std:
                self.ignore_list.append(self.config["global"]["layer_depth"] - int(lid * (max_ttl - min_ttl) + min_ttl))
        

        type_groups = defaultdict(list)
        for name in vecs:
            layer_type = name.split(".")[-1]
            type_groups[layer_type].append((name, vecs[name]))

        for t, entries in type_groups.items():
            norms = torch.tensor([(v["vec_A"].norm() + v["vec_B"].norm()).item() for _, v in entries])
            norms = (norms - torch.min(norms)) / (torch.max(norms) - torch.min(norms))
            mean, std = norms.mean(), norms.std()
            for i, (name, v) in enumerate(entries):
                layer_id = int(name.split('.')[self.config["global"]["layer_index_position_in_name"]])

                if layer_id in self.ignore_list:
                    continue

                parent, attr = self._get_submodule(name)
                if norms[i] > mean + T * std:
                    in_dim = v["in_dim"]
                    out_dim = v["out_dim"]
                    dedicated_lora = SharedLoRAModule(in_dim, out_dim, r, lora_alpha=r * self.config["phase2"]["final_model"]["dedicated_lora_alpha_multiplier"], lora_dropout=self.config["phase2"]["final_model"]["dedicated_lora_dropout"])
                    vec_A = nn.Parameter(torch.ones(r))
                    vec_B = nn.Parameter(torch.ones(out_dim))
                    layer = LoRALinear(parent._modules[attr], dedicated_lora, vec_A, vec_B)
                else:
                    layer = LoRALinear(parent._modules[attr], self.shared_lora_dict[t], v["vec_A"], v["vec_B"])
                parent._modules[attr] = layer

    def _get_submodule(self, module_name):
        parts = module_name.split(".")
        parent = self.model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        return parent, parts[-1]

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)


def freeze_non_lora_parameters(model):
    for name, param in model.named_parameters():
        if "lora" in name or "vec_A" in name or "vec_B" in name:
            param.requires_grad = True
        elif name.endswith(".bias"):
            param.requires_grad = False
        else:
            param.requires_grad = False


def build_phase2_trainer(final_model, tokenizer, train_dataset):
    trainer_cfg = CONFIG["phase2"]["trainer"]
    return SFTTrainer(
        model=final_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=trainer_cfg["max_seq_length"],
        dataset_num_proc=trainer_cfg["dataset_num_proc"],
        args=TrainingArguments(
            per_device_train_batch_size=trainer_cfg["per_device_train_batch_size"],
            save_steps=trainer_cfg["save_steps"],
            num_train_epochs=trainer_cfg["num_train_epochs"],
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


def save_phase2_outputs(final_model, shared_lora_dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = None
    layer_lora_info = {}
    dedicated_loras = {}

    for name, module in final_model.model.named_modules():
        if isinstance(module, LoRALinear):
            entry = {"vec_A": module.vec_A.detach().cpu(), "vec_B": module.vec_B.detach().cpu()}
            if any(module.lora is shared for shared in shared_lora_dict.values()):
                
                for type_key, shared in shared_lora_dict.items():
                    if module.lora is shared:
                        entry["lora_type"] = "shared"
                        entry["lora_key"] = type_key
                        break
            else:
                entry["lora_type"] = "dedicated"
                dedicated_loras[name] = module.lora.state_dict()
            layer_lora_info[name] = entry

    torch.save(shared_lora_dict.state_dict(), os.path.join(output_dir, "shared_lora.pt"))
    torch.save(layer_lora_info, os.path.join(output_dir, "layer_lora_info.pt"))
    torch.save(dedicated_loras, os.path.join(output_dir, "dedicated_loras.pt"))


def main():
    curr_step = args.step
    model_name, save_dir, phase2_output_dir = get_phase2_paths(curr_step)

    set_seed(CONFIG["global"]["seed"])
    vecs, shared_lora_state, base_model, tokenizer = load_phase2_artifacts(model_name, save_dir)
    ttl_list, type_to_shape = build_ttl_and_shapes(vecs)
    shared_lora_dict = build_shared_lora_dict(type_to_shape, shared_lora_state)

    final_cfg = CONFIG["phase2"]["final_model"]
    final_model = FinalLoRAModel(base_model, shared_lora_dict, vecs, ttl_list, r=final_cfg["dedicated_r"], T=final_cfg["threshold_T"], config=CONFIG)
    freeze_non_lora_parameters(final_model)
    total_params = 0
    '''
    print("Trainable parameters:")
    for n, p in final_model.named_parameters():
        if p.requires_grad:
            print(n, p.shape)
            total_params += p.numel()
    print(f"\nTotal trainable parameters: {total_params}")
    '''
    train_dataset = prepare_datasets(CONFIG)
    trainer = build_phase2_trainer(final_model, tokenizer, train_dataset)
    trainer.train()

    tokenizer.save_pretrained(phase2_output_dir)
    save_phase2_outputs(final_model, shared_lora_dict, phase2_output_dir)


if __name__ == "__main__":
    main()
