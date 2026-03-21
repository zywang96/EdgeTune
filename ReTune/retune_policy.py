import os
import argparse
from collections import defaultdict
import subprocess
import sys
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from huggingface_hub import login


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=os.path.join(os.path.dirname(__file__), "config.yaml"), help="path to config yaml")
    return parser.parse_args()


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class SharedLoRAModule(torch.nn.Module):
    def __init__(self, in_features, out_features, r, lora_alpha, lora_dropout):
        super().__init__()
        self.lora_A = torch.nn.Linear(in_features, r, bias=False)
        self.lora_B = torch.nn.Linear(r, out_features, bias=False)
        self.scaling = lora_alpha / r

    def forward(self, x, vec_A, vec_B):
        A = self.lora_A.weight * vec_A.unsqueeze(-1)
        B = self.lora_B.weight * vec_B.unsqueeze(-1)
        return torch.nn.functional.linear(torch.nn.functional.linear(x, A), B) * self.scaling


class RunningStat:
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x):
        x = x.item() if torch.is_tensor(x) else x
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def update_batch(self, values: torch.Tensor):
        values = values.flatten()
        n_batch = values.numel()
        if n_batch == 0:
            return

        mean_batch = values.mean().item()
        M2_batch = ((values - mean_batch) ** 2).sum().item()

        delta = mean_batch - self.mean
        total_n = self.count + n_batch

        self.mean += delta * n_batch / total_n
        self.M2 += M2_batch + delta**2 * self.count * n_batch / total_n
        self.count = total_n

    def get_mean(self):
        return self.mean

    def get_variance(self):
        return self.M2 / self.count if self.count > 0 else float("nan")

    def get_sample_variance(self):
        return self.M2 / (self.count - 1) if self.count > 1 else float("nan")

    def get_std(self):
        return self.get_sample_variance() ** 0.5 if self.count > 1 else float("nan")


def initialize_runtime(config_dict: dict):
    hf_token = config_dict["shared"]["hf_token"] or os.environ.get("HF_TOKEN")
    if hf_token:
        login(hf_token)

    runtime = {
        "device": config_dict["shared"]["device"],
        "task_name": config_dict["shared"]["task_name"],
        "model_name": config_dict["shared"]["model_name"],
        "target_modules": set(config_dict["shared"]["target_modules"]),
        "attn_implementation": config_dict["shared"]["attn_implementation"],
        "alpha_multiplier": config_dict["patch"]["alpha_multiplier"],
    }
    return runtime


def build_config(model_name: str):
    config = AutoConfig.from_pretrained(model_name)
    config.sliding_window = None
    return config


def load_model(model_path: str, runtime: dict, device_map="cpu", torch_dtype=None):
    kwargs = {
        "config": build_config(runtime["model_name"]),
        "device_map": device_map,
        "attn_implementation": runtime["attn_implementation"],
    }
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)


def should_track_weight(k: str, target_modules) -> bool:
    if not k.endswith(".weight"):
        return False
    parts = k.split(".")
    if len(parts) < 2:
        return False
    return parts[-2] in target_modules


def extract_tracked_state(model, target_modules) -> dict:
    return {
        k: v.detach().cpu().clone().float()
        for k, v in model.state_dict().items()
        if should_track_weight(k, target_modules)
    }


def load_filtered_state_from_dir(model_dir: str, runtime: dict) -> dict:
    model = load_model(model_dir, runtime, device_map="cpu", torch_dtype=torch.float32)
    state = extract_tracked_state(model, runtime["target_modules"])
    del model
    return state


def load_filtered_base_state(runtime: dict) -> dict:
    model = load_model(runtime["model_name"], runtime, device_map="cpu", torch_dtype=torch.float32)
    state = extract_tracked_state(model, runtime["target_modules"])
    del model
    return state


def build_shared_modules(shared_lora_dict: dict, runtime: dict) -> dict:
    grouped = defaultdict(dict)
    for k, v in shared_lora_dict.items():
        lora_type, param = k.split(".", 1)
        grouped[lora_type][param] = v

    shared_modules = {}
    for lora_type, state in grouped.items():
        B_shape = state["lora_B.weight"].shape
        A_shape = state["lora_A.weight"].shape
        out_dim = B_shape[0]
        r = B_shape[1]
        in_dim = A_shape[1]
        alpha = r * runtime["alpha_multiplier"]
        module = SharedLoRAModule(in_dim, out_dim, r=r, lora_alpha=alpha, lora_dropout=0.0)
        module.load_state_dict(state)
        shared_modules[lora_type] = module
    return shared_modules


def run_gradcut_for_step(config_dict: dict, step: int):
    if not config_dict.get("execution", {}).get("auto_run_gradcut_on_ft_points", False):
        return

    gradcut_config = config_dict["execution"]["gradcut_config"]
    gradcut_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "GradCut"))

    subprocess.run(
        [sys.executable, "gradcut_phase1.py", "--step", str(step), "--config", gradcut_config],
        cwd=gradcut_dir,
        check=True,
    )
    subprocess.run(
        [sys.executable, "gradcut_phase2.py", "--step", str(step), "--config", gradcut_config],
        cwd=gradcut_dir,
        check=True,
    )

def compute_patch_stats(base_model, shared_lora_dict, dedicated_loras, layer_lora_info, scaling, device, runtime):
    stat = RunningStat()
    shared_modules = build_shared_modules(shared_lora_dict, runtime)

    with torch.no_grad():
        for layer_name, info in layer_lora_info.items():
            module = base_model
            parts = layer_name.split(".")
            for p in parts[:-1]:
                module = getattr(module, p)
            _base_layer = getattr(module, parts[-1])

            vec_A = info["vec_A"].to(device)
            vec_B = info["vec_B"].to(device)

            if info["lora_type"] == "shared":
                lora_type = info["lora_key"]
                lora_module = shared_modules[lora_type]
            else:
                state = dedicated_loras[layer_name]
                A_shape = state["lora_A.weight"].shape
                B_shape = state["lora_B.weight"].shape
                r = A_shape[0]
                in_dim = A_shape[1]
                out_dim = B_shape[0]
                alpha = r * runtime["alpha_multiplier"]
                lora_module = SharedLoRAModule(in_dim, out_dim, r=r, lora_alpha=alpha, lora_dropout=0.0)
                lora_module.load_state_dict(state)

            A = lora_module.lora_A.weight * vec_A.unsqueeze(-1)
            B = lora_module.lora_B.weight * vec_B.unsqueeze(-1)
            delta_w = scaling * (B @ A)
            stat.update_batch((delta_w ** 2).flatten())

    return stat.get_mean(), stat.get_std()


def apply_patch_to_model(base_model, shared_lora_dict, dedicated_loras, layer_lora_info, scaling, device, runtime):
    shared_modules = build_shared_modules(shared_lora_dict, runtime)

    with torch.no_grad():
        for layer_name, info in layer_lora_info.items():
            module = base_model
            parts = layer_name.split(".")
            for p in parts[:-1]:
                module = getattr(module, p)
            base_layer = getattr(module, parts[-1])

            vec_A = info["vec_A"].to(device)
            vec_B = info["vec_B"].to(device)

            if info["lora_type"] == "shared":
                lora_type = info["lora_key"]
                lora_module = shared_modules[lora_type]
            else:
                state = dedicated_loras[layer_name]
                A_shape = state["lora_A.weight"].shape
                B_shape = state["lora_B.weight"].shape
                r = A_shape[0]
                in_dim = A_shape[1]
                out_dim = B_shape[0]
                alpha = r * runtime["alpha_multiplier"]
                lora_module = SharedLoRAModule(in_dim, out_dim, r=r, lora_alpha=alpha, lora_dropout=0.0)
                lora_module.load_state_dict(state)

            A = lora_module.lora_A.weight * vec_A.unsqueeze(-1)
            B = lora_module.lora_B.weight * vec_B.unsqueeze(-1)
            delta_W = scaling * (B @ A)
            base_layer.weight.data += delta_W.to(base_layer.weight.dtype)


def load_patch_artifacts(output_dir: str, device: str):
    shared_lora_dict = torch.load(os.path.join(output_dir, "shared_lora.pt"), map_location=device)
    dedicated_loras = torch.load(os.path.join(output_dir, "dedicated_loras.pt"), map_location=device)
    layer_lora_info = torch.load(os.path.join(output_dir, "layer_lora_info.pt"), map_location=device)
    return shared_lora_dict, dedicated_loras, layer_lora_info


def get_initial_patch_stats(config_dict: dict, runtime: dict):
    base_model = load_model(runtime["model_name"], runtime, device_map="auto")
    scaling = runtime["alpha_multiplier"]
    output_dir = config_dict["patch"]["initial_patch_path"]
    shared_lora_dict, dedicated_loras, layer_lora_info = load_patch_artifacts(output_dir, runtime["device"])
    patch0_mean, patch0_std = compute_patch_stats(
        base_model, shared_lora_dict, dedicated_loras, layer_lora_info, scaling, runtime["device"], runtime
    )
    return base_model, scaling, patch0_mean, patch0_std


def save_merged_model(base_model, tokenizer, merged_dir: str):
    print(merged_dir)
    if not os.path.exists(merged_dir):
        os.makedirs(merged_dir)
    base_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)


def run_sequence_loop(config_dict: dict, runtime: dict, base_model, scaling, patch0_mean, patch0_std):
    kappa = config_dict["selection"]["kappa"]
    k_var = config_dict["selection"]["k_var"]
    start_seq = 0
    curr_seq = 0
    max_seq = config_dict["selection"]["max_seq"]
    delta_accumulator = {}
    all_stats = []
    saved_dir_template = config_dict["paths"]["merged_dir_template"]
    patch_path = config_dict["patch"]["initial_patch_path"]

    # Cache states so each iteration only loads one new checkpoint after step 0
    prev_state = load_filtered_base_state(runtime)

    while curr_seq < max_seq:
        current_model_dir = config_dict["paths"]["local_model_template"].format(step=curr_seq + 1)
        curr_state = load_filtered_state_from_dir(current_model_dir, runtime)

        # step 0: local_thetat_01 - base model
        # step i>0: local_thetat_{i+1} - local_thetat_{i}
        for k in curr_state.keys():
            delta_t = curr_state[k] - prev_state[k]
            if k not in delta_accumulator:
                delta_accumulator[k] = delta_t.clone()
            else:
                delta_accumulator[k] += delta_t

        stat = RunningStat()
        for delta in delta_accumulator.values():
            stat.update_batch((delta ** 2).flatten())
        f_mean = stat.get_mean()
        f_std = stat.get_std()

        

        if f_mean + k_var * f_std > kappa * (patch0_mean + k_var * patch0_std):

            run_gradcut_for_step(config_dict, step=curr_seq + 1)

            output_dir = config_dict["patch"]["patch_output_template"].format(step=curr_seq + 1)
            shared_lora_dict, dedicated_loras, layer_lora_info = load_patch_artifacts(output_dir, runtime["device"])

            patch0_mean, patch0_std = compute_patch_stats(
                base_model, shared_lora_dict, dedicated_loras, layer_lora_info, scaling, runtime["device"], runtime
            )

            print("patch stats: ", patch0_mean, patch0_std)

            start_seq = curr_seq
            delta_accumulator = {}
            print("update the patch")
            patch_path = config_dict["patch"]["patch_output_template"].format(step=curr_seq + 1)
        else:
            print("keep the old patch")

        tokenizer = AutoTokenizer.from_pretrained(patch_path)
        base_model = load_model(current_model_dir, runtime, device_map="cpu")

        shared_lora_dict, dedicated_loras, layer_lora_info = load_patch_artifacts(patch_path, runtime["device"])

        apply_patch_to_model(base_model, shared_lora_dict, dedicated_loras, layer_lora_info, scaling, runtime["device"], runtime)

        merged_dir = saved_dir_template.format(step=curr_seq + 1)
        save_merged_model(base_model, tokenizer, merged_dir)

        prev_state = curr_state
        curr_seq += 1

    return start_seq, curr_seq, all_stats


def main():
    args = parse_args()
    config_dict = load_config(args.config)
    runtime = initialize_runtime(config_dict)

    initial_patch_dir = config_dict["patch"]["initial_patch_path"]
    initial_patch_file = os.path.join(initial_patch_dir, "shared_lora.pt")

    if not os.path.exists(initial_patch_file):
        run_gradcut_for_step(config_dict, step=0)

    base_model, scaling, patch0_mean, patch0_std = get_initial_patch_stats(config_dict, runtime)
    run_sequence_loop(config_dict, runtime, base_model, scaling, patch0_mean, patch0_std)


if __name__ == "__main__":
    main()
