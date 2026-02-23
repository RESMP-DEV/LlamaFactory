# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bayesian hyperparameter sweep using Ax/BoTorch via the new Client API."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml
from ax.api.client import Client
from ax.api.configs import ChoiceParameterConfig, RangeParameterConfig


LORA_PARAMETERS = [
    RangeParameterConfig(name="learning_rate", bounds=(
        1e-5, 1e-3), parameter_type="float", scaling="log"),
    ChoiceParameterConfig(name="lora_rank", values=[
                          8, 16, 32, 64, 128], parameter_type="int", is_ordered=True),
    RangeParameterConfig(name="lora_dropout", bounds=(
        0.0, 0.1), parameter_type="float"),
    RangeParameterConfig(name="warmup_ratio", bounds=(
        0.03, 0.15), parameter_type="float"),
    RangeParameterConfig(name="weight_decay", bounds=(
        0.0, 0.1), parameter_type="float"),
    ChoiceParameterConfig(name="per_device_train_batch_size", values=[
                          1], parameter_type="int", is_ordered=True),
    RangeParameterConfig(name="gradient_accumulation_steps",
                         bounds=(2, 16), parameter_type="int"),
]

FULL_PARAMETERS = [
    RangeParameterConfig(name="learning_rate", bounds=(
        5e-6, 5e-5), parameter_type="float", scaling="log"),
    ChoiceParameterConfig(name="lr_scheduler_type", values=[
                          "cosine", "linear"], parameter_type="str", is_ordered=False),
    RangeParameterConfig(name="warmup_ratio", bounds=(
        0.03, 0.2), parameter_type="float"),
    RangeParameterConfig(name="weight_decay", bounds=(
        0.0, 0.2), parameter_type="float"),
    ChoiceParameterConfig(name="per_device_train_batch_size", values=[
                          1], parameter_type="int", is_ordered=True),
    RangeParameterConfig(name="gradient_accumulation_steps",
                         bounds=(1, 16), parameter_type="int"),
]


def _detect_preset(base_yaml: str, preset: str) -> str:
    if preset != "auto":
        return preset

    with open(base_yaml) as f:
        config = yaml.safe_load(f)

    if config.get("finetuning_type") == "full":
        return "full"

    return "lora"


def _get_parameters_by_preset(preset: str):
    if preset == "full":
        return FULL_PARAMETERS

    return LORA_PARAMETERS


def run_trial(
    trial_index: int,
    params: dict[str, Any],
    base_yaml: str,
    output_dir: str,
    max_steps: int = 100,
) -> float:
    """Launch a single training trial and return eval loss."""
    trial_dir = os.path.join(output_dir, f"trial_{trial_index:03d}")
    os.makedirs(trial_dir, exist_ok=True)

    # Build config by merging base yaml with trial params
    with open(base_yaml) as f:
        config = yaml.safe_load(f)

    trial_overrides = {
        "learning_rate": params["learning_rate"],
        "warmup_ratio": params["warmup_ratio"],
        "weight_decay": params["weight_decay"],
        "per_device_train_batch_size": int(params["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(params["gradient_accumulation_steps"]),
        "output_dir": trial_dir,
        "max_steps": max_steps,
        "cutoff_len": min(int(config.get("cutoff_len", 4096)), 4096),
        "preprocessing_num_workers": None,
        "dataloader_num_workers": 0,
        "overwrite_output_dir": True,
        "report_to": "wandb",
        "run_name": f"sweep-trial-{trial_index:03d}",
        "save_steps": max_steps + 1,
        "eval_strategy": "steps",
        "eval_steps": max(10, max_steps // 5),
        "val_size": min(int(config.get("val_size", 256)), 256),
        "bf16": True,
        "fp16": False,
    }

    if "lora_rank" in params:
        trial_overrides["lora_rank"] = int(params["lora_rank"])

    if "lora_dropout" in params:
        trial_overrides["lora_dropout"] = params["lora_dropout"]

    if "lr_scheduler_type" in params:
        trial_overrides["lr_scheduler_type"] = params["lr_scheduler_type"]

    config.update(trial_overrides)

    trial_yaml = os.path.join(trial_dir, "config.yaml")
    with open(trial_yaml, "w") as f:
        yaml.dump(config, f)

    env = os.environ.copy()
    env.update(
        {
            "TOKENIZERS_PARALLELISM": "false",
            "GLIBC_TUNABLES": "glibc.rtld.optional_static_tls=524288",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "PYTORCH_NVML_BASED_CUDA_CHECK": "1",
            "FORCE_TORCHRUN": "1",
            "DISABLE_VERSION_CHECK": "1",
            "NPROC_PER_NODE": os.getenv("NPROC_PER_NODE", "1"),
            "OMP_NUM_THREADS": "1",
        }
    )
    project_root = Path(__file__).resolve().parents[2]
    venv_bin = str(Path(sys.executable).parent)
    env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
    env["PYTHONPATH"] = f"{project_root / 'src'}:{env.get('PYTHONPATH', '')}"

    python = sys.executable
    cmd = [python, "-m", "llamafactory.cli", "train", trial_yaml]
    log_path = os.path.join(trial_dir, "train.log")

    print(f"\n[Trial {trial_index}] Starting with params: {params}")
    print(f"[Trial {trial_index}] Output dir: {trial_dir}")
    print(f"[Trial {trial_index}] Log: {log_path}")

    start = time.time()
    with open(log_path, "w") as log_f:
        proc = subprocess.run(
            cmd,
            cwd=str(project_root),  # LlamaFactory root
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )

    elapsed = time.time() - start
    print(
        f"[Trial {trial_index}] Finished in {elapsed:.1f}s (exit code {proc.returncode})")

    if proc.returncode != 0:
        print(
            f"[Trial {trial_index}] FAILED — returning high loss sentinel 99.0")
        return 99.0

    # Parse eval loss from trainer_state.json
    state_path = os.path.join(trial_dir, "trainer_state.json")
    if not os.path.exists(state_path):
        print(
            f"[Trial {trial_index}] trainer_state.json not found — returning sentinel 99.0")
        return 99.0

    with open(state_path) as f:
        state = json.load(f)

    # Last eval loss logged
    eval_losses = [
        entry["eval_loss"]
        for entry in state.get("log_history", [])
        if "eval_loss" in entry
    ]

    if not eval_losses:
        # Fall back to last training loss
        train_losses = [
            entry["loss"]
            for entry in state.get("log_history", [])
            if "loss" in entry
        ]
        if train_losses:
            loss = train_losses[-1]
            print(
                f"[Trial {trial_index}] No eval loss found; using last train loss {loss:.4f}")
            return loss
        print(f"[Trial {trial_index}] No loss found — returning sentinel 99.0")
        return 99.0

    loss = eval_losses[-1]
    print(f"[Trial {trial_index}] Eval loss: {loss:.4f}")
    return loss


def run_bayesian_sweep(
    base_yaml: str,
    output_dir: str,
    preset: str = "auto",
    num_trials: int = 10,
    num_initial: int = 4,
    max_steps_per_trial: int = 100,
) -> dict[str, Any]:
    """Run Bayesian sweep with Sobol initialization + BoTorch optimization."""
    resolved_preset = _detect_preset(base_yaml=base_yaml, preset=preset)
    parameters = _get_parameters_by_preset(resolved_preset)

    experiment_name = f"{Path(base_yaml).stem}_{resolved_preset}_sweep"
    client = Client()
    client.configure_experiment(
        name=experiment_name,
        parameters=parameters,
    )
    client.configure_optimization(objective="-eval_loss")
    client.configure_generation_strategy(
        method="quality",
        initialization_budget=num_initial,
    )

    print(f"[Sweep] Preset: {resolved_preset}")
    print(f"[Sweep] Search parameters: {[p.name for p in parameters]}")

    results = []
    for trial_i in range(num_trials):
        next_trials = client.get_next_trials(max_trials=1)
        trial_index, params = next(iter(next_trials.items()))
        loss = run_trial(
            trial_index=trial_index,
            params=params,
            base_yaml=base_yaml,
            output_dir=output_dir,
            max_steps=max_steps_per_trial,
        )
        client.complete_trial(trial_index=trial_index,
                              raw_data={"eval_loss": loss})
        results.append({"trial_index": trial_index,
                       "params": dict(params), "eval_loss": loss})
        print(
            f"\n[Sweep] Trial {trial_i + 1}/{num_trials} done. Loss={loss:.4f}")

    best_params, values, best_trial_index, arm_name = client.get_best_parameterization()
    print(f"\n[Sweep] Best params (trial {best_trial_index}): {best_params}")
    print(f"[Sweep] Best predicted values: {values}")

    # Save results summary
    summary = {
        "best_params": dict(best_params),
        "best_values": str(values),
        "best_trial_index": best_trial_index,
        "all_trials": results,
    }
    summary_path = os.path.join(output_dir, "sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Sweep] Summary saved to {summary_path}")

    return dict(best_params)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Bayesian hyperparameter sweep")
    parser.add_argument("--config", required=True,
                        help="Base training YAML config")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for trial outputs")
    parser.add_argument(
        "--preset",
        choices=["auto", "lora", "full"],
        default="auto",
        help="Search space preset. auto detects from finetuning_type in config.",
    )
    parser.add_argument("--num-trials", type=int, default=10,
                        help="Total number of trials")
    parser.add_argument("--num-initial", type=int, default=4,
                        help="Sobol initialization trials")
    parser.add_argument("--max-steps", type=int,
                        default=100, help="Steps per trial")
    args = parser.parse_args()

    best = run_bayesian_sweep(
        base_yaml=args.config,
        output_dir=args.output_dir,
        preset=args.preset,
        num_trials=args.num_trials,
        num_initial=args.num_initial,
        max_steps_per_trial=args.max_steps,
    )
    print(f"\nBest parameters:\n{json.dumps(best, indent=2)}")
