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

"""Bayesian hyperparameter sweep using Ax/BoTorch via AxClient."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml
from ax.service.ax_client import AxClient, ObjectiveProperties


# ---------------------------------------------------------------------------
# Parameter space
# ---------------------------------------------------------------------------

PARAMETERS = [
    {"name": "learning_rate", "type": "range", "bounds": [1e-5, 5e-4], "log_scale": True},
    {"name": "lora_rank", "type": "choice", "values": [8, 16, 32, 64], "is_ordered": True},
    {"name": "lora_dropout", "type": "range", "bounds": [0.0, 0.1]},
    {"name": "warmup_ratio", "type": "range", "bounds": [0.03, 0.15]},
    {"name": "weight_decay", "type": "range", "bounds": [0.0, 0.1]},
    {"name": "per_device_train_batch_size", "type": "choice", "values": [1, 2, 4], "is_ordered": True},
    {"name": "gradient_accumulation_steps", "type": "range", "bounds": [2, 16], "parameter_type": "int"},
]


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

    config.update(
        {
            "learning_rate": params["learning_rate"],
            "lora_rank": int(params["lora_rank"]),
            "lora_dropout": params.get("lora_dropout", 0.0),
            "warmup_ratio": params["warmup_ratio"],
            "weight_decay": params["weight_decay"],
            "per_device_train_batch_size": int(params["per_device_train_batch_size"]),
            "gradient_accumulation_steps": int(params["gradient_accumulation_steps"]),
            "output_dir": trial_dir,
            "max_steps": max_steps,
            "overwrite_output_dir": True,
            "report_to": "none",
            "save_steps": max_steps + 1,  # don't save intermediate checkpoints
            # Ensure eval fires during the short trial window
            "eval_strategy": "steps",
            "eval_steps": max(10, max_steps // 5),
            "val_size": min(config.get("val_size", 256), 256),
        }
    )

    trial_yaml = os.path.join(trial_dir, "config.yaml")
    with open(trial_yaml, "w") as f:
        yaml.dump(config, f)

    env = os.environ.copy()
    env.update(
        {
            "TOKENIZERS_PARALLELISM": "false",
            "GLIBC_TUNABLES": "glibc.rtld.optional_static_tls=524288",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )

    python = sys.executable
    cmd = [python, "src/train.py", trial_yaml]
    log_path = os.path.join(trial_dir, "train.log")

    print(f"\n[Trial {trial_index}] Starting with params: {params}")
    print(f"[Trial {trial_index}] Output dir: {trial_dir}")
    print(f"[Trial {trial_index}] Log: {log_path}")

    start = time.time()
    with open(log_path, "w") as log_f:
        proc = subprocess.run(
            cmd,
            cwd=str(Path(__file__).resolve().parents[2]),  # LlamaFactory root
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )

    elapsed = time.time() - start
    print(f"[Trial {trial_index}] Finished in {elapsed:.1f}s (exit code {proc.returncode})")

    if proc.returncode != 0:
        print(f"[Trial {trial_index}] FAILED — returning high loss sentinel 99.0")
        return 99.0

    # Parse eval loss from trainer_state.json
    state_path = os.path.join(trial_dir, "trainer_state.json")
    if not os.path.exists(state_path):
        print(f"[Trial {trial_index}] trainer_state.json not found — returning sentinel 99.0")
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
            print(f"[Trial {trial_index}] No eval loss found; using last train loss {loss:.4f}")
            return loss
        print(f"[Trial {trial_index}] No loss found — returning sentinel 99.0")
        return 99.0

    loss = eval_losses[-1]
    print(f"[Trial {trial_index}] Eval loss: {loss:.4f}")
    return loss


def run_bayesian_sweep(
    base_yaml: str,
    output_dir: str,
    num_trials: int = 10,
    num_initial: int = 4,
    max_steps_per_trial: int = 100,
) -> dict[str, Any]:
    """Run Bayesian sweep with Sobol initialization + BoTorch optimization."""
    ax_client = AxClient(enforce_sequential_optimization=True, verbose_logging=False)
    ax_client.create_experiment(
        name="qwen3_coder_next_lora_sweep",
        parameters=PARAMETERS,
        objectives={"eval_loss": ObjectiveProperties(minimize=True)},
        num_initialization_trials=num_initial,
    )

    results = []
    for trial_i in range(num_trials):
        params, trial_index = ax_client.get_next_trial()
        loss = run_trial(
            trial_index=trial_index,
            params=params,
            base_yaml=base_yaml,
            output_dir=output_dir,
            max_steps=max_steps_per_trial,
        )
        ax_client.complete_trial(trial_index=trial_index, raw_data={"eval_loss": loss})
        results.append({"trial_index": trial_index, "params": params, "eval_loss": loss})
        print(f"\n[Sweep] Trial {trial_i + 1}/{num_trials} done. Loss={loss:.4f}")

    best_params, values = ax_client.get_best_parameters()
    print(f"\n[Sweep] Best params: {best_params}")
    print(f"[Sweep] Best predicted loss: {values}")

    # Save results summary
    summary = {
        "best_params": best_params,
        "best_values": str(values),
        "all_trials": results,
    }
    summary_path = os.path.join(output_dir, "sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Sweep] Summary saved to {summary_path}")

    return best_params


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Bayesian hyperparameter sweep")
    parser.add_argument("--config", required=True, help="Base training YAML config")
    parser.add_argument("--output-dir", required=True, help="Directory for trial outputs")
    parser.add_argument("--num-trials", type=int, default=10, help="Total number of trials")
    parser.add_argument("--num-initial", type=int, default=4, help="Sobol initialization trials")
    parser.add_argument("--max-steps", type=int, default=100, help="Steps per trial")
    args = parser.parse_args()

    best = run_bayesian_sweep(
        base_yaml=args.config,
        output_dir=args.output_dir,
        num_trials=args.num_trials,
        num_initial=args.num_initial,
        max_steps_per_trial=args.max_steps,
    )
    print(f"\nBest parameters:\n{json.dumps(best, indent=2)}")
