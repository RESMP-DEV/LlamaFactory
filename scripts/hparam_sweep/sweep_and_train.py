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

"""Orchestrate Bayesian sweep → final full training with best parameters."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from run_sweep import run_bayesian_sweep


def launch_final_training(
    base_yaml: str,
    best_params: dict,
    output_dir: str,
    train_epochs: float = 3.0,
) -> None:
    """Run full training with best hyperparameters from sweep."""
    with open(base_yaml) as f:
        config = yaml.safe_load(f)

    final_overrides = {
        "learning_rate": best_params["learning_rate"],
        "warmup_ratio": best_params["warmup_ratio"],
        "weight_decay": best_params["weight_decay"],
        "per_device_train_batch_size": int(best_params["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(best_params["gradient_accumulation_steps"]),
        "output_dir": output_dir,
        "num_train_epochs": train_epochs,
        "max_steps": -1,
        "overwrite_output_dir": True,
        "save_steps": 500,
        "report_to": "none",
    }

    if "lora_rank" in best_params:
        final_overrides["lora_rank"] = int(best_params["lora_rank"])

    if "lora_dropout" in best_params:
        final_overrides["lora_dropout"] = best_params["lora_dropout"]

    config.update(final_overrides)

    final_yaml = os.path.join(output_dir, "final_config.yaml")
    os.makedirs(output_dir, exist_ok=True)
    with open(final_yaml, "w") as f:
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
    cmd = [python, "src/train.py", final_yaml]
    root = str(Path(__file__).resolve().parents[2])
    log_path = os.path.join(output_dir, "final_train.log")

    print(f"\n[Final Training] Config: {final_yaml}")
    print(f"[Final Training] Output dir: {output_dir}")
    print(f"[Final Training] Log: {log_path}")
    print(f"[Final Training] Params: {best_params}")

    with open(log_path, "w") as log_f:
        proc = subprocess.run(
            cmd,
            cwd=root,
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )

    if proc.returncode != 0:
        print(
            f"[Final Training] FAILED with exit code {proc.returncode}. Check {log_path}")
        sys.exit(1)

    print(f"[Final Training] Complete! Model saved to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep + train pipeline")
    parser.add_argument("--config", required=True,
                        help="Base training YAML config")
    parser.add_argument("--num-trials", type=int,
                        default=8, help="Sweep trials")
    parser.add_argument("--num-initial", type=int, default=3,
                        help="Sobol initialization trials")
    parser.add_argument(
        "--preset",
        choices=["auto", "lora", "full"],
        default="auto",
        help="Search space preset. auto detects from finetuning_type in config.",
    )
    parser.add_argument(
        "--output-dir", default="saves/qwen3-coder-next/sweep_run", help="Root output dir")
    parser.add_argument("--train-epochs", type=float,
                        default=3.0, help="Epochs for final training")
    parser.add_argument("--max-steps-per-trial", type=int,
                        default=100, help="Steps per sweep trial")
    parser.add_argument("--skip-sweep", action="store_true",
                        help="Skip sweep; load best_params.json")
    args = parser.parse_args()

    sweep_dir = os.path.join(args.output_dir, "sweep")
    final_dir = os.path.join(args.output_dir, "final")
    best_params_path = os.path.join(sweep_dir, "best_params.json")

    if args.skip_sweep:
        print("[Pipeline] Skipping sweep; loading best params from",
              best_params_path)
        with open(best_params_path) as f:
            best_params = json.load(f)
    else:
        print("[Pipeline] Phase 1: Bayesian hyperparameter sweep")
        best_params = run_bayesian_sweep(
            base_yaml=args.config,
            output_dir=sweep_dir,
            preset=args.preset,
            num_trials=args.num_trials,
            num_initial=args.num_initial,
            max_steps_per_trial=args.max_steps_per_trial,
        )
        os.makedirs(sweep_dir, exist_ok=True)
        with open(best_params_path, "w") as f:
            json.dump(best_params, f, indent=2)
        print(f"[Pipeline] Best params saved to {best_params_path}")

    print("\n[Pipeline] Phase 2: Final training with best parameters")
    launch_final_training(
        base_yaml=args.config,
        best_params=best_params,
        output_dir=final_dir,
        train_epochs=args.train_epochs,
    )
    print("\n[Pipeline] Pipeline complete!")


if __name__ == "__main__":
    main()
