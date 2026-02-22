# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's Transformers and Optimum library.
# https://github.com/huggingface/transformers/blob/v4.41.0/src/transformers/utils/quantization_config.py
# https://github.com/huggingface/optimum/blob/v1.20.0/optimum/gptq/data.py
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

import os
import random
from typing import TYPE_CHECKING, Any

import torch
from datasets import load_dataset
from transformers import BitsAndBytesConfig, EetqConfig, GPTQConfig, HqqConfig
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_utils import is_fsdp_enabled

from ...extras import logging
from ...extras.constants import FILEEXT2TYPE, QuantizationMethod
from ...extras.misc import check_version, get_current_device


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedTokenizer

    from ...hparams import ModelArguments


logger = logging.get_logger(__name__)


def _get_quantization_dataset(tokenizer: "PreTrainedTokenizer", model_args: "ModelArguments") -> list[dict[str, Any]]:
    r"""Prepare the tokenized dataset to perform AutoGPTQ. Do not use tensor output for JSON serialization."""
    if os.path.isfile(model_args.export_quantization_dataset):
        data_path = FILEEXT2TYPE.get(model_args.export_quantization_dataset.split(".")[-1], None)
        data_files = model_args.export_quantization_dataset
    else:
        data_path = model_args.export_quantization_dataset
        data_files = None

    dataset = load_dataset(
        path=data_path,
        data_files=data_files,
        split="train",
        cache_dir=model_args.cache_dir,
        token=model_args.hf_hub_token,
    )

    samples = []
    maxlen = model_args.export_quantization_maxlen
    for _ in range(model_args.export_quantization_nsamples):
        n_try = 0
        while True:
            if n_try > 100:
                raise ValueError("Cannot find satisfying example, considering decrease `export_quantization_maxlen`.")

            sample_idx = random.randint(0, len(dataset) - 1)
            sample: dict[str, torch.Tensor] = tokenizer(dataset[sample_idx]["text"], return_tensors="pt")
            n_try += 1
            if sample["input_ids"].size(1) > maxlen:
                break  # TODO: fix large maxlen

        word_idx = random.randint(0, sample["input_ids"].size(1) - maxlen - 1)
        input_ids = sample["input_ids"][:, word_idx : word_idx + maxlen]
        attention_mask = sample["attention_mask"][:, word_idx : word_idx + maxlen]
        samples.append({"input_ids": input_ids.tolist(), "attention_mask": attention_mask.tolist()})

    return samples


def configure_quantization(
    config: "PretrainedConfig",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    is_trainable: bool,
    init_kwargs: dict[str, Any],
) -> None:
    r"""Priority: PTQ-quantized (train/infer) > AutoGPTQ (export) > On-the-fly quantization (train/infer)."""
    if getattr(config, "quantization_config", None):  # ptq
        if model_args.quantization_bit is not None:
            logger.warning_rank0("`quantization_bit` will not affect on the PTQ-quantized models.")

        quantization_config: dict[str, Any] = getattr(config, "quantization_config", None)
        quant_method = quantization_config.get("quant_method", "")

        if quant_method not in (QuantizationMethod.MXFP4, QuantizationMethod.FP8) and (
            is_deepspeed_zero3_enabled() or is_fsdp_enabled()
        ):
            # mxfp4 will dequant the model weights
            raise ValueError("DeepSpeed ZeRO-3 or FSDP is incompatible with PTQ-quantized models.")

        if quant_method == QuantizationMethod.MXFP4:
            from transformers import Mxfp4Config

            quant_config = Mxfp4Config(dequantize=True)
            init_kwargs["quantization_config"] = quant_config
            init_kwargs["ignore_mismatched_sizes"] = True

        if quant_method == QuantizationMethod.FP8:
            from transformers import FineGrainedFP8Config

            quant_config = FineGrainedFP8Config(dequantize=True)
            init_kwargs["quantization_config"] = quant_config
            init_kwargs["ignore_mismatched_sizes"] = True

        if quant_method == QuantizationMethod.GPTQ:
            check_version("gptqmodel>=2.0.0", mandatory=True)
            quantization_config.pop("disable_exllama", None)  # remove deprecated args
            quantization_config["use_exllama"] = False  # disable exllama

        if quant_method == QuantizationMethod.AWQ:
            check_version("autoawq", mandatory=True)

        if quant_method == QuantizationMethod.AQLM:
            check_version("aqlm>=1.1.0", mandatory=True)
            quantization_config["bits"] = 2

        quant_bits = quantization_config.get("bits", "?")
        logger.info_rank0(f"Loading {quant_bits}-bit {quant_method.upper()}-quantized model.")

    elif model_args.export_quantization_bit is not None:  # gptqmodel
        if model_args.export_quantization_bit not in [8, 4, 3, 2]:
            raise ValueError("AutoGPTQ only accepts 2/3/4/8-bit quantization.")

        check_version("optimum>=1.24.0", mandatory=True)
        check_version("gptqmodel>=2.0.0", mandatory=True)
        from accelerate.utils import get_max_memory

        if getattr(config, "model_type", None) == "chatglm":
            raise ValueError("ChatGLM model is not supported yet.")

        try:
            from optimum.gptq import utils as gq_utils

            if "language_model.model.layers" not in gq_utils.BLOCK_PATTERNS:
                gq_utils.BLOCK_PATTERNS.insert(0, "language_model.model.layers")
        except ImportError:
            pass

        block_name_to_quantize = None
        if getattr(config, "model_type", None) in ["gemma3", "paligemma"]:
            block_name_to_quantize = "language_model.model.layers"

        init_kwargs["quantization_config"] = GPTQConfig(
            bits=model_args.export_quantization_bit,
            tokenizer=tokenizer,
            dataset=_get_quantization_dataset(tokenizer, model_args),
            block_name_to_quantize=block_name_to_quantize,
        )
        init_kwargs["device_map"] = "auto"
        init_kwargs["max_memory"] = get_max_memory()
        model_args.compute_dtype = torch.float16  # force fp16 for gptqmodel
        logger.info_rank0(f"Quantizing model to {model_args.export_quantization_bit} bit with GPTQModel.")

    elif model_args.quantization_bit is not None:  # on-the-fly
        if model_args.quantization_method == QuantizationMethod.BNB:
            if model_args.quantization_bit == 8:
                check_version("bitsandbytes>=0.37.0", mandatory=True)
                init_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            elif model_args.quantization_bit == 4:
                check_version("bitsandbytes>=0.39.0", mandatory=True)
                init_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=model_args.compute_dtype,
                    bnb_4bit_use_double_quant=model_args.double_quantization,
                    bnb_4bit_quant_type=model_args.quantization_type,
                    bnb_4bit_quant_storage=model_args.compute_dtype,  # crucial for fsdp+qlora
                )
            else:
                raise ValueError("Bitsandbytes only accepts 4-bit or 8-bit quantization.")

            # Do not assign device map if:
            # 1. deepspeed zero3 or fsdp (train)
            # 2. auto quantization device map (inference)
            if is_deepspeed_zero3_enabled() or is_fsdp_enabled() or model_args.quantization_device_map == "auto":
                if model_args.quantization_bit != 4:
                    raise ValueError("Only 4-bit quantized model can use fsdp+qlora or auto device map.")

                check_version("bitsandbytes>=0.43.0", mandatory=True)
            else:
                # CPU-first loading: tensors materialize to RAM in bf16, Params4bit built on CPU
                # (unquantized). loader.py then calls model.to(cuda) which triggers
                # Params4bit.to(cuda) -> 4-bit quantization per-layer, never exceeding VRAM.
                init_kwargs["device_map"] = {"": "cpu"}
            logger.info_rank0(f"Quantizing model to {model_args.quantization_bit} bit with bitsandbytes (CPU-first).")
        elif model_args.quantization_method == QuantizationMethod.HQQ:
            if model_args.quantization_bit not in [8, 6, 5, 4, 3, 2, 1]:
                raise ValueError("HQQ only accepts 1/2/3/4/5/6/8-bit quantization.")

            if is_deepspeed_zero3_enabled() or is_fsdp_enabled():
                raise ValueError("HQQ quantization is incompatible with DeepSpeed ZeRO-3 or FSDP.")

            check_version("hqq", mandatory=True)
            init_kwargs["quantization_config"] = HqqConfig(
                nbits=model_args.quantization_bit, quant_zero=False, quant_scale=False, axis=0
            )  # use ATEN kernel (axis=0) for performance
            logger.info_rank0(f"Quantizing model to {model_args.quantization_bit} bit with HQQ.")
        elif model_args.quantization_method == QuantizationMethod.EETQ:
            if model_args.quantization_bit != 8:
                raise ValueError("EETQ only accepts 8-bit quantization.")

            if is_deepspeed_zero3_enabled() or is_fsdp_enabled():
                raise ValueError("EETQ quantization is incompatible with DeepSpeed ZeRO-3 or FSDP.")

            check_version("eetq", mandatory=True)
            init_kwargs["quantization_config"] = EetqConfig()
            logger.info_rank0(f"Quantizing model to {model_args.quantization_bit} bit with EETQ.")


def _should_move_bnb_model_to_gpu(model_args: "ModelArguments") -> bool:
    """Return True when BnB 4-bit CPU-first loading was used and the model needs .to(cuda)."""
    return (
        getattr(model_args, "quantization_bit", None) == 4
        and getattr(model_args, "quantization_method", None) == QuantizationMethod.BNB
        and getattr(model_args, "quantization_device_map", None) != "auto"
    )


def _quantize_qwen3next_experts(model: "torch.nn.Module", model_args: "ModelArguments") -> None:
    """Quantize Qwen3NextExperts 3D parameters (gate_up_proj, down_proj) via BnB NF4.

    Qwen3Next stores expert weights as 3D nn.Parameter [num_experts, out, in] rather than
    individual nn.Linear modules, so replace_with_bnb_linear skips them.  We handle them
    here by quantizing each expert's 2-D slice independently and patching the forward.
    """
    try:
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextExperts
    except ImportError:
        return

    import bitsandbytes as bnb

    bnb_cfg = model_args.quantization_type if hasattr(model_args, "quantization_type") else "nf4"
    double_quant = bool(getattr(model_args, "double_quantization", True))

    found = 0
    for mod_name, module in model.named_modules():
        if not isinstance(module, Qwen3NextExperts):
            continue
        found += 1
        for attr in ("gate_up_proj", "down_proj"):
            p = getattr(module, attr, None)
            if p is None or not isinstance(p, torch.nn.Parameter):
                continue
            N, O, I = p.shape  # [num_experts, out_features, in_features]
            # Quantize each expert separately so slicing quant_state is trivial.
            q_list, qs_list = [], []
            data_cpu = p.data.cpu()  # ensure CPU, no-op if already there
            for expert_idx in range(N):
                slice_2d = data_cpu[expert_idx].contiguous()  # [O, I]
                q, qs = bnb.functional.quantize_4bit(
                    slice_2d,
                    blocksize=64,
                    compress_statistics=double_quant,
                    quant_type=bnb_cfg,
                    quant_storage=torch.uint8,
                )
                q_list.append(q)   # stays on CPU for now
                qs_list.append(qs)
            # Remove the large bf16 parameter, store quantized data as buffers/attrs.
            del module._parameters[attr]
            # Stack quantized bytes — shape [N * O // 2, I] packed in last dim by BnB convention
            stacked = torch.cat(q_list, dim=0)           # uint8, packed
            module.register_buffer(f"_bnb_{attr}_quant", stacked)
            setattr(module, f"_bnb_{attr}_qs_list", qs_list)
            setattr(module, f"_bnb_{attr}_shape", (N, O, I))

        # Patch the forward once, only if we actually modified weights.
        _patch_qwen3next_experts_forward(module)

    if found:
        logger.info_rank0(
            f"Quantized Qwen3NextExperts in {found} block(s) using BnB {bnb_cfg.upper()} "
            f"(double_quant={double_quant}). Expert weights are now 4-bit on CPU."
        )


def _patch_qwen3next_experts_forward(module: "torch.nn.Module") -> None:
    """Replace Qwen3NextExperts.forward with a quantized-aware version."""
    import torch.nn.functional as F
    import bitsandbytes as bnb

    def quantized_forward(self, hidden_states, top_k_index, top_k_weights):
        device = hidden_states.device
        final_hidden_states = torch.zeros_like(hidden_states)

        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        gu_qs = self._bnb_gate_up_proj_qs_list
        dn_qs = self._bnb_down_proj_qs_list
        N_gu = self._bnb_gate_up_proj_shape[0]
        N_dn = self._bnb_down_proj_shape[0]

        gu_buf = self._bnb_gate_up_proj_quant  # on CUDA (moved by buffer loop in loader.py)
        dn_buf = self._bnb_down_proj_quant
        gu_bytes_per_expert = gu_buf.shape[0] // N_gu
        dn_bytes_per_expert = dn_buf.shape[0] // N_dn

        for expert_idx_t in expert_hit:
            expert_idx = expert_idx_t[0].item()
            if expert_idx >= self.num_experts:
                continue

            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]

            # QuantState.to() is in-place; moves tensors inside qs to device once, no-op thereafter.
            gu_qs[expert_idx].to(device)
            gu_q_slice = gu_buf[expert_idx * gu_bytes_per_expert : (expert_idx + 1) * gu_bytes_per_expert]
            gu_w = bnb.functional.dequantize_4bit(gu_q_slice, gu_qs[expert_idx]).to(hidden_states.dtype)
            gate, up = F.linear(current_state, gu_w).chunk(2, dim=-1)
            current_hidden = self.act_fn(gate) * up
            del gu_w

            dn_qs[expert_idx].to(device)
            dn_q_slice = dn_buf[expert_idx * dn_bytes_per_expert : (expert_idx + 1) * dn_bytes_per_expert]
            dn_w = bnb.functional.dequantize_4bit(dn_q_slice, dn_qs[expert_idx]).to(hidden_states.dtype)
            current_hidden = F.linear(current_hidden, dn_w)
            del dn_w

            current_hidden = current_hidden * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden.to(final_hidden_states.dtype))

        return final_hidden_states

    import types
    module.forward = types.MethodType(quantized_forward, module)


