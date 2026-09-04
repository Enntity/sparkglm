#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Create a weightless, kernel-faithful miniature GLM-5.3 model directory.

The checkpoint is used with vLLM's ``--load-format dummy``. It preserves the
production per-layer dimensions and operations but deliberately reduces depth,
vocabulary, and routed-expert count. It is not a language model and its outputs
must never be evaluated for quality.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SCHEMA_REVISION = "tinyglm-v1"


def revision(experts: int, vocab_size: int, max_length: int) -> str:
    return f"{SCHEMA_REVISION}-e{experts}-v{vocab_size}-l{max_length}"


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _tokenizer(vocab_size: int) -> dict:
    special = ["<pad>", "<eos>", "<unk>", "<bos>"]
    words = [
        "the",
        "a",
        "to",
        "of",
        "and",
        "in",
        "is",
        "for",
        "test",
        "token",
        "prefill",
        "decode",
        "spark",
        "glm",
    ]
    tokens = special + words
    tokens.extend(f"t{i}" for i in range(vocab_size - len(tokens)))
    vocab = {token: index for index, token in enumerate(tokens)}
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": index,
                "content": token,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
            for index, token in enumerate(special)
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"},
    }


def model_config(experts: int, vocab_size: int, max_length: int) -> dict:
    # Two real GLM layer species: KDA+sparse-MoE and sparse-MLA+sparse-MoE.
    # Dimensions that control native kernel selection match GLM-5.3 exactly.
    layer_types = ["linear_attention", "deepseek_sparse_attention"]
    return {
        "architectures": ["Glm5NextForCausalLM"],
        "model_type": "glm5_next",
        "dtype": "bfloat16",
        "tie_word_embeddings": False,
        "quantization_config": {
            "bits": 4,
            "codebook": "mcg",
            "head_bits": 16,
            "quant_method": "exl3",
            "scope": "glm53_routed_experts_only",
            "version": "tinyglm-v1",
        },
        "text_config": {
            "model_type": "glm5_next_text",
            "vocab_size": vocab_size,
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_hidden_layers": len(layer_types),
            "num_attention_heads": 64,
            "num_key_value_heads": 64,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-5,
            "max_position_embeddings": max_length,
            "tie_word_embeddings": False,
            "moe_intermediate_size": 2048,
            "moe_router_dtype": "float32",
            "scoring_func": "sigmoid",
            "n_routed_experts": experts,
            "num_experts_per_tok": 8,
            "n_shared_experts": 1,
            "routed_scaling_factor": 2.5,
            "norm_topk_prob": True,
            "first_k_dense_replace": 0,
            "mlp_layer_types": ["sparse", "sparse"],
            "layer_types": layer_types,
            "linear_attn_config": {
                "head_dim": 128,
                "num_heads": 64,
                "short_conv_kernel_size": 4,
                "gate_lower_bound": -5.0,
            },
            "q_lora_rank": 1536,
            "kv_lora_rank": 512,
            "qk_head_dim": 256,
            "qk_nope_head_dim": 256,
            "qk_rope_head_dim": 0,
            "v_head_dim": 256,
            "mla_use_nope": True,
            "index_head_dim": 128,
            "index_topk": 2048,
            "index_n_heads": 32,
            "index_kpool": 4,
            "index_kpool_compress": True,
            "index_kpool_always_select_tail": True,
            "mhc": True,
            "hc_mult": 4,
            "hc_eps": 1e-6,
            "hc_sinkhorn_iters": 20,
            "swiglu_limit": 10.0,
            "n_group": 1,
            "topk_group": 1,
            "topk_method": "noaux_tc",
            "num_nextn_predict_layers": 0,
            "pad_token_id": 0,
            "eos_token_id": 1,
            "bos_token_id": 3,
            "use_cache": True,
        },
    }


def build(output: Path, experts: int, vocab_size: int, max_length: int) -> Path:
    if experts < 8:
        raise ValueError("experts must be at least the model's top-8 routing width")
    if vocab_size < 32:
        raise ValueError("vocab-size must be at least 32")
    if max_length < 2048:
        raise ValueError("max-length must be at least the sparse index top-k")

    snapshot_revision = revision(experts, vocab_size, max_length)
    snapshot = output / "snapshots" / snapshot_revision
    snapshot.mkdir(parents=True, exist_ok=True)
    (output / "refs").mkdir(parents=True, exist_ok=True)
    (output / "refs" / "main").write_text(snapshot_revision + "\n")

    config = model_config(experts, vocab_size, max_length)
    _write_json(snapshot / "config.json", config)
    _write_json(snapshot / "quantization_config.json", config["quantization_config"])
    _write_json(snapshot / "tokenizer.json", _tokenizer(vocab_size))
    _write_json(
        snapshot / "tokenizer_config.json",
        {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "model_max_length": max_length,
            "pad_token": "<pad>",
            "eos_token": "<eos>",
            "unk_token": "<unk>",
            "bos_token": "<bos>",
            "chat_template": (
                "{% for message in messages %}{{ message['role'] + ': ' + "
                "message['content'] + '\\n' }}{% endfor %}assistant:"
            ),
        },
    )
    _write_json(
        snapshot / "special_tokens_map.json",
        {
            "pad_token": "<pad>",
            "eos_token": "<eos>",
            "unk_token": "<unk>",
            "bos_token": "<bos>",
        },
    )
    _write_json(
        snapshot / "generation_config.json",
        {"do_sample": False, "pad_token_id": 0, "eos_token_id": 1},
    )
    (snapshot / "README.md").write_text(
        "# tinyGLM\n\nSynthetic, weightless GLM-5.3 kernel-integration fixture. "
        "Use only with `--load-format dummy` and `SPARKGLM_TINY_DUMMY=1`. "
        "Outputs are meaningless.\n"
    )
    return snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=32768)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot = build(args.output, args.experts, args.vocab_size, args.max_length)
    print(snapshot)


if __name__ == "__main__":
    main()
