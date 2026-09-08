#!/usr/bin/env python3
"""Install DFlash2 onto the glm53-flash vLLM image (idempotent)."""

from __future__ import annotations

from pathlib import Path

SITE = Path("/usr/local/lib/python3.12/dist-packages/vllm")
OPT = Path("/opt/glm53")


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if new in text:
        return
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{path}: expected one patch target, found {n}: {old!r}")
    path.write_text(text.replace(old, new))


def main() -> None:
    src_model = OPT / "qwen3_dflash2.py"
    src_spec = OPT / "dflash2_speculator.py"
    dst_model = SITE / "model_executor/models/qwen3_dflash2.py"
    dst_dir = SITE / "v1/worker/gpu/spec_decode/dflash2"
    dst_spec = dst_dir / "speculator.py"
    dst_init = dst_dir / "__init__.py"
    dst_model.write_text(src_model.read_text())
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_spec.write_text(src_spec.read_text())
    if not dst_init.exists():
        dst_init.write_text("# SPDX-License-Identifier: Apache-2.0\n")

    qwen = SITE / "model_executor/models/qwen3_dflash.py"
    replace_once(
        qwen,
        'def _dflash_layer_causal(config: Qwen3Config, layer_idx: int) -> bool:\n'
        '    """``dflash_config.causal`` overrides all layers; else only SWA layers causal."""\n'
        '    override = (getattr(config, "dflash_config", None) or {}).get("causal")\n',
        'def _dflash_layer_causal(config: Qwen3Config, layer_idx: int) -> bool:\n'
        '    """Honor checkpoint ``is_causal`` (incoai DFlash2: false) before SWA=causal."""\n'
        '    is_causal = getattr(config, "is_causal", None)\n'
        "    if is_causal is not None:\n"
        "        return bool(is_causal)\n"
        '    override = (getattr(config, "dflash_config", None) or {}).get("causal")\n',
    )
    replace_once(
        qwen,
        "@support_torch_compile\n"
        "class DFlashQwen3Model(nn.Module):\n"
        "    hf_to_vllm_mapper = WeightsMapper(\n",
        "@support_torch_compile\n"
        "class DFlashQwen3Model(nn.Module):\n"
        "    decoder_layer_cls = DFlashQwen3DecoderLayer\n"
        "    hf_to_vllm_mapper = WeightsMapper(\n",
    )
    replace_once(
        qwen,
        "        self.layers = nn.ModuleList(\n"
        "            [\n"
        "                DFlashQwen3DecoderLayer(\n",
        "        self.layers = nn.ModuleList(\n"
        "            [\n"
        "                self.decoder_layer_cls(\n",
    )
    replace_once(
        qwen,
        "class DFlashQwen3ForCausalLM(Qwen3ForCausalLM):\n"
        "    def __init__(self, *, vllm_config: VllmConfig, prefix: str = \"\"):\n",
        "class DFlashQwen3ForCausalLM(Qwen3ForCausalLM):\n"
        "    model_cls = DFlashQwen3Model\n"
        "\n"
        "    def __init__(self, *, vllm_config: VllmConfig, prefix: str = \"\"):\n",
    )
    replace_once(
        qwen,
        "        self.model = DFlashQwen3Model(\n"
        "            vllm_config=vllm_config,\n"
        '            prefix=maybe_prefix(prefix, "model"),\n'
        "            start_layer_id=target_layer_num,\n"
        "        )\n",
        "        self.model = self.model_cls(\n"
        "            vllm_config=vllm_config,\n"
        '            prefix=maybe_prefix(prefix, "model"),\n'
        "            start_layer_id=target_layer_num,\n"
        "        )\n",
    )

    # ModelOpt MXFP8 serializes every draft linear (including DFlash2's
    # grouped-conv projections and selector). Preserve the quantized K/V
    # projection's E8M0 block scales when the per-layer tensors are fused.
    replace_once(
        qwen,
        "        self.norm = RMSNorm(\n"
        "            self.config.hidden_size,\n"
        "            eps=self.config.rms_norm_eps,\n"
        "        )\n"
        "\n"
        "    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:\n",
        "        self.norm = RMSNorm(\n"
        "            self.config.hidden_size,\n"
        "            eps=self.config.rms_norm_eps,\n"
        "        )\n"
        "        self._fused_kv_linear = nn.Module()\n"
        "        self._fused_kv_quant_method = None\n"
        "        self._fused_kv_weight: torch.Tensor | None = None\n"
        "        self._fused_kv_weight_scale: torch.Tensor | None = None\n"
        "\n"
        "    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:\n",
    )
    replace_once(
        qwen,
        "        self._fused_kv_weight = torch.cat(kv_weights, dim=0)\n"
        "        if has_bias:\n",
        "        self._fused_kv_weight = torch.cat(kv_weights, dim=0)\n"
        "\n"
        "        from vllm.model_executor.layers.quantization.modelopt import (\n"
        "            ModelOptMxFp8LinearMethod,\n"
        "        )\n"
        "\n"
        "        quant_method = layers_attn[0].qkv_proj.quant_method\n"
        "        if isinstance(quant_method, ModelOptMxFp8LinearMethod):\n"
        "            if not all(\n"
        "                isinstance(a.qkv_proj.quant_method, ModelOptMxFp8LinearMethod)\n"
        "                for a in layers_attn\n"
        "            ):\n"
        "                raise ValueError(\n"
        '                    "Every DFlash attention layer must use the same MXFP8 "\n'
        '                    "linear format for the fused context K/V projection."\n'
        "                )\n"
        "            kv_scales = [a.qkv_proj.weight_scale[a.q_size :] for a in layers_attn]\n"
        "            self._fused_kv_weight_scale = torch.cat(kv_scales, dim=0)\n"
        "        else:\n"
        "            self._fused_kv_weight_scale = None\n"
        "        if has_bias:\n",
    )
    replace_once(
        qwen,
        "        self._attn_layers = [layer.self_attn.attn for layer in self.layers]\n"
        "\n"
        "    def _project_context_kv(\n",
        "        self._attn_layers = [layer.self_attn.attn for layer in self.layers]\n"
        "\n"
        "    def process_weights_after_loading(self) -> None:\n"
        "        from vllm.model_executor.layers.quantization.modelopt import (\n"
        "            ModelOptMxFp8LinearMethod,\n"
        "        )\n"
        "\n"
        "        quant_method = self.layers[0].self_attn.qkv_proj.quant_method\n"
        "        if not isinstance(quant_method, ModelOptMxFp8LinearMethod):\n"
        "            return\n"
        "        if self._fused_kv_weight is None or self._fused_kv_weight_scale is None:\n"
        "            raise RuntimeError(\n"
        '                "The DFlash MXFP8 context projection requires serialized K/V "\n'
        '                "weights and block scales to be fused during checkpoint loading."\n'
        "            )\n"
        "        output_size, input_size = self._fused_kv_weight.shape\n"
        "        self._fused_kv_linear.input_size_per_partition = input_size\n"
        "        self._fused_kv_linear.output_size_per_partition = output_size\n"
        "        self._fused_kv_linear.logical_widths = [output_size]\n"
        "        self._fused_kv_linear.register_parameter(\n"
        '            "weight", nn.Parameter(self._fused_kv_weight, requires_grad=False)\n'
        "        )\n"
        "        self._fused_kv_linear.register_parameter(\n"
        '            "weight_scale",\n'
        "            nn.Parameter(self._fused_kv_weight_scale, requires_grad=False),\n"
        "        )\n"
        "        quant_method.process_weights_after_loading(self._fused_kv_linear)\n"
        "        self._fused_kv_quant_method = quant_method\n"
        "        self._fused_kv_weight = None\n"
        "        self._fused_kv_weight_scale = None\n"
        "\n"
        "    def _project_context_kv(\n",
    )
    replace_once(
        qwen,
        "        all_kv_flat = F.linear(\n"
        "            normed_context_states, self._fused_kv_weight, self._fused_kv_bias\n"
        "        )\n",
        "        if self._fused_kv_quant_method is None:\n"
        "            if self._fused_kv_weight is None:\n"
        '                raise RuntimeError("DFlash context K/V projection is not initialized.")\n'
        "            all_kv_flat = F.linear(\n"
        "                normed_context_states, self._fused_kv_weight, self._fused_kv_bias\n"
        "            )\n"
        "        else:\n"
        "            all_kv_flat = self._fused_kv_quant_method.apply(\n"
        "                self._fused_kv_linear,\n"
        "                normed_context_states,\n"
        "                self._fused_kv_bias,\n"
        "            )\n",
    )
    replace_once(
        qwen,
        "        self.model.precompute_and_store_context_kv(\n"
        "            context_states, context_positions, context_slot_mapping\n"
        "        )\n"
        "\n"
        "    def combine_hidden_states(\n",
        "        self.model.precompute_and_store_context_kv(\n"
        "            context_states, context_positions, context_slot_mapping\n"
        "        )\n"
        "\n"
        "    def process_weights_after_loading(self) -> None:\n"
        "        self.model.process_weights_after_loading()\n"
        "\n"
        "    def combine_hidden_states(\n",
    )

    registry = SITE / "model_executor/models/registry.py"
    replace_once(
        registry,
        '    "DFlashDraftModel": ("qwen3_dflash", "DFlashQwen3ForCausalLM"),\n',
        '    "DFlashDraftModel": ("qwen3_dflash", "DFlashQwen3ForCausalLM"),\n'
        '    "DFlash2DraftModel": ("qwen3_dflash2", "DFlash2Qwen3ForCausalLM"),\n',
    )

    dflash_utils = SITE / "v1/worker/gpu/spec_decode/dflash/utils.py"
    replace_once(
        dflash_utils,
        "    speculative_config = vllm_config.speculative_config\n"
        "    assert speculative_config is not None\n"
        "    draft_model_config = speculative_config.draft_model_config\n"
        "    # Select an attention backend that supports the drafter's attention: mixing\n"
        "    # a non-causal layer onto a causal-only backend would fail.\n"
        "    draft_vllm_config = replace(\n"
        "        vllm_config,\n"
        "        attention_config=replace(\n"
        "            vllm_config.attention_config,\n"
        "            use_non_causal=dflash_has_any_non_causal(draft_model_config.hf_config),\n"
        "            backend=speculative_config.attention_backend,\n"
        "        ),\n"
        "        cache_config=(\n"
        "            replace(\n"
        "                vllm_config.cache_config,\n"
        "                cache_dtype=speculative_config.kv_cache_dtype,\n"
        "            )\n"
        "            if speculative_config.kv_cache_dtype is not None\n"
        "            else vllm_config.cache_config\n"
        "        ),\n"
        "    )\n",
        "    speculative_config = vllm_config.speculative_config\n"
        "    assert speculative_config is not None\n"
        "    draft_model_config = speculative_config.draft_model_config\n"
        "    # Dense DFlash2 attention cannot use the target's MLA-only fp8_ds_mla\n"
        "    # layout, and SM121 has no FA3/FA4 for plain FP8 KV. Keep draft KV in\n"
        "    # the model dtype unless speculative_config.kv_cache_dtype is set.\n"
        "    draft_kv = speculative_config.kv_cache_dtype\n"
        '    if draft_kv is None and vllm_config.cache_config.cache_dtype in (\n'
        '        "fp8_ds_mla",\n'
        '        "fp8",\n'
        '        "fp8_e4m3",\n'
        '        "fp8_e5m2",\n'
        '        "nvfp4",\n'
        "    ):\n"
        '        draft_kv = "auto"\n'
        "    # Select an attention backend that supports the drafter's attention: mixing\n"
        "    # a non-causal layer onto a causal-only backend would fail.\n"
        "    draft_vllm_config = replace(\n"
        "        vllm_config,\n"
        "        attention_config=replace(\n"
        "            vllm_config.attention_config,\n"
        "            use_non_causal=dflash_has_any_non_causal(draft_model_config.hf_config),\n"
        "            backend=speculative_config.attention_backend,\n"
        "        ),\n"
        "        cache_config=(\n"
        "            replace(\n"
        "                vllm_config.cache_config,\n"
        "                cache_dtype=draft_kv,\n"
        "            )\n"
        "            if draft_kv is not None\n"
        "            else vllm_config.cache_config\n"
        "        ),\n"
        "    )\n",
    )

    spec_init = SITE / "v1/worker/gpu/spec_decode/__init__.py"
    replace_once(
        spec_init,
        '    if speculative_config.method == "dflash":\n'
        "        from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (\n"
        "            DFlashSpeculator,\n"
        "        )\n"
        "\n"
        "        return DFlashSpeculator(vllm_config, device)\n",
        '    if speculative_config.method == "dflash":\n'
        '        if "DFlash2DraftModel" in speculative_config.draft_model_config.architectures:\n'
        "            from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import (\n"
        "                DFlash2Speculator,\n"
        "            )\n"
        "\n"
        "            return DFlash2Speculator(vllm_config, device)\n"
        "        from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (\n"
        "            DFlashSpeculator,\n"
        "        )\n"
        "\n"
        "        return DFlashSpeculator(vllm_config, device)\n",
    )

    compile(dst_model.read_text(), str(dst_model), "exec")
    compile(dst_spec.read_text(), str(dst_spec), "exec")
    compile(qwen.read_text(), str(qwen), "exec")
    compile(dflash_utils.read_text(), str(dflash_utils), "exec")
    compile(spec_init.read_text(), str(spec_init), "exec")
    print("dflash2 overlay installed")


if __name__ == "__main__":
    main()
