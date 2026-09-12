#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Original ModelOpt fixture: dense KDA, routed KDA, and routed sparse MLA."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile

MARKER = 'tinyglm-modelopt-v1'


def build(config_path, output):
    data = Path(config_path).read_bytes()
    pin = json.loads((Path(__file__).parent / 'candidate.json').read_text())
    if hashlib.sha256(data).hexdigest() != pin['config_sha256']:
        raise ValueError('NVIDIA checkpoint configuration does not match the pinned hash')
    quant = json.loads(data)['quantization_config']
    # The synthetic causal model uses model.layers, not the multimodal wrapper.
    # Preserve the publisher's precision boundary, including BF16 shared experts.
    quant['ignore'] = ['lm_head', 'model.embed_tokens', 'model.layers.*.self_attn*',
                       'model.layers.*.mlp.gate', 'model.layers.*.mlp.shared_experts*']
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new empty fixture output directory')
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location('tiny', root / 'scripts/make_tinyglm.py')
    tiny = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tiny)
    with tempfile.TemporaryDirectory() as tmp:
        snapshot = tiny.build(Path(tmp), 16, 256, 32768)
        config = json.loads((snapshot / 'config.json').read_text())
        config['_sparkglm_fixture'] = MARKER
        config['quantization_config'] = quant
        text = config['text_config']
        text.update(num_hidden_layers=3, first_k_dense_replace=1,
                    mlp_layer_types=['dense', 'sparse', 'sparse'],
                    layer_types=['linear_attention', 'linear_attention',
                                 'deepseek_sparse_attention'])
        (snapshot / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        (snapshot / 'quantization_config.json').write_text(json.dumps(quant, indent=2) + '\n')
        shutil.copytree(snapshot, output, dirs_exist_ok=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(build(args.config, args.output))
