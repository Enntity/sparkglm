#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""G0 only: transform bounds, launch ownership, and candidate build wiring."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'research/experiments/exl3-direct-epilogue'
sys.path.insert(0, str(HERE))
import patch
spec = importlib.util.spec_from_file_location('direct_epilogue_build', HERE / 'build.py')
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


def main():
    source = (ROOT / 'overlay/exl3_fat_gemm.cu').read_text()
    result = patch.transform(source)
    for bad in (source + '\n', result):
        try:
            patch.transform(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('drift or double application was accepted')
    start, end = source.index(patch.GROUP_START), source.index(patch.GROUP_END)
    assert result.startswith(source[:start]), 'standalone oracle changed'
    assert result[result.index(patch.GROUP_END):] == source[end:], 'dispatch/activation changed'
    before = source[start:end]
    after = result[result.index(patch.GROUP_START):result.index(patch.GROUP_END)]
    marker = '        for (int row = warp; row < rows; row += 8)\n'
    assert before.split(marker)[0] == after.split('        // Each warp owns')[0], 'GEMM/fragment exchange changed'
    assert after.count('__syncthreads();') == before.count('__syncthreads();') - 1
    assert after.count('atomicAdd(') == 4
    assert 'sh_c[i]' not in after
    assert 'reinterpret_cast<float4*>(dst)[0] = values;' in after
    # Exact Hadamard statement order is copied, not algebraically rewritten.
    old_helper = source[source.index(patch.HELPER_START):source.index(patch.HELPER_END)]
    new_helper = result[result.index('__device__ inline float4 direct_had_ff_128_values('):result.index(patch.GROUP_START)]
    normalize = lambda s: s[s.index('    int lane'):].replace(
        '    reinterpret_cast<float4*>(output_ptr)[lane] = v;', '    return v;')
    assert normalize(old_helper) == normalize(new_helper)
    # Every element of each full/tail row has exactly one warp/lane owner.
    for rows in range(1, 17):
        owned = [(row, lane * 4 + element)
                 for warp in range(8) for row in range(warp, rows, 8)
                 for lane in range(32) for element in range(4)]
        assert len(owned) == rows * 128 and len(set(owned)) == len(owned)
        assert set(owned) == {(r, c) for r in range(rows) for c in range(128)}
    for stride in (1024, 2048, 4096):
        for row in (0, 1, 15, 63):
            for n_base in range(0, stride, 128):
                assert (row * stride + n_base) % 4 == 0, 'unaligned float4 destination'
    assert json.loads((HERE / 'sources.json').read_text()) == build.declaration()
    coalesced = patch.transform(source, coalesced=True)
    declared = json.loads((HERE / 'sources-coalesced.json').read_text())
    assert hashlib.sha256(coalesced.encode()).hexdigest() == declared['changes']['exl3']['exl3-fat-kernel/exl3_fat_gemm.cu']['candidate_sha256']
    for chunk in range(4):
        addresses = []
        for lane in range(32):
            source_lane = lane // 4 + chunk * 8
            component = lane & 3
            address = chunk * 32 + lane
            assert source_lane * 4 + component == address
            addresses.append(address)
        assert addresses == list(range(chunk * 32, (chunk + 1) * 32))
    recipe = build.render()
    apply_at = recipe.index('RUN python3 /opt/glm53/direct-epilogue.py --apply')
    compile_at = recipe.index('python3 /opt/glm53/patch_exl3_fat_kernel.py /tmp/exllamav3')
    assert apply_at < compile_at
    assert recipe.count('--candidate-manifest /opt/sparkglm-candidate-sources.json') == 1
    assert '--kind native' in recipe and '--kind python' in recipe
    original_recipe = (ROOT / 'Dockerfile').read_text()
    anchor = 'COPY overlay/exl3_fat_gemm.cu /opt/glm53/exl3-fat-kernel/exl3_fat_gemm.cu\n'
    prefix = original_recipe[:original_recipe.index(anchor) + len(anchor)]
    assert recipe.startswith(prefix), 'candidate invalidates the unchanged vLLM build prefix'
    incremental = (HERE / 'Dockerfile.incremental').read_text()
    assert incremental.index('--kind python') < incremental.index('direct-epilogue.py --apply')
    assert incremental.index('--kind exl3') < incremental.index('direct-epilogue.py --apply')
    assert 'TORCH_CUDA_ARCH_LIST=12.1a MAX_JOBS=8' in incremental
    assert 'c5d9c657966ffeeaa9353f0cc899f18629da4a13.tar.gz' in incremental
    assert 'build-profile="candidate"' in recipe
    original_hash = hashlib.sha256((ROOT / 'overlay/exl3_fat_gemm.cu').read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory() as temporary:
        local = Path(temporary) / 'kernel.cu'
        local.write_text(source)
        subprocess.run([sys.executable, str(HERE / 'patch.py'), str(local), '--apply'], check=True, capture_output=True)
        assert local.read_text() == result
        protected = Path(temporary) / 'overlay'
        protected.mkdir()
        (protected / 'kernel.cu').write_text(source)
        refused = subprocess.run([sys.executable, str(HERE / 'patch.py'), str(protected / 'kernel.cu'), '--apply'], capture_output=True)
        assert refused.returncode != 0
        assert (protected / 'kernel.cu').read_text() == source
    assert hashlib.sha256((ROOT / 'overlay/exl3_fat_gemm.cu').read_bytes()).hexdigest() == original_hash
    environment = dict(os.environ, SPARKGLM_GPU_RESERVED='')
    refused = subprocess.run(['bash', str(HERE / 'run-g1.sh'), 'sparkglm:local',
                              'sparkglm-candidate:test', '/tmp/unused-direct-epilogue'],
                             env=environment, capture_output=True)
    assert refused.returncode == 2 and b'Reserve an idle GPU' in refused.stderr
    dry_run = subprocess.run([sys.executable, str(HERE / 'build.py'), '--print-dockerfile'],
                             check=True, capture_output=True, text=True)
    assert dry_run.stdout == recipe
    for path in HERE.glob('*.py'):
        compile(path.read_text(), str(path), 'exec')
    assert 'tokens = max(max(count_list), cap + 1)' in (HERE / 'bench.py').read_text()
    # Scratch offsets pack fat experts, unlike route offsets. Exercise the
    # actual oracle builder's loop without importing its CUDA dependencies.
    import ast
    tree = ast.parse((HERE / 'bench.py').read_text())
    run_case = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_case')
    loop = next(n for n in run_case.body if isinstance(n, ast.For) and
                isinstance(n.iter, ast.Call) and ast.unparse(n.iter) == 'enumerate(count_list)')
    fat_branch = loop.body[0]
    assert isinstance(fat_branch, ast.If)
    compact_loop = ast.For(target=loop.target, iter=loop.iter, body=[
        ast.If(test=fat_branch.test, body=fat_branch.body[-2:], orelse=[]), loop.body[-1]], orelse=[])
    module = ast.fix_missing_locations(ast.Module(body=[compact_loop], type_ignores=[]))
    scope = dict(count_list=[0, 1, 128, 129, 255], cap=128, begin=0, scratch_begin=0,
                 slices=[], transformed=None, gu=None, down_input=None)
    exec(compile(module, '<oracle-offset-test>', 'exec'), scope)
    assert [(s[0], s[1]) for s in scope['slices']] == [(0, 129), (129, 384)]
    assert scope['begin'] == 513
    print('direct epilogue G0: PASS (source transform and build wiring; no CUDA execution)')


if __name__ == '__main__':
    main()
