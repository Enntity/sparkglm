# Day-two Atlas standalone sources

These are useful source-only reproductions and helper tests from the Atlas
campaign. Their [recorded outcomes](../../../../results/candidates/2026-09-11-atlas-day2/)
remain separate from newly compiled artifacts. This directory does not enable
an engine feature or change SparkGLM's vLLM default. Rejected and partial-pass
routes stay rejected; passing a synthetic operator is not appliance qualification.

Original Atlas-derived source is AGPL-3.0-only. Production snapshots retain their
source headers. The NVIDIA enum header and `sparse-native/NATIVE-BRIDGE-NOTICE.txt`
retain BSD-3-Clause notices; FlashKDA is an external MIT dependency. See
[licensing](../../../../docs/LICENSING.md) and the per-file `source-map.json`.
Raw provider responses, captured model routes, compiled artifacts, machine
controllers and bundle-only result analyzers remain in private research. Public
runner changes remove omitted-file dependencies only; CUDA arithmetic, fixtures,
allocation guards and fixed performance gates are unchanged.

## Inventory and entry points

Run commands from the indicated subdirectory. `bash build.sh` is CPU compilation;
GPU execution is a separate deliberate operation. Use an idle GB10 CUDA13 runtime,
not the compiler image's stub driver. Compile with the recorded explicit
`compute_121a`/`sm_121a` target and precise arithmetic flags in each script.

| Directory | CPU preparation | Manual GPU entry | Scope/disposition |
|---|---|---|---|
| `head-batchm` | `bash build.sh` | `./bench --check`, then `./bench --run` | M3 head batchm operator passed; preserves scalar-GEMV arithmetic, differs slightly from old GEMM |
| `mla-o-batching` | `bash build.sh` | `./bench --check`, then `./bench --run` | Within-owner M3 accepted; incremental M12 failed its gate |
| `split-decode` | `bash build.sh` | `bash run.sh` | Fixed S8 split+merge passed; S16 exploratory, not silently selected |
| `moe-tp-shape` | `bash build.sh` | `bash run.sh` | TP-shaped routed pipeline rejected |
| `moe-down-compact` | `bash build.sh` | `bash run.sh` | Compact down+builder routed pipeline rejected |
| `moe-joint-m12` | `bash build.sh` | `bash run.sh` | Balanced case failed, skew passed; overall no integration. Both fixed cases run; no best-case promotion |
| `owner-cohort-copy` | `bash build.sh` | `./copy-test --gpu` | Generic bounded copy helper qualification |
| `ordered-router-workers` | `bash build-all.sh` | `./bench-existing`, `./bench-ds`, `./bench-glm` separately | Three fixed same-order router screens; retain each result and declared gate |
| `flash-kda` | External pinned FlashKDA build plus probe/shim commands below | `python3 probe.py`, then `python3 shim_test.py` | Preparation-inclusive recurrence/adapter gates; original diagnoses retained |
| `sparse-native` | External object/reference dependencies below | Explicit selected checker | Native preparation/ABI/split-merge gates; exact upstream object source remains unresolved |
| `moe-route-replay` | Host adapter self-test only | **No public run entry** | Actual-route benchmark source retained; requires separately regenerated private `capture_routes.hpp` and original provenance gates |
| `route-analysis-helper` | `python3 -m unittest discover -v` | None | Pure synthetic count/reuse analysis, not a latency predictor |
| `profile-event-helper` | Tests below | None | Pure interval-union attribution helper and independent oracle |

Head allocation cap is1,400MiB, MLA O256MiB, split decode128MiB and router32MiB;
these benchmarks retain4GiB free-reserve checks. MoE screens require at least
12GiB initially (8GiB allocation cap plus4GiB reserve) and recheck after case
allocations. The copy fixture uses about128MiB under a256MiB cap plus4GiB reserve.
The FlashKDA/reference Python probes are separate from these caps: inspect their
explicit shapes and require an otherwise idle GPU with ample free memory.
Do not lower guards, run benchmarks alongside serving, or interpret a memory
refusal as a performance failure. Exit3 normally means the predeclared performance
gate missed; exit2/error means a correctness/guard failure. Preserve raw output.

## Source and CPU checks

From this directory:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 verify-sources.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s route-analysis-helper -v
PROFILE_EVENT_CANDIDATE="$PWD/profile-event-helper/ds-reviewed.py" \
  PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s profile-event-helper -v
PYTHONDONTWRITEBYTECODE=1 python3 moe-joint-m12/fixture-plan.py
```

`verify-sources.py` checks every public source hash, retained checksum manifests,
Python/shell syntax and quoted include closure. It does not import GPU modules,
compile CUDA, or run a benchmark. The explicit include exceptions are the
reconstructed engine baseline and private replay header described here.
Generic C++ host checks can be compiled with C++17 and warnings enabled:
`head-batchm/host-check.cpp`, `mla-o-batching/host-check.cpp`,
`split-decode/host-check.cpp`, and `moe-route-replay/host-check.cpp`.
The replay host check uses synthetic data and never emits a replacement capture.

Original worker drafts named `original` or `ds-fast-original` are historical
source evidence. Build scripts select only the reviewed headers; do not switch
back to an original that failed review. Helper tests default to a reviewed
candidate only when the explicit environment variable above is supplied.

## FlashKDA dependency preparation

Reconstruct the pinned engine with [the project entry](../../project/README.md).
The external FlashKDA library is rebuilt using
[the retained recipe](../../flash_kda/rebuild.sh), preserving its pinned source,
slot patch and separate original toolchain. Put the resulting
`libatlas_glm53_flash_kda.so` alongside this probe's source. Build the baseline
probe with the reconstructed engine include path:

```sh
nvcc -std=c++17 -O3 --fmad=false -shared -Xcompiler=-fPIC \
  -gencode=arch=compute_121a,code=sm_121a \
  -I"$ATLAS_SOURCE/kernels/gb10/common" bridge_probe.cu -o libprobe.so
```

`build-shim.sh` expects this directory mounted at `/work` and builds the shim
and smoke executable. It no longer runs a GPU Python test as part of compilation;
run the explicit probe/shim checks later on an idle GPU. `probe.py` needs Torch.
Changing the dependency compiler is not proof of reproducing the recorded bytes.

## Native sparse dependency boundary

`sparse-native` host includes are complete, including the unchanged NVIDIA enum.
`build-wrapper.sh` expects this directory at `/eval` and the reconstructed engine
at `/source`. `build-unified.sh` also requires the external object at
`/native/csrc_sparse_mla_sm120_prefill.cuda.o`, SHA256
`9e372b5a47ade0a8332a73878ec3f6b917ae137447fea2335d2003dd146d549a`.
Exact source identity of that retained object is unresolved; no binary is included
and these wrappers do not claim to rebuild it. Native CPU linking uses CUDA/C++
only; Torch/TVM are separate reference-check dependencies.

Build `native-prep.cu` according to its compile-command comment. The separate
reference merge library can be compiled from `merge-ds.cu`:

```sh
nvcc -std=c++17 -O3 --fmad=false -shared -Xcompiler=-fPIC \
  -gencode=arch=compute_121a,code=sm_121a merge-ds.cu -o libatlas_sparse_merge.so
```

The Python checks expect `/eval/libatlas_sparse.so`,
`/eval/libatlas_sparse_merge.so`, `/eval/libatlas_native_prep.so`,
`/eval/libatlas_glm_sparse_native.so` as appropriate, and the external
`/native/sparse_mla_sm120.so` TVM reference module. Use each checker's documented
arguments/default paths. No check should be described as runnable without its
explicit reference artifacts; prep and unified checks retain1GiB/2GiB Torch
allocation caps. The unresolved native source dependency also limits the exact
latest engine build, as documented in the project entry.
