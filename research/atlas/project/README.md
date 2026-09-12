# Active Atlas research project

This is the source reconstruction entry for the maintained Atlas experiment.
It is separate from SparkGLM's recommended vLLM serving path and starts no
service. Two explicit source checkpoints are available:

| Selection | Revision | Complete tree |
|---|---|---|
| `--checkpoint latest` (default; diagnostic source) | `faf4e874b2d418b67a5bc4ec5793c450c9f27ebb` | `38b7f5cb239e186958a474e59c6d275fe8657813` |
| `--checkpoint measured` (best measured r23 profile source) | `9b3e316a7f5ca92ec22a68414e7aefeda32023dd` | `72aec3455db6ab35a75985b3f390cb659d274244` |
The latest checkpoint retains the engine's implementation, regression tests and reviewed standalone
sources through the bounded route diagnostic. The measured checkpoint stops at r23,
before that later route-capture instrumentation. Source availability does not
establish full-model quality, C4 parity or release qualification.

Atlas-derived source and project glue are **AGPL-3.0-only**. Third-party
FlashKDA MIT and NVIDIA BSD notices remain applicable; the root Apache license
does not replace them. See [licensing](../../../docs/LICENSING.md).

## Reconstruct locally

Requires Python3.9+ and Git. First acquire the public upstream source explicitly;
this command is a separate network operation:

```sh
GIT_LFS_SKIP_SMUDGE=1 git clone \
  https://github.com/Mango-kid/atlas.git atlas-upstream
```

A clean existing local checkout containing the pinned base also works. Set
`SPARKGLM_SOURCE` to this public repository's absolute path. Choose a new,
absent output path whose parent already exists:

```sh
python3 "$SPARKGLM_SOURCE/research/atlas/project/reconstruct.py" \
  --checkpoint latest --source-repo "$PWD/atlas-upstream" \
  --destination "$PWD/atlas-source"
```

For the exact source paired with `profile.json`, use `--checkpoint measured`
and a fresh destination such as `atlas-source-r23`. Omitting `--checkpoint`
selects latest; never infer a measured build from that default.

`reconstruct.py` itself makes no network calls. It validates the
selected [latest manifest](../nvidia-day2-source.json) or
[measured manifest](../nvidia-r23-source.json) and all four patch hashes, requires a clean
local source repository, clones locally without shared object hardlinks, checks
out the exact Mango base, requires a clean destination before applying patches,
and verifies the complete Git tree plus all95 latest or81 measured incremental changed-file hashes. Git LFS
smudging and recursive submodule operations are disabled. No tensors or compiled
artifacts are supplied by this package.

The source repository is unchanged. The destination has the reconstructed
changes staged for inspection but no invented upstream commit. A successful
receipt is written inside `.git/sparkglm-atlas-source.json`; HEAD remains the
upstream base, so **use the verified tree and receipt, not HEAD alone, to identify
this reconstructed source**. Failures retain the destination for inspection;
the utility never removes an existing directory or overwrites unrelated work.
For another attempt, choose another fresh path or deliberately clean up the
failed attempt yourself. The generated source directory can be an ignored build
cache rather than an additional worktree.

The ordered patch chain is:

1. Mango base `90b3584abc71b44b609637092b85d8423d8ff20f`.
2. `nvidia-modelopt-refresh.patch`: initial NVIDIA compatibility.
3. `nvidia-prefill-mtp2.patch`: selected r11 prefill/MTP2 source.
4. `nvidia-prefill-rejected-standalones.patch`: retained rejected r11 experiments.
5. Latest selects `nvidia-day2.patch`: complete95-file delta from
   `dd0ffd157fb7c6e96d2f3858c888513c6f142cf6` to `faf4e874`.
   Measured selects `nvidia-r23.patch`: complete81-file delta from the same
   base to `9b3e316a`. The first three patches are shared, byte-for-byte;
   the latest manifest and patch remain unchanged.

Do not apply the older `atlas-glm53.patch` to this route; it has a different
upstream base. Rejected standalone files in the reconstructed tree remain
rejected. Diagnostic flags are not an ordinary serving profile.

## Build boundary and unresolved dependency

The source reconstruction is self-contained relative to the pinned upstream
checkout. The exact measured r23 **native-sparse configuration is not yet
fully rebuildable from this source package alone**:

- FlashKDA has a pinned source recipe at [flash_kda/rebuild.sh](../flash_kda/rebuild.sh),
  upstream `1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b` and CUTLASS
  `5c149f52a436782210263fb2f19b354443a61c6a`. The new engine shim lives in
  `research/flash-kda-prefill/`. Preserve the older dependency toolchain/flags
  and the current shim's separate build contract; qualify rebuilt artifacts.
- Native sparse prefill currently requires external object
  `csrc_sparse_mla_sm120_prefill.cuda.o`, SHA256
  `9e372b5a47ade0a8332a73878ec3f6b917ae137447fea2335d2003dd146d549a`.
  Exact source identity of that retained FlashInfer0.6.15/121a cache object has
  not been established. It is intentionally not distributed. The reconstructed
  `scripts/dev/glm_sparse_native/README.md` explains the ABI, BSD notice,
  wrapper build and reference-test dependencies. A newer source snapshot is
  not asserted to rebuild this object.

Do not silently disable a missing dependency and label the result the measured
configuration. Engine source can be inspected and tested separately; a native
build or alternate rebuilt operator needs its own qualification. The previous
[r11 build/profile](../nvidia-prefill/README.md) is historical and does not
contain the new native library stages or current runtime settings.

The [NVIDIA MTP converter](../nvidia-mtp-converter/README.md) still prepares the
external predictor overlay from pinned NVIDIA checkpoint
`423acf37583782c51c142d145aef733d72943d93`. Keep weights, converted tensors,
compiled libraries and local deployment settings outside Git.

## CPU checks

```sh
PYTHONDONTWRITEBYTECODE=1 python3 \
  "$SPARKGLM_SOURCE/research/atlas/project/test_reconstruct.py" -v
```

The tests exercise exact synthetic reconstruction, source immutability,
existing-destination protection, dirty source refusal, patch tampering,
path/pin validation and final tree/file mismatch failures. They also validate
both manifests, checkpoint selection and the shared plus checkpoint-specific
public patch hashes. To verify both real complete trees without a new worktree,
pass `--source-repo /absolute/path/to/a/local/Atlas/checkout` containing both
pinned revisions. This uses a fresh temporary bare object directory and index
with read-only access to the source object store; it verifies each full tree
and every listed file hash while preserving source HEAD and status. The exported engine retains its focused
Rust tests; these Python checks do not replace native operator/integration tests
or the [qualification methodology](../../../docs/METHODOLOGY.md).
