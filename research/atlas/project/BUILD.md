# Portable Atlas build and measured-profile boundary

These are experimental source and deployment recipes. They do not replace SparkGLM's recommended vLLM default. No clean build of this Dockerfile or GPU run of these newly portable commands is claimed.

## Keep the two source identities separate

`profile.json` represents the best measured r23 S8 configuration, source `9b3e316a7f5ca92ec22a68414e7aefeda32023dd`, retained binary SHA256 `5967ca4e1a40fb437eab1226e79de74501c322a6e026de0740b970c90accd16a`. Its original private launch manifest is hash-bound in the profile. All compute, memory, communication and server options match both ranks; only site locators and the Nsight profiling wrapper/mounts are made portable. Explicit artifact mounts replace libraries formerly embedded in site images.

Use `reconstruct.py --checkpoint measured` for the exact r23 source tree `72aec3455db6ab35a75985b3f390cb659d274244`. The default `--checkpoint latest` instead exports later source `faf4e874b2d418b67a5bc4ec5793c450c9f27ebb`, including the bounded route diagnostic. The two paths have separate manifests and final patches. A newly built binary from either export remains a new artifact requiring native and model qualification; source-tree identity does not prove binary identity or reproduce the external libraries. Diagnostic flags are absent from this profile. Do not label the latest source or an arbitrary image as the measured r23 binary.

## Draft native build

Use a verified reconstructed source tree as Docker context, and record its reconstruction receipt/tree, toolchain versions, image digest and binary hash. The Dockerfile requires an explicit source revision label; the operator must ensure it matches that tree. For measured r23 source, using a clean local upstream checkout and a fresh absent destination:

```sh
python3 "$SPARKGLM_SOURCE/research/atlas/project/reconstruct.py" \
  --checkpoint measured --source-repo "$PWD/atlas-upstream" \
  --destination "$PWD/atlas-source-r23"
docker build \
  -f "$SPARKGLM_SOURCE/research/atlas/project/Dockerfile" \
  --build-arg ATLAS_SOURCE_REVISION=9b3e316a7f5ca92ec22a68414e7aefeda32023dd \
  -t atlas-project:r23-source "$PWD/atlas-source-r23"
```

For latest diagnostic source, explicitly select `--checkpoint latest`, use another fresh destination, and label the build with revision `faf4e874b2d418b67a5bc4ec5793c450c9f27ebb`.

These commands are documentation, not a recorded successful native build. Run natively on the intended ARM64 Spark build environment only when authorized. The inherited recipe pins CUDA base digests, Rust1.93.1 and CUTLASS; apt/rustup acquisition still requires network and current package availability. It builds the Rust server, not the external libraries below.

The builder creates `/opt/cuda-stubs/libcuda.so.1` pointing to CUDA's unversioned driver stub and exposes it only in the builder's loader path. This addresses CPU test processes that dynamically load the versioned driver soname; it does not provide GPU execution or validate driver behavior. The runtime stage receives neither the stub nor its loader environment. Select only tests documented as CPU-safe for a no-GPU test run; native CUDA filters still require hardware and the real driver.

## Required operator artifacts, never committed

Supply all three exact shared libraries, independently verify provenance, and retain their applicable notices:

| File | SHA256 | Runtime location |
|---|---|---|
| `libatlas_mango_flash.so` | `c0d2a9c7c0d922aaad4097bd3bf26fe3fb3b4b2da8ecf50531a62fa036597bda` | `/opt/atlas-flash/libatlas_mango_flash.so` |
| `libatlas_glm53_flash_kda.so` | `7fe3fa22f7fcf2159848bf47189f7caa1adb7286d99e4a334376a410047dc092` | `/opt/atlas-flash/libatlas_glm53_flash_kda.so` |
| `libatlas_glm_sparse_native.so` | `583f41ebc63a1c4c1b3b757bcff8136b6a46b5a3d006dc1ce1db45ff84038925` | `/opt/atlas-sparse/libatlas_glm_sparse_native.so` |

The FlashKDA shim dynamically links the second library via `$ORIGIN`; put both in the supplied directory. The [FlashKDA dependency recipe](../flash_kda/rebuild.sh) and current reconstructed shim source are distinct stages. A rebuilt library may have different bytes and must be qualified and recorded as a new artifact; these hashes are not a claim of reproducible compilation.

The native sparse library depends on retained NVIDIA/FlashInfer object `csrc_sparse_mla_sm120_prefill.cuda.o`, SHA256 `9e372b5a47ade0a8332a73878ec3f6b917ae137447fea2335d2003dd146d549a`. Its exact historical source identity has not been proven. The cached FlashInfer0.6.15/121a object is intentionally absent from this public repository. The reconstructed `scripts/dev/glm_sparse_native/` wrapper sources and newer reference source alone do not prove they can reproduce that object. A clean native-sparse library rebuild and full GPU qualification remain blocked on establishing that exact dependency or qualifying a replacement. Do not quietly unset `ATLAS_GLM_SPARSE_NATIVE`, FlashKDA, or other retained flags to disguise missing dependencies as the measured configuration.

Also supply the NVIDIA predictor overlay, original checkpoint paths used by that overlay, CUDA cache directory, and the retained template. Template SHA256 is `2e24c13c22cbd63370908e2302a045bceeac801ff83d28ec683d1983175a3886`. Preserve overlay symlink targets: the original checkpoint bind keeps the operator's same absolute path. Weights, converted tensors, libraries and private deployment values stay outside Git.

## Render and verify on each owning host

The renderer only prints argv; it never starts a container or connects to a host. All site parameters and both external-library locations are mandatory. Use `--verify-local-artifacts` on the machine owning those paths to hash the template and all three libraries before accepting the output. Without that option it is only a path-based draft for review; it does not verify remote files, image provenance, model completeness, idle hardware or available memory.

Example with documentation-only locators and operator-owned paths:

```sh
python3 "$SPARKGLM_SOURCE/research/atlas/project/render.py" \
  --rank 0 --master 192.0.2.1 --bind 127.0.0.1 --port 8893 \
  --nic fabric0 --hca roce0 --name atlas-r23-r0 --image atlas-r23:verified \
  --overlay /srv/models/overlay --original /srv/models/original \
  --cuda-cache /srv/cuda-cache --template /srv/chat_template.jinja \
  --flash-library-dir /srv/atlas-flash \
  --native-library /srv/atlas-sparse/libatlas_glm_sparse_native.so \
  --verify-local-artifacts --format shell
```

Use the other host's explicit paths, rank1, distinct container name and port for the peer. The master address and transport interfaces must match the operator's fabric. Inspect the output before running it; do not pipe the renderer directly into a shell. Neither generic image labels nor successful rendering establishes the retained binary's identity.

## Local profile checks

```sh
python3 "$SPARKGLM_SOURCE/research/atlas/project/test_profile.py"
```

Maintainers with the exact private receipt can additionally pass `--verified-manifest /absolute/path/to/the/reviewed-manifest.json`. The comparison verifies its hash and compares complete Docker/runtime argv for both ranks, excluding only Nsight mounts/wrapper and the two explicit operator artifact mounts. It retains exact guards, all environment values and their order, command arguments, original-checkpoint bind semantics and model/template locations after substituting site parameters. These CPU checks do not replace native tests or matched C4 evidence.
