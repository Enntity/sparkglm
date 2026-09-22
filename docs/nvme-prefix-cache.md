# Experimental NVMe prefix cache

This opt-in connector saves completed hybrid prefix chunks on each TP rank's
local SSD. An idle conversation whose GPU pages were evicted can reload those
chunks instead of repeating the entire prefill. Hot GPU hits still use the
existing path. This is experimental; no serving default or performance claim
is changed by this implementation.

Build `Dockerfile.nvme-prefix-cache` with `--build-arg BASE=<qualified-image>`.
The patch installer accepts only the pinned vLLM source hashes and verifies all
files before changing any. This derivative does not replace your entrypoint;
use the updated SparkGLM entrypoint or its identical NVMe option block.

Enable with `SPARKGLM_NVME_PREFIX_CACHE=1`, `SPARKGLM_NVME_ROOT` pointing to a
private persistent local SSD mount, and `SPARKGLM_NVME_NAMESPACE` identifying
the exact target/draft checkpoints, serving build, TP topology, and KV layout.
Both ranks must use the same namespace. Change it whenever any of those inputs
changes. Do not reuse a namespace for unrelated builds. Set
`PYTORCH_CUDA_ALLOC_CONF` without `expandable_segments:True`; the launcher sets
`PYTHONHASHSEED=0` for stable cross-process prefix hashing.

`SPARKGLM_NVME_CAPACITY_BYTES` caps newly reserved payload bytes per rank and
namespace (default 64 GiB). `SPARKGLM_NVME_RESERVE_BYTES` preserves free disk
space (default 64 GiB). Failed or repeated stores retain conservative quota
charges until restart. `SPARKGLM_NVME_IO_THREADS` defaults to two, each with a
chunk-sized pinned bounce buffer. There is no resident CPU cache tier.

A rank-zero commit record is published only after all TP workers acknowledge
successful writes. Partial or failed writes cannot become hits. The loader
checks exact file length and SHA-256 before copying each chunk. If any rank
fails a load, all transfers finish before the hybrid allocation is released;
the request skips cache lookup and recomputes its prompt. Corrupt commit
records are invalidated. No distributed filesystem is required.

Cache bytes are prompt-derived private data: directories use mode 0700 and
files mode 0600. Keep them outside source/result bundles. Do not run an online
TTL sweeper or manually delete files during serving. This version stops storing
when its quota/reserve is reached; it does not reclaim old conversations.
Offline cleanup requires all ranks stopped. Persistent reuse requires retaining
both ranks' cache files and the head's commit index.

Qualification must include local fault checks, a CUDA byte roundtrip, tinyGLM
hybrid restore and a failed-rank-file cold retry, then the actual model's cold,
hot and GPU-evicted SSD-restored continuation. A filesystem page-cache hit must
not be presented as a physical SSD benchmark. Restart reuse and multi-session
pressure require explicit measurements before claiming them.
