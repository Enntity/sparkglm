# RigMark cell set

The RigMark numbers in the result bundle use the 8 × 128K profile
(`PROFILE=8x128k` in `.env`, or `install/profiles/8x128k.json`) and the RiNGSiDE TP2 cell set, thinking on with
`reasoning_effort: low`, two runs per cell.

RigMark is by Alex Ellis / OpenFaaS Ltd ([alexellis/rigmark](https://github.com/alexellis/rigmark),
MIT). We run othexmr's fork, which adds the staggered-arrival suite, and
compare with the TP2 figures that
[RiNGSiDE](https://github.com/othexmr/GLM-5.3-Flash-NVFP4-2x-4x-DGX-Sparks-RiNGSiDE)
publishes.

```sh
git clone https://github.com/othexmr/rigmark.git && cd rigmark
git checkout 40fabcaf6e96                       # the published TP2 comparison commit
cp metadata.example.json metadata.json          # describe your hardware
./rigmark run --base-url http://127.0.0.1:8893 --model auto --metadata metadata.json \
  --comparison-id atlas-vs-ringside-tp2 --runs 2 --prefill-runs 2 --concurrency 1,2,3,4,5,6,8 \
  --concurrency-runs 2 --extra-body '{"chat_template_kwargs":{"reasoning_effort":"low"}}' \
  --label atlas-code-full --prefill-depths 8192,32768,65536 --output rig-code-full.json
./rigmark run --base-url http://127.0.0.1:8893 --model auto --metadata metadata.json \
  --comparison-id atlas-vs-ringside-tp2 --runs 2 --prefill-runs 2 --concurrency 1,2,3,4,5,6,8 \
  --concurrency-runs 2 --extra-body '{"chat_template_kwargs":{"reasoning_effort":"low"}}' \
  --label atlas-prose-concurrency --concurrency-workload prose --skip-prefill \
  --staggered 2,4,6,8 --staggered-runs 2 --staggered-depth 32768 --staggered-incumbent-tokens 1024 \
  --staggered-arrival-tokens 256 --staggered-delay 1.0 --staggered-workload prose \
  --output rig-prose.json
```

The whole set takes about an hour.

## Metadata: KV pool and context ceiling

RigMark records whatever `metadata.json` declares in the receipt (`run.appliance`), but its card shows neither the KV
pool nor the per-request ceiling. Declare them, using the field names from RigMark's own
[serving records](https://github.com/alexellis/rigmark/tree/master/examples/serving-records), and read the values from
rank 0's startup log (`docker logs atlas-sparkglm-rank0`, or the `lloom-atlas-sparkglm-*` container under LLooM):

| Field | Where it comes from | Production 2026-10-07 (4 × 512K, 0.91, carveout on) |
|---|---|---|
| `context_limit` | the profile's `--max-seq-len` | `524288` |
| `max_sequences` | `--max-num-seqs` | `4` |
| `gpu_memory_utilisation` | `GPU_MEMORY_UTILIZATION` | `0.91` |
| `logical_kv_tokens` | `KV cache: display carveout ... → N blocks` (or, carveout off, `→ N blocks × 16 tok/block`), times 16 | `1354400` (84,650 blocks) |
| `kv_cache_memory_per_rank` | `KV cache: N blocks × 11 layers` (8,448 B per block per layer) plus `Sparse index cache: ... = X MiB` | `9052818944` |
| `kv_cache_dtype` | `--kv-cache-dtype` | `fp8_g128` |

The four sequences share one pool: a single request can grow to the full 512K while the others leave room. The block
count moves by about 1% between starts (it is sized from free memory), so take it from the run you publish.
