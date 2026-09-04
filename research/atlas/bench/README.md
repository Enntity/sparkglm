# GLM-5.3 test gates

## Staggered serving probe

Run identical bytes against each recipe. Start with concurrency 1, then 2 and
4. The useful comparison is per-request TTFT, decode tokens/s, and total wall
time—not GPU utilization alone.

```bash
python3 bench/glm53/staggered_openai.py \
  --base-url http://127.0.0.1:8888 \
  --model GLM-5.3-Flash-EXL3 \
  --concurrency 4 \
  --stagger-ms 500 \
  --prompt-tokens 256 \
  --output-tokens 64 \
  --min-output-tokens 64 > glm53-c4.jsonl
```

Compare active-decode prefill slabs such as 128, 64, and 32 tokens under the
same `4 decode : 1 prefill` cadence. Do one warm-up and three measured repeats.
The script reports visible stream events, p95/max inter-token gaps, output
SHA-256, TTFT, and decode rate. Reject any output mismatch, collective-order
error, request failure, cancellation corruption, or slot-reuse corruption.

For Mia's published decode condition, do not use the 64-token smoke test.
Generate 400 tokens, warm once, and report the median of five:

```bash
python3 bench/glm53/staggered_openai.py \
  --base-url http://127.0.0.1:8888 \
  --model GLM-5.3-Flash-EXL3 \
  --concurrency 1 --stagger-ms 0 \
  --prompt-style mia-structured \
  --prompt-tokens 64 --output-tokens 400 --min-output-tokens 400 \
  --disable-loop-watchdog
```

The short test overstates C1 steady-state decode because all tokens accepted in
the first speculative block arrive at the TTFT boundary, while the rate
numerator still includes all but one completion token. It remains useful for
C2/C4 admission, graph-capture, and state-isolation checks.

The synthetic fixture only approximates a tokenizer length: the 256-token
setting produced 498 tokens with the tested GLM tokenizer. Preserve the exact
request bytes when comparing recipes.

## Tiny KDA probes

The synthetic probes include the production CUDA source and compare it with
independent CPU loops. They do not load model weights.

```bash
nvcc -std=c++17 -arch=sm_121f --fmad=false -O3 \
  bench/glm53/kda_sm121_probe.cu -o /tmp/kda_sm121_probe
/tmp/kda_sm121_probe
```

`kda_decode_layer_sm121_probe.cu` exercises fragmented recurrent/conv slots;
`kda_prefill_glue_sm121_probe.cu` covers conv4+SiLU, beta transpose, and gated
RMSNorm. `flash_kda_slots_probe.py` compares the pinned AOT bridge with upstream
FlashKDA using physical slots `[3,1]` and proves inactive slots are unchanged.

## Sparse DSA/index probe

Compile `dsa_sm121_probe.cu` beside the production `glm53_dsa_index.cu`. It
checks chunk-equivalent learned BF16 pooling, left padding, top-512 pool
expansion plus the incomplete tail, and sparse MLA output.

## Real-checkpoint micro-probes

These load one tensor or expert, not the 106B model.

The EXL3 probe loads one routed expert—gate/up/down, about 12.6 MiB—and compares
the raw-pointer fused kernel with reconstructed weights:

```bash
nvcc -std=c++17 -arch=sm_121f --fmad=false -shared -Xcompiler -fPIC \
  bench/glm53/exl3_raw_launcher.cu -o /tmp/libatlas_glm53_exl3_probe.so
python3 bench/glm53/exl3_checkpoint_probe.py \
  --model /path/to/GLM-5.3-Flash-EXL3-TR3-4bpw \
  --library /tmp/libatlas_glm53_exl3_probe.so --rows 8
```

`dsa_projection_checkpoint_probe.py` loads one 32 MiB BF16 `kv_b_proj` and
checks both absorbed-query and value-expansion orientations.

`router_checkpoint_probe.py` loads the 2.25 MiB BF16 gate plus FP32 correction
bias. It checks FP32 gate logits, exact top-8 IDs, and normalized/scaled route
weights. It also reports how many rows would select different experts if the
FP32 logits were rounded to BF16 before top-k.

Passing these gates proves individual ABIs and numeric paths. It does not prove
full-model logits, EP collective order, scheduler behavior, or throughput.
