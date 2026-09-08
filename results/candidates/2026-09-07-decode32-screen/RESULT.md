# Cooperative decode through 32 tokens: bounded screen

The preserved implementation accepts a scratch limit up to 32, while the
serving default enables cooperative decode only through 16. The extended
probe checks 17, 31, 32, and fallback 33 in addition to the existing shapes:

```sh
python3 benchmarks/exl3_decode_moe_ab.py --max-tokens 32 --experts 288 --graph
```

The GB10 H4096/I1024/E288 screen passed its declared numerical bounds and
changed-input CUDA graph replay. At M32 the measured stock/cooperative ratio
was 1.037; M17 was 0.963 and M31 was 1.025. These small, mixed effects do not
justify a default change or a full-model claim. The limit remains 16.

This historical operator harness uses synthetic weights and clamp 75, rather
than a real-model quality workload. It compares cooperative dispatch against
stock thin dispatch; production already uses cooperative dispatch through 16,
so the lower-token rows do not measure a new optimization. No sanitizer,
TP2 integration, or full speculative C4 qualification is claimed here.
