# Bit-exact paired and fused EXL3 fat pipeline

Date: 2026-09-03

## Verdict

Accept for the two-Spark GLM-5.3 appliance.

The candidate removes four device copies from every fat expert by reading the
immutable gate and up trellises directly, then replaces the separate clamp,
sigmoid, multiply, FP16 cast, SUH multiply, and Hadamard operations with one
SM121 kernel. The combined fat-expert inner pipeline is bit-exact with the
accepted composed path and improves every comparable 16K/32K endpoint row.

Both subfeatures remain independently reversible with `EXL3_FAT_PAIR` and
`EXL3_FAT_FUSED_ACT`.

## Artifact and configuration

- Frozen control image: `sparkglm-vllm:fp16-cpasync-0c03250`
- Candidate image: `sparkglm-vllm:exl3-fat-rounded-final`
- Measured head container image ID:
  `sha256:e89cd1711bab6c60b271cee294dbb68f78fe8a7011b10eafb3c018267cdb83aa`
- Measured worker container image ID:
  `sha256:061ea36268175a01313fa2dd4dbe2e87e334ac36bee62c0477822889cda9d3f6`
- Final head tag ID after adding the rollback-only test:
  `sha256:851107d99acc58914f135c428865d69f0fd2409a525c0d4114054d03f2021974`
- Final worker tag ID after adding the rollback-only test:
  `sha256:461f5cba31ee8bb6981a44b597a9d7e0b324cb530f19a7f2c1270d7deb1ea7cf`
- Candidate extension SHA-256 on both ranks:
  `a3d0d7bb9e8e12d3a6f1f2032d0d61a57c4b12b9d065d17ff0ffcf4a05e873ca`
- Candidate EXL3 Python SHA-256 on both ranks:
  `edf9a35071c6c20980b4aa41af6275b3b985400abbab4db8592f9b7395ff3083`
- TP2 over ConnectX-7, EXL3 4 bpw weights, FP8 KV, 1M context
- Work-conserving 7,168-token mixed scheduler, four sequences
- DFlash2 K7 with draft TP2
- Candidate flags: pair=1, fused activation=1

The images have different assembly-layer identities because they were built
locally on each Spark. The executable extension and Python implementation are
byte-identical, which is the relevant rank contract.

The final tags add only a static rollback assertion to the embedded test. Their
CPU self-check passed on both ranks while the measured containers remained
resident; the byte-identical executable artifact had already passed the full
GPU suite on both unloaded ranks.

## Operator proof

The GB10 extension suite passed independently on both ranks. In particular:

- the paired gate/up output is bit-exact with the accepted packed launch;
- fused SwiGLU plus SUH/Hadamard is bit-exact with the composed PyTorch plus
  ExLlamaV3 path;
- the diagnostic schema fails closed when either enabled symbol is absent;
- direct, paired, fused-activation, and scatter counters advance together;
- decode-sized inputs do not enter the fat path;
- CUDA graph capture, mixed thin/fat composition, expert-map handling, and
  existing E2 fallback gates pass.

The combined benchmark includes gate/up packing, gate/up GEMM, SwiGLU,
down-input Hadamard, down GEMM, and scatter:

| Expert rows | Accepted (ms) | Candidate (ms) | Speedup | Max absolute difference |
| ---: | ---: | ---: | ---: | ---: |
| 129 | 0.2132 | 0.1686 | 1.264x | 0 |
| 512 | 0.4449 | 0.3371 | 1.320x | 0 |
| 2,048 | 1.5501 | 1.2690 | 1.222x | 0 |
| 7,168 | 6.0827 | 4.3525 | 1.398x | 0 |

## Same-prompt endpoint A/B

The candidate used the frozen control's exact prompt salt
`fat-fused-paired-1`, 128 forced output tokens, and a five-second C2 stagger.
The control's medium C1 row was a first-shape JIT outlier and is deliberately
excluded from percentage claims. The service received a separate all-shape
warmup matrix before the measured candidate run.

| Case | Candidate/control aggregate prefill (tok/s) | Prefill delta | Candidate/control TTFT (s) | TTFT delta | Candidate/control wall (s) | Wall delta | Candidate/control aggregate decode (tok/s) | Decode delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Medium C2 | 1313.9 / 1198.1 | +9.67% | 21.19, 19.06 / 23.17, 21.38 | +8.56%, +10.88% | 29.89 / 32.46 | +7.92% | 29.18 / 27.36 | +6.62% |
| Large C1 | 1273.1 / 1233.1 | +3.24% | 25.28 / 26.10 | +3.14% | 29.75 / 30.82 | +3.45% | 28.37 / 26.92 | +5.38% |
| Large C2 | 1323.1 / 1197.8 | +10.46% | 31.74, 43.64 / 35.50, 48.72 | +10.61%, +10.44% | 54.00 / 60.04 | +10.05% | 11.41 / 10.36 | +10.15% |

All six candidate requests completed 128/128 tokens, retained their own marker,
contained no foreign marker, and reported no error. Endpoint output hashes are
not a cross-run equality oracle here because the checkpoint's generation
configuration remains probabilistic even when the request sends temperature
zero. Exact correctness is established at the changed operator boundary.

## Decode and quality gates

The change is prefill-only: the fat path is not selected at decode row counts.
The live endpoint nevertheless passed both five-run 400-token gates:

| Gate | Median tok/s | Range | Median DFlash acceptance | Accepted tokens/step |
| --- | ---: | ---: | ---: | ---: |
| Structured count | 65.01 | 63.73-65.21 | 0.9776 | 6.843 |
| Hash-map prose | 28.97 | 27.76-30.90 | 0.3616 | 2.531 |

Every run produced 400/400 tokens, finished by length, contained no NaN, and
left the endpoint healthy. The additional Paris, decimal-comparison, and color
coherence probes all passed.

## Live engagement and health

After the complete endpoint suite, both ranks independently reported the same
diagnostic state: 2,058 prefill-layer calls, 1,941 kernel calls, 116 thin calls,
and 124,618 direct, pair, fused-activation, and scatter calls. There were no
batched, sorted, or legacy fallbacks, and zero matching ERROR, traceback, Xid,
OOM, CUDA-error, or NCCL-abort lines on either rank.

Startup exposed 1,325,783 KV tokens and 20.68 GiB of KV cache, far above the
target matrix. The new scratch allocation was 432,017,408 bytes per rank and
remained persistent rather than growing with requests.
