# MXFP8 DFlash2 experiment

This directory makes the EXL3 branch's optional MXFP8 draft build self-contained.
The two serving files are byte-identical to the reviewed copy on the NVFP4
branch, pinned to loud1990 commit d00b2ffa70e0ccbfd582088ab32448b01bfe9c67.
Their original Apache-2.0 and MIT attribution is retained. The wrapper and
source checks are original Apache-2.0.

Build this layer over the verified local SparkGLM reference image with this
Dockerfile, then build any E3 experiment layer over that result. Supply the
complete source revision as SPARKGLM_SOURCE_REVISION and preserve both rank
image IDs. LLooM materialize --mxfp8-draft selects the pinned 1.2 GiB draft;
Draft TP stays at 2. The inherited DFlash loader preserves the target parallel
group and ignores the independently configured draft TP1; LLooM now rejects
that request rather than mislabeling the run. A true TP1 implementation needs
separate load/forward group handling and compatible shared embeddings/head.

The copied code preserves quantization configuration in the selector and
grouped-convolution projections and retains/repacks fused context K/V scales.
The target quantization and ordinary BF16 draft remain unchanged by default.
Source checks and an image build do not establish numeric or model quality.
Record actual MXFP8 backend selection, serving semantics, speculative acceptance,
and the complete frozen performance matrix before recommending this option.

The first full EXL3/MXFP8 trial used 16384-token prefill chunks and 65536
context. It loaded both models but failed cache admission: 15.20 GiB needed
versus 15.18 GiB automatically available, with an estimated maximum context
of 64512. Large prefill chunks inflate the padded sliding-window draft cache
reservation. This was a failed startup, not a performance result; the next
screen returned to the established 7168-token prefill budget.
