# NVIDIA 512K / 11 GiB source-preview default

The maintainer requested NVIDIA weights, 512K context and 11 GiB KV per rank
as the repository default. The managed pair successfully started and returned
an exact synthetic reply through the authenticated gateway as
`sparkglm-nvfp4-nvidia`, HTTP 200, without failover. It was left running for
manual use. This is a maintainer-selected source preview, not G5 promotion.

An earlier 11 GiB load was interrupted by an assistant-authored guard after
three samples below 1.5 GiB. The user explicitly requested removing that abort
behavior. The subsequent startup completed without the guard. No CUDA OOM was
established by the interrupted run. Do not equate a guard-triggered stop with
an intrinsic model startup failure. Only passive memory sampling was used in
the successful run.

See measurement.json for head-memory samples and gateway outcome, and
environment.json for both live image identities and core settings. No new
C4 benchmark, broad semantic suite, repeated cold-start qualification or
endurance test was run at 11 GiB. Four active sequences remain configured;
throughput and preemption behavior at this cache size remain unmeasured.

The source-only defaults select the existing NVIDIA profile and retain Red Hat
512K/9 GiB through --profile nvfp4. Model revisions, DFlash2 depth 7, TP2,
CUTLASS and prefill settings are unchanged. No weights or external serving code
were copied or modified by this default selection.
