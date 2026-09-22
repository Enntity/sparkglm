# Compact draft KV: capacity tradeoff

The deployed capacity profile uses compact DFlash pages256, K5,262144 context, eight sequence/admission slots and11GiB KV per rank. It was selected for capacity, accepting an observed4–6% C2/C4 throughput cost. A single clean K5 C2 screen was2.7% longer in wall time and reserved fewer recurrent-state blocks. These are bounded selection observations, not universal speed improvements.

The retained screen summary includes TTFT/event-gap regressions and non-identical output hashes. It does not prove token parity or broad quality equivalence. Eight slots do not imply eight simultaneously resident full262K contexts; physical KV remains shared. No full release/endurance gate is claimed. The source switch remains opt-in.

Original implementation: `overlay/patch_dflash_compact_kv.py` adapts the draft cache allocation geometry using existing vLLM primitives. Underlying vLLM source and inherited patches retain their notices. This is capacity-oriented source preservation, not a new default promotion.
