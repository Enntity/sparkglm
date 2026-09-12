# Atlas NVIDIA NVFP4 bring-up experiment

The refreshed Atlas engine runs the full NVIDIA target on two Sparks in ordinary
TP2/EP2 C1 mode. It remains disabled in SparkGLM defaults. The patch and image
identities are bound in measurement.json. No vLLM comparison or promotion is claimed.

The final build passes 68 focused native tests. Live chat, arithmetic, JSON and
tool-result probes complete, with observed unrelated reasoning and premature
text alongside a tool call. Retrieval passes at 577 and 1897 input tokens. A
3395-token case places the correct code only in reasoning, with no final answer;
the initial image passed that case. These are quality failures, not a semantic
gate pass.

The streaming correction delivers a previously lost short reasoning tail in
its original channel. It does not reclassify reasoning as final-answer content.
Cancellation releases the slot for the next correct reply. The unused predictor
skip reduces reported uploaded weights from 103.90 to 96.80 GiB per rank.

Native MTP was refused by preflight: 116.18 GiB estimated peak plus a 4 GiB reserve
exceeds 114.84 GiB free. Docker reported no OOM kill. Staged predictor loading
is still needed.

Raw evidence remains private; selected synthetic response fields and source
receipt hashes are retained in measurement.json. This is bounded bring-up,
not the complete G1/G2/G3/G4/G5 qualification matrix.
