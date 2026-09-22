# Three repeated C2 coding-agent pairs

| Pair completion | SparkGLM NVFP4 adaptive | Stock Mia EXL3 |
|---|---:|---:|
|1|197.93s|260.65s|
|2|281.99s|228.29s|
|3|355.18s|263.05s|
|Median|281.99s|260.65s|

Mia had a7.6% shorter median and a lower observed spread on this one offline coding task. All12 sessions passed the original14-case CLI verifier. Expanded500-case summary checks plus an empty-ID regression passed5/6 SparkGLM implementations and6/6 Mia implementations. SparkGLM s1/lane-2 silently dropped ID-less events. Both SparkGLM s1 sessions and Mia m1/lane-1 omitted the requested new test. Mia m1/lane-2 removed an existing exit-code assertion while adding a retry test. There is no evidence here of intentional test evasion or a quantization-caused quality difference.

SparkGLM generated26,207 output tokens across89 model calls; Mia generated19,556 across97. Different trajectories and output lengths contribute to task time. This is not a matched decode throughput measurement. Both recipes made edit errors and recovered; initial expected failing tests are not automatically tool-reasoning failures.

Each pair contains two OMP18.2.6 sessions starting10s apart on fresh copies of the same fixture, explicit High reasoning, thinking enabled, clear_thinking=true and32768 output cap. Checkpoint temperature/top-p defaults matched. Inputs/tools matched after normalizing temporary workspace names. All186 calls had intended local backend attribution, HTTP200 and no failover. No request preemptions or foreign benchmark traffic were observed. Every measured pair is retained, including the fastest SparkGLM pair containing the buggy solution.

The three SparkGLM pairs ran first; stock Mia received its pinned startup warmup before its three pairs. Order was grouped to save reload cost, not randomized. This is one small task and three repetitions, not a general coding-quality ranking. Raw requests, SSE, model reasoning, tools and full source snapshots are preserved privately; only selected aggregate/quality evidence is public.

Mia revision:775a58b704924b13cf5c38de97559b3630f67bbf, stock fixed DFlash K7, EXL3 TR3 4bpw, four sequence slots and7168 batched tokens. SparkGLM: NVIDIA NVFP4/MXFP8, adaptive2/4/5, eight sequence slots and2048 batched tokens, SSD prefix option. They share the base vLLM version but differ in complete serving configuration. [Source profile](../../../research/adaptive-verification/README.md). EXL3 quantization credit and required notice: [Brandon M. Music / ShapleyMCG attribution](../../../docs/QUANT_ATTRIBUTION.md).
