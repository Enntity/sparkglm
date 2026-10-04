# Fast strict JSON, bounded grammar memory, display-carveout KV (engine sparkglm/atlas-20261004-gram)

**Grammar memory.** A strict object schema with many required keys used host memory superlinearly in the previous
engine: 0.77 GB at 32 keys, 2.7 GB at 64, and the engine was killed at 96. About 3 GB stayed resident across
requests. Grammar compile is now linear in schema size and exact: golden digests of every reachable token mask
match the previous engine across a varied grammar set. On the pair, every schema from 8 to 96 keys costs 80-144 MiB
while it runs, and idle memory returns to baseline. A schema whose grammar an FSM edge cannot encode (more than
32,768 rules or 10,922 bounded repetitions) is refused with HTTP 400 before compile, and is never cached.

**Fast strict JSON** (`ATLAS_GLM_STRICT_SPEC=1`, on in both profiles). Strict requests keep DFlash speculation.
Every verified row is masked by the grammar state after the drafts before it, following the design of vLLM #14702
and #44297, and each rank masks its own vocab half before the cross-rank argmax. A GPU bitcheck passed 8,640 checks.
Measured on the pair:

| Request | Serial masked decode | Speculative masked decode |
|---|---:|---:|
| small strict schema | 8-10 s | 2.8-3.1 s |
| 36-memory consolidation (~2,000 tokens) | 135-155 s | 27-35 s |
| 96-key strict object at temperature 0 | killed (previous engine) | 39 s, 59.6 tok/s, all 96 keys |

All outputs are schema-valid, with 0 reasoning when thinking is off. Requests without a schema are unchanged:
logprob hash `8c75d2886794`, same decode speed. Like other DFlash requests, verify positions are greedy.

**Display carveout** (fork-only, `DISPLAY_CARVEOUT=1`, default off). GB10 firmware reserves 2,046 MiB for display
that the driver never uses on GB10. With the switch on, whole per-layer KV pools are placed there. Measured by the
GPU unified memory optimization session: 70,936 -> 91,350 blocks per rank at 0.91 (+28.8%, 1.135M -> 1.462M tokens),
with unchanged system memory, identical greedy hashes, det and prefix-cache checks, and the same decode speed.
The container needs CAP_SYS_ADMIN to export the memory; the launcher drops it before the engine starts.

Raw output: `raw/gm.log`.
