# 512K context / 9 GiB native NVFP4 candidate

LLooM-managed TP2, native FLASHINFER_CUTLASS, MXFP8 DFlash2 TP2 k7, 2048-token prefill chunks, FP8 target cache, maximum four sequences. No sidecars on either GPU host. Same immutable images as native-2k candidate; explicit MoE backend and 524288 context limit.

The 523264-token cold request passed with TTFT 341.758 seconds; two cached tool turns took 1.935 and 2.068 seconds. All three automatic tool calls had correct arguments and tool_calls finish reasons.

Four clients each submitted a distinct 64512-token prompt and two cached tool continuations. All twelve calls were correct. Cold TTFTs were 42.726, 85.565, 124.355, and 164.252 seconds. Cached TTFTs ranged from 7.380 to 151.347 seconds under contention. The engine sometimes queued the fourth request for capacity and also reached four active requests; this is four-client completion, not a guarantee of uninterrupted four-way residency. The previous 512K prefix remained eligible for caching when this test started. Two preemptions were recorded by the end of the concurrent workload; the preceding 512K single-client test had zero. This limits the concurrency performance claim.

Startup reported 662356 effective cache tokens and 1.26x concurrency at the full 524288-token limit. This is not a flat token pool to divide across arbitrary sequence lengths.

Minimum sampled available RAM during these requests: head 2.737 GiB; worker 7.678 GiB. Samples are five seconds apart and can miss transient peaks. Both nodes remained reachable; this is a bounded successful test, not endurance qualification or a large safety reserve.

The preceding 1M/11GiB trial failed and required physical worker recovery. Recovered previous-boot logs contained NVIDIA NV_ERR_NO_MEMORY allocation errors; exact peak allocation ownership is unresolved. That configuration is not qualified. Earlier 512K/11GiB tests completed, but are separate runs.

Final installed operating point is 512K/9GiB under LLooM. Presence stays on the authorized cloud fallback; the local Qwen lane remains suspended with keep-warm disabled. Embedding and Flux remain on the Mac. No public default promotion or release.
