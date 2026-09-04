#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Model-free top-k correctness/graph probe for the restored video foundation.

Uses the vLLM operator contract, torch.topk as an exact selected-value oracle,
and deterministic inputs. This is not an endpoint performance benchmark.
"""
import hashlib
import json

import torch
import vllm._C_stable_libtorch  # noqa: F401: registers torch.ops._C


def main():
    results = []
    for dtype in (torch.float16, torch.float32):
        for mode, rows in (("decode", 1), ("decode", 4), ("decode", 32),
                           ("prefill", 64), ("prefill", 512)):
            for columns in (4096, 8192, 32769):
                torch.manual_seed(17)
                logits = torch.randn((rows, columns), dtype=dtype, device="cuda")
                ends = torch.full((rows,), columns, dtype=torch.int32, device="cuda")
                starts = torch.zeros_like(ends)
                out = torch.empty((rows, 512), dtype=torch.int32, device="cuda")

                def invoke():
                    if mode == "prefill":
                        torch.ops._C.top_k_per_row_prefill(
                            logits, starts, ends, out, rows, columns, 1, 512)
                    else:
                        torch.ops._C.top_k_per_row_decode(
                            logits, 1, ends, out, rows, columns, 1, 512)

                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        invoke()
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    invoke()
                signature = None
                index_sets = []
                for _ in range(3):
                    graph.replay()
                    torch.cuda.synchronize()
                    assert bool(((out >= 0) & (out < columns)).all()), "out-of-range index"
                    ids = out.long()
                    assert bool((ids.sort(dim=1).values.diff(dim=1) > 0).all()), "duplicate index"
                    selected = logits.gather(1, ids).sort(dim=1).values
                    oracle = logits.topk(512, dim=1).values.sort(dim=1).values
                    assert torch.equal(selected, oracle), "selected values differ from torch.topk"
                    # Native top-k promises selected values, not a sorted output
                    # or stable choice among equal FP16 boundary scores.
                    current = hashlib.sha256(selected.cpu().numpy().tobytes()).hexdigest()
                    index_sets.append(hashlib.sha256(
                        ids.sort(dim=1).values.cpu().numpy().tobytes()).hexdigest())
                    assert signature in (None, current), "graph replay changed selected values"
                    signature = current
                result = {"mode": mode, "dtype": str(dtype), "rows": rows,
                          "columns": columns, "top_k": 512, "selected_values_sha256": signature,
                          "index_sets_stable": len(set(index_sets)) == 1,
                          "exact_values": True, "graph_replays": 3}
                results.append(result)
                print(json.dumps(result), flush=True)
    print(json.dumps({"passed": True, "cases": results}), flush=True)


if __name__ == "__main__":
    main()
