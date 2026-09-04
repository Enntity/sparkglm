# TP2 collective schedule and transport diagnosis

Date: 2026-09-03

Status: measured and rejected as an endpoint change. Retain the harness and
transport diagnosis; do not adopt the isolated schedule.

## Question

GLM-5.3 executes 91 TP reductions per target forward on this two-Spark
appliance. This experiment asks whether the production PyNCCL path itself is
inefficient, whether a different exact two-rank primitive wins, and whether a
smaller NCCL schedule better fits GB10.

`benchmarks/sparkglm/tp2_collectives.py` runs directly against
`torch.distributed` from the production image without loading model weights. It
measures the exact BF16 row-parallel shapes, including the 2,048x4,096 target
prefill tensor and the 7,104x4,096 maximum scheduled tensor. Every alternative
is checked against a conventional all-reduce with non-zero rank-local inputs.

## Baseline

The default NCCL schedule selected 64 channels. Median all-reduce times were:

| Tokens x hidden | Tensor | Median | Algorithmic bandwidth |
| --- | ---: | ---: | ---: |
| 128 x 4,096 | 1 MiB | 0.6337 ms | 1.541 GiB/s |
| 512 x 4,096 | 4 MiB | 0.8264 ms | 4.726 GiB/s |
| 2,048 x 4,096 | 16 MiB | 1.5246 ms | 10.249 GiB/s |
| 7,104 x 4,096 | 55.5 MiB | 5.7372 ms | 9.447 GiB/s |

The 55.5 MiB result matches the 4.5-6.6 ms all-reduces in the synchronized
production trace. The standalone harness therefore reproduces the serving
transport cost without vLLM scheduling or model execution.

## Exact primitive alternatives

At 55.5 MiB, a two-rank peer exchange followed by a local add took 5.825 ms.
Reduce-scatter followed immediately by all-gather took 6.138 ms. Both produced
exactly the same BF16 tensor as all-reduce, but neither removed cost when there
was no useful sequence-sharded operation between its halves.

Forcing NCCL's Simple protocol was tied at 5.764 ms. LL took 46.53 ms and LL128
took 8.43 ms. Protocol forcing is rejected.

## Channel and RoCE QP sweep

Eight channels gave the best result across the activation sizes that dominate
the target forward. Adding four unsplit QPs per connection improved the long
transfer while preserving the mid-size result:

| Configuration | 4 MiB | 16 MiB | 55.5 MiB |
| --- | ---: | ---: | ---: |
| NCCL default | 0.8264 ms | 1.5246 ms | 5.7372 ms |
| max channels 8 | 0.3665 ms | 1.4268 ms | 5.1414 ms |
| max channels 8, QPs 4, split 0 | 0.3544 ms | 1.3041 ms | 4.7417 ms |

The selected candidate is therefore:

```text
NCCL_MAX_NCHANNELS=8
NCCL_IB_QPS_PER_CONNECTION=4
NCCL_IB_SPLIT_DATA_ON_QPS=0
```

Against the standalone default, this is 57.1% faster at 4 MiB, 14.5% faster at
16 MiB, and 17.4% faster at 55.5 MiB. TP communication is only 10-12% of the
profile, so the expected endpoint effect is approximately 1-2%, not a
double-digit claim. This qualified the schedule for the endpoint A/B below.

## Root transport finding

NCCL INFO reports `GPU Direct RDMA Disabled` on both ranks. The direct CX-7
fabric is linked at 200 Gbit/s, but the Ethernet interfaces use MTU 1,500 and
the active RDMA MTU is 1,024 despite a maximum of 4,096.

The host RDMA userspace is version 50 and lacks the `MLX5_1.25` symbols
`mlx5dv_reg_dmabuf_mr` and `mlx5dv_get_data_direct_sysfs_path` that NCCL probes.
An isolated current rdma-core build supplies both symbols, but using it alone
does not enable GDR. The installed open NVIDIA driver includes
`nvidia-peermem.ko`, but that module is not loaded.

The next reversible root experiment is:

1. load `nvidia-peermem` on both Sparks;
2. set the direct CX-7 interfaces to MTU 9,000 on both Sparks;
3. prove jumbo transport and active RDMA MTU 4,096;
4. rerun NCCL INFO and this benchmark with current rdma-core userspace;
5. roll back MTU and unload the module if correctness or connectivity fails.

No persistent network or module configuration should change until that test
demonstrates GDR and a repeatable collective improvement.

## Model-loaded endpoint gate

The candidate was loaded in the accepted M64 image with no other configuration
change. Both ranks reported the intended three NCCL environment values. The
identical five-second-stagger medium-C2 gate then ran twice with new salts and
128 forced output tokens.

| Run | Prefill tok/s | Aggregate decode tok/s | Wall |
| --- | ---: | ---: | ---: |
| Frozen accepted M64 control | 1,379.47 | 29.23 | 28.84 s |
| NCCL candidate pass 1 | 1,314.55 | 25.14 | 31.33 s |
| NCCL candidate pass 2 | 1,352.06 | 19.66 | 29.51 s |
| Restored default pass 1, cold | 1,046.30 | 28.87 | 35.56 s |
| Restored default pass 2 | 1,297.44 | 27.69 | 30.50 s |
| Restored default pass 3 | 1,325.03 | 29.22 | 29.79 s |

All ten requests completed 128/128 tokens, had their own marker and no peer
marker, and returned no API error. Correctness passed. The default schedule's
third converged aggregate-decode result reproduced the frozen 29.23 tok/s
control, while both candidate passes were lower by 14.0% and 32.7%. Candidate B
incurred a 5.06-second maximum inter-token gap.

Absolute prefill and wall time varied materially across boots: the candidate's
1,315-1,352 tok/s range overlaps the restored default's converged 1,325 tok/s.
The data therefore does not prove a prefill regression from the channel cap.
It does prove an unacceptable concurrent-decode regression, which is sufficient
to fail the appliance gate.

The isolated transfer speedup does not compose with the model's communication
and compute schedule. Capping channels likely removes useful overlap or
progress capacity even though a lone all-reduce completes sooner. The recipe
change is rejected and the large-C2 run is intentionally skipped after the
primary medium gate failed.

## Verdict

Do not adopt the 8-channel, four-QP schedule. The useful result is diagnostic:
the generic all-reduce primitive is not replaceable by a simple peer exchange,
and standalone NCCL timing is insufficient to select an endpoint schedule.
The larger opportunity remains below vLLM: the present transport is host-staged
and using a small RoCE MTU. Repairing GPUDirect and the fabric MTU is the next
root-layer experiment.
