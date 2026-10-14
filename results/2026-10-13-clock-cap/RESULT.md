# GPU clock cap: a Spark that powered off under load — 2026-10-13

One Spark of our pair (spark-b, rank 1) twice powered itself **off** during long
cold prefills: a 97K-token and then a 112K-token prompt with images, which skip
the prefix cache, so every turn was a full prefill. The unit was off, not hung.
Nothing appeared in any log, memory was flat (5.7 GB free, zero pressure) up
to the last second, and the GPU read 81 °C. The other Spark (spark-a, rank 0)
logged an RDMA completion error 40 s after the peer dropped.

## Reproduced without SparkGLM

`raw/gpustress.py`: back-to-back bf16 8192³ matmuls (about 3 GB), run on
both Sparks at once in `vllm/vllm-openai:v0.30.0`. `raw/stressguard.sh`
stops it if any ACPI thermal zone reaches the guard temperature. A sampler on a
third host logged both Sparks every second (`raw/sampler-*-1s.log`; columns:
MemAvailable kB, memory PSI, GPU °C / W / SM MHz / clock-event bitmask, hottest
thermal zone °C).

| Run | Spark | Result | GPU W avg (max) | SM MHz avg (max) | Hottest zone max | TFLOPS |
|---|---|---|---|---|---|---|
| Uncapped | spark-a | stopped by us at 98 °C after 335 s | 89.8 (93.5) | 2130 (2229) | 98 °C | 92–94 |
| Uncapped | spark-b | **powered off after ~100 s** | 89.9 (93.4) | 2237 (2314) | 98 °C | 93 |
| `-lgc 300,2200`, guard 98 °C | spark-a | stopped by guard at 456 s | 86.7 (90.9) | 2101 (2177) | 98 °C | 91–93 |
| `-lgc 300,2200`, guard 98 °C | spark-b | stopped by guard at 406 s | 80.3 (84.6) | 2151 (2184) | 98 °C | 91–93 |
| `-lgc 300,2200`, guard 101 °C, started hot | spark-a | **full 600 s** | 84.1 (90.0) | 2071 (2125) | 98 °C | 90.4–91.1 |
| `-lgc 300,2200`, guard 101 °C, started hot | spark-b | **full 600 s** | 77.5 (84.6) | 2114 (2184) | 98 °C | 90.4–90.8 |

The uncapped spark-a run and the guard-98 runs were stopped by us, not by the
Spark. The uncapped spark-a row uses the sampler only up to spark-b's power-off;
its stress log is in `raw/uncapped-both.log`.

## What it shows

- The trip is not the OS-visible temperature. Capped, both Sparks held the
  hottest zone at 97–98 °C for minutes without powering off; uncapped, spark-b
  died at the same 97–98 °C. The kernel's ACPI critical trip is 104 °C and never
  fired.
- What the cap changes is clock and power: spark-b, which boosted higher
  uncapped (2237 MHz average, peaks of 2314), drew ~12 W less capped (89.9 W to
  77.5 W). Community reports on GB10 power-offs describe the same symptom and
  the same fix ([NVIDIA forum](https://forums.developer.nvidia.com/t/hard-power-off-under-sustained-gpu-load-at-90w-persists-after-full-platform-firmware-update/378315),
  [an 8-unit diagnosis](https://petronellatech.com/blog/dgx-spark-shutdown-under-load-our-8-unit-gb10-diagnosis/)).
- nvidia-smi's GPU temperature reads 11–17 °C below the hottest board zone, so
  it is no guide to how close a Spark is to tripping.
- Cost on this worst-case matmul: about 3% (93 to 90.5 TFLOPS). The cost on
  SparkGLM decode and prefill has not been measured.
- Firmware was not the cause: EC `0x03000508`, SoC `0x02009b0b` and USB-C PD
  `0x516` matched on both Sparks, with ESRT last-attempt status 0. spark-b's
  DMI BIOS string is an older factory build (`5.36_0ACUM018`, 2025-08-06) than
  spark-a's (`5.36_0ACUM027`, 2026-06-12).

## Change

`gpu-clock-cap.sh` installs `gpu-clock-cap.service`, which runs
`nvidia-smi -lgc 300,2200` at boot (the lock does not survive a reboot).
`./start.sh clock-cap` runs it on both Sparks with `sudo`, and `./start.sh`
refuses to serve until the service is active on both (`CLOCK_CAP=0` skips the
check). Two Sparks, one ambient; a Spark that still powers off at 2200 needs a
lower cap or better airflow.
