# Cross-condition TTFT cache-timing comparison

`known_exact` vs. `cold_baseline` TTFT, idle (no concurrent victim traffic),
`max_tokens=1` streamed TTFT probing. Gap = `baseline_mean - known_exact_mean`
(ms); SNR = `gap / baseline_std`, consistent with `attacker_probe/analyze_probes.py`.

| Condition              | Gap (ms) | SNR   | p-value   |
|------------------------|----------|-------|-----------|
| A6000 flat (TCP)       | 316      | 8.54  | n/a*      |
| A6000 disagg (TCP)     | 594      | 15.52 | n/a*      |
| Spark disagg (RDMA)    | 55.4     | 9.61  | 1.71e-18  |
| Spark flat (single)    | 25.9     | 22.28 | 2.14e-13  |

\* p-value not supplied for the A6000 rows — left as "n/a" rather than invented.

## Provenance

- **Spark disagg (RDMA)** and **Spark flat (single)** rows: measured directly
  in this session.
  - Spark disagg: `runs/probes_idle_fixed.jsonl` (idle probe run against the
    Dynamo PD-disaggregated setup, prefill on spark-523e + decode on
    spark-4f80, before it was stopped). n=14/13 (known_exact/cold_baseline).
  - Spark flat: `runs/probes_flat_spark_idle.jsonl` (this run — flat
    single-node vLLM on spark-523e, `--enable-prefix-caching`, no
    disaggregation, 120s duration, 1.5s interval). n=24/24.
- **A6000 flat (TCP)** and **A6000 disagg (TCP)** rows: supplied by the user
  as pre-existing values from an external paper. **Not independently
  measured or verified in this session** — no A6000 hardware or prior paper
  data was available here to check them against.

## Note on comparing Gap (ms) across rows

The four rows come from different hardware, network transports (TCP vs
RDMA), and — for the A6000 rows — an entirely different session with
unknown prompt/config details, so raw Gap (ms) isn't strictly apples-to-apples:
absolute TTFT and its variance both shift with hardware/interconnect/serving
architecture, independent of whether caching is working. That's the reason
this project reports SNR (gap normalized by baseline std) as the primary
cross-condition metric — e.g. Spark flat has the smallest raw gap (25.9ms)
of the four but the highest SNR (22.28), because its baseline TTFT is both
low and extremely tight (std ≈ 1.2ms) rather than because caching is doing
more there. Read Gap and SNR together, not Gap alone.

## Spark flat — full summary (this run)

```
known_exact:       mean=144.3ms  n=24
cold_baseline:      mean=170.2ms  n=24
near_miss_control:  mean=183.8ms  n=24  (SNR=-11.66, p=1.62e-12 — correctly
                                          shows NO cache-hit benefit, consistent
                                          with the per-probe-unique near-miss fix)

Welch's t-test, known_exact vs cold_baseline:
  t=14.62, p=2.136e-13, gap=25.9ms, SNR=22.28  -> SIGNIFICANT reduction
```
