# MacReduce run report

Timings are local monotonic durations. Cross-node clocks are not synchronized. Transfer timings include request and server overhead; byte counts are application payload, not radio traffic.
Nested phases overlap (for example HTTP polling inside barrier wait); do not add all rows to estimate total elapsed time. First-step timings include warmup effects.

| Node | Phase | Count | Total s | p50 ms | p95 ms |
|---|---|---:|---:|---:|---:|
| coordinator | checkpoint_commit | 3 | 0.002 | 0.676 | 0.809 |
| coordinator | decode_update | 6 | 0.004 | 0.710 | 0.907 |
| coordinator | evaluator_init | 1 | 0.027 | 27.175 | 27.175 |
| coordinator | reduce | 3 | 0.000 | 0.145 | 0.154 |
| coordinator | validation | 3 | 0.005 | 1.828 | 1.872 |
| large | backend_init | 1 | 0.000 | 0.404 | 0.404 |
| large | backend_probe | 1 | 0.029 | 28.577 | 28.577 |
| large | barrier_wait | 4 | 0.312 | 77.684 | 155.947 |
| large | checkpoint_export | 3 | 0.001 | 0.270 | 0.305 |
| large | checkpoint_load | 3 | 0.003 | 0.770 | 1.197 |
| large | data_batch | 12 | 0.000 | 0.018 | 0.035 |
| large | http/artifact/chunk | 6 | 0.007 | 1.026 | 1.519 |
| large | http/artifact/complete | 3 | 0.011 | 4.579 | 4.939 |
| large | http/artifact/manifest | 3 | 0.002 | 0.677 | 0.793 |
| large | http/artifact/update | 3 | 0.003 | 0.986 | 1.102 |
| large | http/job | 6 | 0.005 | 0.784 | 0.930 |
| large | http/register | 1 | 0.004 | 3.537 | 3.537 |
| large | train_step | 12 | 0.027 | 1.574 | 5.601 |
| small | backend_init | 1 | 0.000 | 0.439 | 0.439 |
| small | backend_probe | 1 | 0.032 | 31.760 | 31.760 |
| small | barrier_wait | 4 | 0.317 | 79.192 | 157.895 |
| small | checkpoint_export | 3 | 0.001 | 0.288 | 0.354 |
| small | checkpoint_load | 3 | 0.003 | 0.899 | 1.358 |
| small | data_batch | 12 | 0.000 | 0.022 | 0.040 |
| small | http/artifact/chunk | 6 | 0.007 | 1.191 | 1.556 |
| small | http/artifact/complete | 3 | 0.009 | 2.059 | 4.718 |
| small | http/artifact/manifest | 3 | 0.002 | 0.806 | 0.820 |
| small | http/artifact/update | 3 | 0.003 | 1.141 | 1.191 |
| small | http/job | 6 | 0.005 | 0.802 | 1.267 |
| small | http/register | 1 | 0.003 | 2.893 | 2.893 |
| small | train_step | 12 | 0.065 | 1.564 | 22.715 |

| Node | Local training tokens | Completed step time s | Tokens per training second |
|---|---:|---:|---:|
| large | 1536 | 0.027337 | 56187.3 |
| small | 768 | 0.065016 | 11812.5 |

These are compute-only rates, including warmup, not end-to-end cluster throughput.
Committed tokens: 2304. Retry events: 0.
Successful request payloads: 748434 bytes sent and 752350 bytes received.
Failed-attempt bytes are not included because delivery is unknown.

| Version | Validation loss | Tokens in round |
|---:|---:|---:|
| 1 | 4.52436 | 768 |
| 2 | 4.03579 | 768 |
| 3 | 3.71383 | 768 |

Peak observed single-process RSS: 56492032 bytes.
This is sampled, not the allocator peak or summed cluster RAM.

System power is only available if separately captured. ANE activity and M5 tensor dispatch cannot be inferred from these training spans.
