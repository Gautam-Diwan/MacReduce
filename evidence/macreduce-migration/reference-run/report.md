# MacReduce run report

Timings are local monotonic durations. Cross-node clocks are not synchronized. Transfer timings include request and server overhead; byte counts are application payload, not radio traffic.
Nested phases overlap (for example HTTP polling inside barrier wait); do not add all rows to estimate total elapsed time. First-step timings include warmup effects.

| Node | Phase | Count | Total s | p50 ms | p95 ms |
|---|---|---:|---:|---:|---:|
| coordinator | checkpoint_commit | 8 | 0.007 | 0.869 | 1.238 |
| coordinator | decode_update | 16 | 0.013 | 0.687 | 1.427 |
| coordinator | evaluator_init | 1 | 0.784 | 783.611 | 783.611 |
| coordinator | reduce | 8 | 0.001 | 0.165 | 0.193 |
| coordinator | validation | 8 | 0.012 | 1.458 | 1.673 |
| large | backend_init | 1 | 0.488 | 487.675 | 487.675 |
| large | barrier_wait | 9 | 0.789 | 154.524 | 158.000 |
| large | checkpoint_export | 8 | 0.002 | 0.314 | 0.358 |
| large | checkpoint_load | 8 | 0.010 | 1.109 | 1.661 |
| large | data_batch | 80 | 0.002 | 0.013 | 0.043 |
| large | http/checkpoint | 8 | 0.011 | 1.333 | 1.875 |
| large | http/job | 14 | 0.014 | 0.963 | 1.473 |
| large | http/register | 1 | 0.004 | 3.589 | 3.589 |
| large | http/update | 8 | 0.031 | 3.939 | 5.772 |
| large | train_step | 80 | 0.061 | 0.629 | 1.105 |
| small | backend_init | 1 | 0.487 | 487.273 | 487.273 |
| small | barrier_wait | 9 | 0.637 | 1.306 | 158.244 |
| small | checkpoint_export | 8 | 0.003 | 0.305 | 0.572 |
| small | checkpoint_load | 8 | 0.011 | 1.217 | 2.136 |
| small | data_batch | 80 | 0.002 | 0.013 | 0.045 |
| small | http/checkpoint | 8 | 0.008 | 1.070 | 1.341 |
| small | http/job | 13 | 0.011 | 0.872 | 1.335 |
| small | http/register | 1 | 0.002 | 2.462 | 2.462 |
| small | http/update | 8 | 0.034 | 3.802 | 6.635 |
| small | train_step | 80 | 0.075 | 0.502 | 0.931 |

| Node | Local training tokens | Completed step time s | Tokens per training second |
|---|---:|---:|---:|
| large | 10240 | 0.061233 | 167231.5 |
| small | 5120 | 0.075317 | 67979.3 |

These are compute-only rates, including warmup, not end-to-end cluster throughput.
Committed tokens: 15360. Retry events: 0.
Successful request payloads: 1975316 bytes sent and 1985367 bytes received.
Failed-attempt bytes are not included because delivery is unknown.

| Version | Validation loss | Tokens in round |
|---:|---:|---:|
| 1 | 3.85209 | 1920 |
| 2 | 3.25100 | 1920 |
| 3 | 2.74762 | 1920 |
| 4 | 2.42794 | 1920 |
| 5 | 2.15102 | 1920 |
| 6 | 2.01371 | 1920 |
| 7 | 1.87973 | 1920 |
| 8 | 1.76806 | 1920 |

Peak observed single-process RSS: 208994304 bytes.
This is sampled, not the allocator peak or summed cluster RAM.

System power is only available if separately captured. ANE activity and M5 tensor dispatch cannot be inferred from these training spans.
