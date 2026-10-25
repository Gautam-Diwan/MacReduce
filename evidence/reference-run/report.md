# MacTrain run report

Timings are local monotonic durations. Cross-node clocks are not synchronized. Transfer timings include request and server overhead; byte counts are application payload, not radio traffic.
Nested phases overlap (for example HTTP polling inside barrier wait); do not add all rows to estimate total elapsed time. First-step timings include warmup effects.

| Node | Phase | Count | Total s | p50 ms | p95 ms |
|---|---|---:|---:|---:|---:|
| coordinator | checkpoint_commit | 8 | 0.008 | 0.939 | 1.165 |
| coordinator | decode_update | 16 | 0.012 | 0.686 | 1.364 |
| coordinator | evaluator_init | 1 | 0.398 | 398.018 | 398.018 |
| coordinator | reduce | 8 | 0.001 | 0.153 | 0.174 |
| coordinator | validation | 8 | 0.012 | 1.433 | 1.649 |
| large | backend_init | 1 | 0.431 | 431.000 | 431.000 |
| large | barrier_wait | 9 | 0.635 | 1.318 | 157.894 |
| large | checkpoint_export | 8 | 0.002 | 0.250 | 0.304 |
| large | checkpoint_load | 8 | 0.010 | 1.149 | 1.787 |
| large | data_batch | 80 | 0.001 | 0.011 | 0.042 |
| large | http/checkpoint | 8 | 0.007 | 0.936 | 1.115 |
| large | http/job | 13 | 0.012 | 0.898 | 1.388 |
| large | http/register | 1 | 0.003 | 2.777 | 2.777 |
| large | http/update | 8 | 0.031 | 4.001 | 6.219 |
| large | train_step | 80 | 0.056 | 0.552 | 0.992 |
| small | backend_init | 1 | 0.433 | 433.195 | 433.195 |
| small | barrier_wait | 9 | 0.783 | 154.256 | 157.344 |
| small | checkpoint_export | 8 | 0.002 | 0.304 | 0.376 |
| small | checkpoint_load | 8 | 0.011 | 1.241 | 1.864 |
| small | data_batch | 80 | 0.001 | 0.013 | 0.041 |
| small | http/checkpoint | 8 | 0.010 | 1.099 | 1.964 |
| small | http/job | 14 | 0.012 | 0.847 | 1.325 |
| small | http/register | 1 | 0.003 | 2.968 | 2.968 |
| small | http/update | 8 | 0.028 | 3.625 | 5.336 |
| small | train_step | 80 | 0.050 | 0.479 | 0.911 |

| Node | Local training tokens | Completed step time s | Tokens per training second |
|---|---:|---:|---:|
| large | 10240 | 0.055674 | 183929.0 |
| small | 5120 | 0.050431 | 101524.9 |

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

Peak observed single-process RSS: 208601088 bytes.
This is sampled, not the allocator peak or summed cluster RAM.

System power is only available if separately captured. ANE activity and M5 tensor dispatch cannot be inferred from these training spans.
