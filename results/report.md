# Report

Run started 2026-10-28T17:03:38-0300, wall time 21 s; 9 uncached LLM calls (20 s of model time) out of 662 calls.

## Answers

| system | accuracy (95% CI) | answerable | false "I don't know" | stale (now) | wrong period (as of) | unanswerable declined | isolation declined | isolation leaks | correct and cites a gold episode | median prompt tokens | median answer latency |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Batch GraphRAG | 60.5% [49.4, 71.6] | 49.2% | 15.9% | 33.3% | 8.3% | 100.0% | 100.0% | 0 | 100.0% | 1260 | 5.73 s |
| Hybrid (BM25 + LSA + RRF over episodes) | 61.7% [50.6, 71.6] | 50.8% | 6.3% | 4.8% | 25.0% | 100.0% | 100.0% | 0 | 96.9% | 714 | 4.95 s |
| Incremental temporal graph | 95.1% [90.1, 98.8] | 93.7% | 6.3% | 0.0% | 0.0% | 100.0% | 100.0% | 0 | 98.3% | 905 | 2.77 s |

Isolation probes against one index shared by all brokers (hybrid, for contrast): 8 leaks and 0 declined out of 9.

### Accuracy by question type

| system | single_hop (15) | temporal_current (15) | as_of (12) | multi_hop (12) | aggregation (9) | unanswerable (9) | isolation (9) |
|---|---|---|---|---|---|---|---|
| Batch GraphRAG | 93.3% | 66.7% | 25.0% | 0.0% | 44.4% | 100.0% | 100.0% |
| Hybrid (BM25 + LSA + RRF over episodes) | 86.7% | 93.3% | 33.3% | 8.3% | 0.0% | 100.0% | 100.0% |
| Incremental temporal graph | 100.0% | 100.0% | 66.7% | 100.0% | 100.0% | 100.0% | 100.0% |

### Paired bootstrap (accuracy, points, 95% CI, 10,000 resamples)

| comparison | all | single_hop | temporal_current | as_of | multi_hop | aggregation | unanswerable | isolation |
|---|---|---|---|---|---|---|---|---|
| batch_graph vs hybrid | -1.2 [-9.9, +8.6] | +6.7 [+0.0, +20.0] | -26.7 [-53.3, -6.7] | -8.3 [-41.7, +25.0] | -8.3 [-25.0, +0.0] | +44.4 [+11.1, +77.8] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| incremental_graph vs batch_graph | +34.6 [+23.5, +45.7] | +6.7 [+0.0, +20.0] | +33.3 [+13.3, +60.0] | +41.7 [+0.0, +83.3] | +100.0 [+100.0, +100.0] | +55.6 [+22.2, +88.9] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| incremental_graph vs hybrid | +33.3 [+22.2, +44.4] | +13.3 [+0.0, +33.3] | +6.7 [+0.0, +20.0] | +33.3 [-8.3, +75.0] | +91.7 [+75.0, +100.0] | +100.0 [+100.0, +100.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |

## Graph quality against the ground truth

| graph | nodes | wrong-merge rate | duplication rate | clients: nodes / true | buildings: nodes / true | relation precision | relation recall | current-value accuracy | latest-stated value correct | superseded facts closed (exact date) |
|---|---|---|---|---|---|---|---|---|---|---|
| incremental_graph | 80 | 0.0% (0) | 2.6% | 21 / 21 | 20 / 18 | 96.6% | 98.6% | 97.9% | 97.9% | 33 (33) of 33 |
| incremental_no_llm_merge | 79 | 0.0% (0) | 1.3% | 21 / 21 | 19 / 18 | 96.6% | 98.6% | 97.9% | 97.9% | 33 (33) of 33 |
| batch_graph | 106 | 0.0% (0) | 36.4% | 39 / 21 | 28 / 18 | 96.6% | 98.6% | 81.5% | 97.9% | 0 (0) of 33 |

Resolution routes (incremental): {"exact": 119, "llm": 3, "llm-new": 1, "new": 58, "rule": 10, "thread": 38, "thread-new": 21}

## Cost of absorbing new episodes

32 new episodes (the last quarter of each broker's stream, 130 in total).

| strategy | LLM calls | tokens | LLM seconds | python seconds | LLM seconds per new episode |
|---|---|---|---|---|---|
| incremental: fold the new episodes in | 34 | 21,313 | 71 | 0.000 | 2.2 |
| batch: rebuild from scratch | 146 | 101,826 | 1106 | 0.001 | 34.6 |
| batch: rebuild, extractions replayed from a cache | 16 | 12,558 | 389 | 0.001 | 12.1 |
| hybrid: re-index | 0 | 0 | 0 | 0.010 | 0 |

One extraction call: median 4.2 s, mean 5.5 s, 657 prompt and 29 output tokens on average.

Calls by tag: {"ablation:extract": 130, "answer:batch_graph": 81, "answer:hybrid": 81, "answer:hybrid_shared_index": 9, "answer:incremental_graph": 81, "batch:extract": 130, "batch:summary": 16, "inc:extract": 130, "inc:merge": 4}

