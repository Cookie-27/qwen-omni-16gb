# Historical evidence and its boundaries

Experiment date: 2026-09-29. One conversational Qwen2.5-Omni-3B Thinker, fixed checkpoint revision f75b40e3da2003cdd6e1829b1f420ca70797c34e. Main modules BF16, full-vocabulary last-token head FP32; batch one, no KV cache, no generated text, no training.

The fixed MUSIC-AVQA development subset contains 16 videos and 32 training-source questions, grouped into eight crossed groups. Each question has two source-label-derived candidates, two option orders, and four media conditions. It is easier than the original 42-answer setting and is not an independent benchmark.

| Condition | Correct / 64 contexts |
|---|---:|
| Real audio and images | 40 |
| Silent audio, original images | 41 |
| Original audio, gray images | 35 |
| Silent audio, gray images | 32 |

The denominator is 32 questions times two option orders, not 64 independent questions. The results do not establish an audio accuracy gain. They demonstrate a completed runtime, not successful multimodal decision training.

There were 269 forwards: eight short numerical comparisons, one longest-input probe, 256 main contexts and four fresh-process reloads. The four short stock/headsplit full-vocabulary vectors are byte-identical in the exported .npy files. The longest probe and four reloads equal their main-run results. All 256 candidate probabilities and answer mappings are recomputed by verify_evidence.py.

The original stock longest-context attempt OOMed. An earlier query-row partitioning attempt failed a predeclared candidate-probability tolerance (about 0.028 versus a 0.02 limit). That attempt was stopped. The head-partition version kept the same tolerance. This history is not evidence that every query-row implementation is wrong or that head partitioning is optimal.

The CPU historical operator archive contains 32 checks. A separate length-5521 operator test compared 16 selected boundary/end rows to independent FP64 calculations in two precisions. It is not a complete GPU-model equivalence proof on long inputs.

Runtime: 10.202 GiB peak torch allocated, 12,001 MiB sampled whole-card memory; real-condition median forward 2.4559 seconds. Loading, media decoding, preparation and deployment overhead are excluded. No throughput or acceleration claim relative to SFT/FlashAttention is made.
