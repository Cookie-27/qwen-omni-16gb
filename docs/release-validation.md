# Release-time validation

On 2026-09-29 the portable public entry points passed:

- Standard-library verification of the historical evidence and eight stored complete-vocabulary vectors.
- 32 small CPU operator cases in FP32/BF16 against stock SDPA and independent FP64 attention, plus rejection and context-restoration tests.
- The public media preparation function rebuilt the historical 5,521-token input with every tensor exactly equal, including complete audio and sampled frames.
- A real RTX 5080 run of `infer.py --compare-stock` on a 428-token input: complete-vocabulary output was exactly equal, with zero candidate-logit and probability difference.
- A real `infer.py` run on the 5,521-token input: success, 10.201777935 GiB peak allocated and 10.40625 GiB peak reserved. Candidate logits and probabilities matched the historical longest-input record exactly.

The short stock run precedes the headsplit run and is not a fair timing benchmark. No long-input stock equivalence is claimed because the historical stock run OOMed. No generation, training or complete benchmark rerun was performed for this release. Machine-readable values and code hashes are in release-validation.json.
