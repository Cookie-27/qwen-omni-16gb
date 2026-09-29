# Supported scope

The adapter is inference-only and pinned to Transformers 5.17.0. It routes modules whose class is Qwen2_5OmniAttention. Its public context manager temporarily changes the process-global SDPA registry and rejects nested or existing patches. It is not thread-safe for concurrent model execution.

Only batch-one candidate scoring is supported by infer.py. Training, gradient computation, speech generation, streaming, unrestricted generation, torch.compile, quantization and alternate model families have not been validated. Audio and visual encoders are unchanged. A 16GB card does not guarantee every input fits; input duration, frame count, resolution and allocator/backend selection matter.

The model checkpoint includes more than the marketing 3B language component: the loaded Thinker contains 4,703,464,448 parameters. Downloads and CPU RAM needs exceed GPU residency. Set HF_HOME and temporary directories to a drive with sufficient space if needed.

The provided example requires WAV audio and explicitly sampled frame files, not an MP4 decoder. All supplied audio is retained, and feature-mask length is checked. Supplied frames should be sampled at 1fps to match the documented prompt. This does not preserve every event between frames.

The CPU tests check the algorithm and routing behavior; they are not a substitute for whole-model GPU numerical checks. Stock attention should be preferred where a fused backend is already efficient. See the [PyTorch SDPA documentation](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html).

The evidence includes one small exposed MUSIC subset, oracle-derived answer candidates and no complete semantic label review. Do not report its counts as an official benchmark score or interpret its runtime success as a learning result.
