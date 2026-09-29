# Upstream options and integration boundaries

This example targets one constrained candidate-scoring workload. Choose an upstream path based on the behavior you need:

| Need | Starting point | Relationship to this repository |
|---|---|---|
| Qwen Omni model capabilities, chat or speech output | [Official Qwen2.5-Omni repository](https://github.com/QwenLM/Qwen2.5-Omni) | This project uses the Thinker for one last-token forward; it does not implement the complete chat/speech interface. |
| A supported efficient attention backend | [Official FlashAttention-2 usage](https://github.com/QwenLM/Qwen2.5-Omni#flash-attention-2-to-speed-up-generation) | Check upstream backend support for your environment first. No matched performance comparison with FlashAttention is provided here. |
| A reproducible text-attention fallback on the pinned runtime | [omni_headsplit.py](../omni_headsplit.py) | Groups heads while retaining complete sequences; preserves audio and vision attention. |
| Report a possible Transformers bug | [Upstream contributing guide](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md) | First isolate a minimal upstream reproducer. A problem in this adapter should be reported here. |

The adapter temporarily changes a process-global Transformers attention registry, so it belongs in a dedicated inference process. It is not a drop-in server plugin, a training implementation or a thread-safe library. Do not copy the patch into another model and assume equivalence: head layout, masks, numerical behavior and runtime versions require new checks.

For an upstream contribution, a self-contained reproducer and numerical comparison are more useful than a repository link. Claims need the input length, frame/audio preparation, attention backend, dtype, device and measurement boundary. Keep upstream attribution and third-party license terms when reusing code or data.
