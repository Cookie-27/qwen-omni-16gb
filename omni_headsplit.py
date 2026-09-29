"""Inference-only SDPA head partitioning for Qwen2.5-Omni's text attention.

Each query head keeps its entire sequence and its corresponding KV head.
This is a memory workaround, not a new attention algorithm or training backend.
"""
from collections import Counter
from contextlib import contextmanager
import torch
from transformers.integrations.sdpa_attention import sdpa_attention_forward as stock_sdpa

_active = False


class _GroupView:
    def __init__(self, module, groups):
        self.module = module
        self.num_key_value_groups = groups

    def __getattr__(self, name):
        return getattr(self.module, name)


def _select_heads(tensor, start, end, total):
    if tensor is None or tensor.ndim == 2:
        return tensor
    if tensor.ndim != 4:
        raise ValueError('Mask/bias must be 2D or [batch, 1/heads, query, key]')
    if tensor.shape[-3] == 1:
        return tensor
    if tensor.shape[-3] != total:
        raise ValueError('Unsupported head-specific mask/bias layout')
    return tensor[..., start:end, :, :]


def partitioned_sdpa(module, query, key, value, attention_mask, dropout=0.0,
                     scaling=None, is_causal=None, position_bias=None,
                     *, heads_per_call=2, **kwargs):
    if dropout != 0.0 or getattr(module, 'training', False) or torch.is_grad_enabled():
        raise ValueError('Use model.eval() and torch.inference_mode(); training is unsupported')
    if not isinstance(heads_per_call, int) or heads_per_call < 1:
        raise ValueError('heads_per_call must be a positive integer')
    if any(t.ndim != 4 for t in (query, key, value)):
        raise ValueError('Expected [batch, heads, sequence, channel] tensors')
    qheads, kvheads = query.shape[1], key.shape[1]
    if kvheads < 1 or qheads < 1 or qheads % kvheads or value.shape[1] != kvheads:
        raise ValueError('Invalid grouped-query head counts')
    groups = qheads // kvheads
    if groups != getattr(module, 'num_key_value_groups', 1):
        raise ValueError('Module and tensor KV groups disagree')
    outputs = []
    start = 0
    while start < qheads:
        kv_index = start // groups
        end = min(start + heads_per_call, (kv_index + 1) * groups)
        proxy = _GroupView(module, end - start)
        output, _ = stock_sdpa(
            proxy, query[:, start:end], key[:, kv_index:kv_index + 1],
            value[:, kv_index:kv_index + 1],
            _select_heads(attention_mask, start, end, qheads),
            dropout=dropout, scaling=scaling, is_causal=is_causal,
            position_bias=_select_heads(position_bias, start, end, qheads), **kwargs)
        outputs.append(output)
        start = end
    return torch.cat(outputs, dim=2), None


@contextmanager
def head_partitioned_sdpa(heads_per_call=2):
    """Temporarily route Qwen2_5OmniAttention only; restore the original registry.

    Transformers' attention registry is process-global. Use one model at a time,
    without concurrent model calls or nested contexts in the same process.
    """
    global _active
    from transformers import __version__
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    if __version__ != '5.17.0':
        raise RuntimeError('This adapter is validated with transformers==5.17.0 only')
    if _active or ALL_ATTENTION_FUNCTIONS['sdpa'] is not stock_sdpa:
        raise RuntimeError('SDPA is already patched; nested/concurrent patching is unsupported')
    if not isinstance(heads_per_call, int) or heads_per_call < 1:
        raise ValueError('heads_per_call must be a positive integer')
    stats = Counter()
    previous = ALL_ATTENTION_FUNCTIONS['sdpa']

    def routed(module, query, key, value, attention_mask, dropout=0.0,
               scaling=None, is_causal=None, position_bias=None, **kwargs):
        if type(module).__name__ != 'Qwen2_5OmniAttention':
            stats['other_attention_calls'] += 1
            return previous(module, query, key, value, attention_mask, dropout=dropout,
                            scaling=scaling, is_causal=is_causal, position_bias=position_bias, **kwargs)
        stats['text_partitioned_calls'] += 1
        stats['max_query_length'] = max(stats['max_query_length'], query.shape[-2])
        return partitioned_sdpa(module, query, key, value, attention_mask,
                                dropout=dropout, scaling=scaling, is_causal=is_causal,
                                position_bias=position_bias, heads_per_call=heads_per_call, **kwargs)

    _active = True
    ALL_ATTENTION_FUNCTIONS.register('sdpa', routed)
    try:
        yield stats
    finally:
        ALL_ATTENTION_FUNCTIONS.register('sdpa', previous)
        _active = False
