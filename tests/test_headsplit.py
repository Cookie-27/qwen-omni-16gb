import unittest
from types import SimpleNamespace
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from omni_headsplit import partitioned_sdpa, stock_sdpa, head_partitioned_sdpa


class HeadSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_attention_against_stock_and_independent_double(self):
        torch.manual_seed(77)
        for dtype in (torch.float32, torch.bfloat16):
            for heads, kvheads in ((8, 2), (4, 4)):
                for qn, kn in ((17, 17), (13, 19)):
                    for mode in ('causal', 'noncausal', 'head_bool', 'head_float_bias'):
                        with self.subTest(dtype=dtype, heads=heads, qn=qn, mode=mode):
                            module = SimpleNamespace(num_key_value_groups=heads//kvheads, is_causal=mode == 'causal', training=False)
                            q = torch.randn(1, heads, qn, 32, dtype=dtype)
                            k = torch.randn(1, kvheads, kn, 32, dtype=dtype)
                            v = torch.randn_like(k)
                            mask = bias = None
                            if mode.startswith('head_'):
                                mask = torch.rand(1, heads, qn, kn) > .25
                                mask[..., 0] = True
                                if mode == 'head_float_bias':
                                    mask = torch.where(mask, 0., torch.finfo(dtype).min).to(dtype)
                                    bias = torch.randn(1, heads, qn, kn, dtype=dtype) * .03
                            scale = .8 / 32**.5
                            with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
                                a = stock_sdpa(module, q, k, v, mask, scaling=scale, position_bias=bias)[0]
                                b = partitioned_sdpa(module, q, k, v, mask, scaling=scale, position_bias=bias, heads_per_call=3)[0]
                            kk, vv = (k[:, :, :qn], v[:, :, :qn]) if mode == 'causal' else (k, v)
                            scores = q.double() @ kk.double().repeat_interleave(heads//kvheads, dim=1).transpose(-2, -1) * scale
                            if mode == 'causal':
                                scores = scores.masked_fill(torch.arange(kk.shape[2])[None, :] > torch.arange(qn)[:, None], float('-inf'))
                            if mask is not None:
                                scores = scores.masked_fill(~mask, float('-inf')) if mask.dtype == torch.bool else scores + mask.double()
                            if bias is not None:
                                scores = scores + bias.double()
                            reference = (scores.softmax(-1) @ vv.double().repeat_interleave(heads//kvheads, dim=1)).transpose(1, 2)
                            atol, rtol = (2e-6, 2e-5) if dtype == torch.float32 else (.002, .016)
                            torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
                            torch.testing.assert_close(b.double(), reference, atol=atol, rtol=rtol)

    def test_training_and_unsupported_masks_rejected(self):
        q = torch.zeros(1, 4, 2, 8)
        k = v = torch.zeros(1, 1, 2, 8)
        module = SimpleNamespace(num_key_value_groups=4, training=False)
        with self.assertRaises(ValueError):
            partitioned_sdpa(module, q, k, v, None)
        with torch.inference_mode():
            with self.assertRaises(ValueError):
                partitioned_sdpa(module, q, k, v, None, dropout=.1)
            with self.assertRaises(ValueError):
                partitioned_sdpa(module, q, k, v, torch.ones(4, 2, 2, dtype=torch.bool))

    def test_registry_restored_after_error_and_nested_context_rejected(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        old = ALL_ATTENTION_FUNCTIONS['sdpa']
        with self.assertRaisesRegex(RuntimeError, 'deliberate'):
            with head_partitioned_sdpa():
                with self.assertRaises(RuntimeError):
                    with head_partitioned_sdpa():
                        pass
                raise RuntimeError('deliberate')
        self.assertIs(ALL_ATTENTION_FUNCTIONS['sdpa'], old)


if __name__ == '__main__':
    unittest.main()
