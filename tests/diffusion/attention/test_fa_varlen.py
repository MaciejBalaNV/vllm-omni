# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Validate pinned FA versions without requiring CUDA kernels."""

import sys
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.attention.backends.utils import fa

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.mark.parametrize("version", [None, 0, -1, 1, 5, 2, 3, 4])
@pytest.mark.cpu
def test_resolved_varlen_version_is_validated_without_rediscovery(monkeypatch, version):
    query = torch.zeros(2, 1, 4)
    output = torch.empty_like(query)
    lse = torch.zeros(1, 2)
    kernel = Mock(return_value=(output, lse))
    wrapper = ModuleType("vllm.vllm_flash_attn")
    wrapper.flash_attn_varlen_func = kernel
    monkeypatch.setitem(sys.modules, wrapper.__name__, wrapper)
    resolver = Mock(side_effect=AssertionError("Pinned FA versions must not be rediscovered"))
    monkeypatch.setattr(fa, "resolve_vllm_flash_attn_version", resolver)
    offsets = torch.tensor([0, 2], dtype=torch.int32)
    kwargs = dict(
        cu_seqlens_q=offsets,
        cu_seqlens_k=offsets,
        max_seqlen_q=2,
        max_seqlen_k=2,
        fa_version=version,
        fa_version_is_resolved=True,
        out=output,
    )
    if version in (2, 3, 4):
        out, actual_lse = fa.vllm_flash_attn_varlen_with_lse(query, query, query, **kwargs)
        assert out is output and actual_lse is lse
        kernel.assert_called_once()
        assert kernel.call_args.kwargs["fa_version"] == version
        assert kernel.call_args.kwargs["out"] is output
    else:
        with pytest.raises(ValueError, match="resolved FlashAttention version"):
            fa.vllm_flash_attn_varlen_with_lse(query, query, query, **kwargs)
        kernel.assert_not_called()
    resolver.assert_not_called()


@pytest.mark.cpu
def test_dense_adapter_uses_bundled_fa4_and_preserves_output_buffer(monkeypatch):
    query = torch.zeros(3, 5, 8, 128)
    key = torch.zeros(3, 7, 2, 128)
    output = torch.empty_like(query)
    lse = torch.zeros(3, 8, 5)
    kernel = Mock(return_value=(output, lse, None, None))
    interface = ModuleType("vllm.vllm_flash_attn.cute.interface")
    interface._flash_attn_fwd = kernel
    monkeypatch.setitem(sys.modules, interface.__name__, interface)
    monkeypatch.setitem(sys.modules, "flash_attn.cute", None)
    actual, actual_lse = fa.vllm_flash_attn4_dense_with_lse(query, key, key, out=output)
    assert actual is output and actual_lse is lse
    kernel.assert_called_once()
    assert kernel.call_args.args[0] is query
    assert kernel.call_args.kwargs == dict(softmax_scale=None, causal=False, num_splits=0, return_lse=True, out=output)


@pytest.mark.cpu
def test_dense_adapter_rejects_packed_inputs():
    query = torch.zeros(6, 8, 128)
    with pytest.raises(ValueError, match="Dense FlashAttention requires"):
        fa.vllm_flash_attn4_dense_with_lse(query, query, query)


@pytest.mark.cpu
def test_dense_adapter_rejects_wrong_lse_axes(monkeypatch):
    query = torch.zeros(3, 5, 8, 128)
    interface = ModuleType("vllm.vllm_flash_attn.cute.interface")
    interface._flash_attn_fwd = Mock(return_value=(query, torch.zeros(3, 5, 8), None, None))
    monkeypatch.setitem(sys.modules, interface.__name__, interface)
    with pytest.raises(ValueError, match="Expected dense FlashAttention LSE"):
        fa.vllm_flash_attn4_dense_with_lse(query, query, query)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and bundled FA4")
@hardware_test(res={"cuda": "B200"}, num_cards=1)
@pytest.mark.parametrize("query_heads,kv_heads", [(8, 2), (16, 4)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("query_length,key_length", [(257, 193), (128, 128)])
def test_dense_fa4_matches_varlen_and_batched_oracle(query_heads, kv_heads, dtype, query_length, key_length):
    from vllm.vllm_flash_attn.flash_attn_interface import is_fa_version_supported

    if not is_fa_version_supported(4):
        pytest.skip("Bundled FA4 is unavailable on this device")
    torch.manual_seed(42)
    batch, dim = 3, 128
    q = torch.randn(batch, query_length, query_heads, dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, key_length, kv_heads, dim, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    offsets_q = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * query_length
    offsets_k = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * key_length
    output = torch.empty_like(q)
    with torch.inference_mode():
        actual, lse = fa.vllm_flash_attn4_dense_with_lse(q, k, v, out=output)
        packed, packed_lse = fa.vllm_flash_attn_varlen_with_lse(
            q.flatten(0, 1),
            k.flatten(0, 1),
            v.flatten(0, 1),
            cu_seqlens_q=offsets_q,
            cu_seqlens_k=offsets_k,
            max_seqlen_q=query_length,
            max_seqlen_k=key_length,
            fa_version=4,
            fa_version_is_resolved=True,
        )
        repeated_k, repeated_v = [x.double().repeat_interleave(query_heads // kv_heads, 2) for x in (k, v)]
        scores = torch.einsum("bqhd,bkhd->bhqk", q.double(), repeated_k) / dim**0.5
        expected = torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), repeated_v)
    tolerance = 1e-2 if dtype == torch.bfloat16 else 1e-3
    assert actual.data_ptr() == output.data_ptr()
    torch.testing.assert_close(actual.double(), expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(actual, packed.view_as(actual), atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(lse.double(), scores.logsumexp(-1), atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(
        lse, packed_lse.view(query_heads, batch, query_length).transpose(0, 1), atol=1e-3, rtol=1e-3
    )
