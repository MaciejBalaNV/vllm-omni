# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Exactness and eligibility coverage for Cosmos3's eager Q/K RoPE fusion."""

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm.triton_utils import HAS_TRITON
from vllm_omni.diffusion.models.cosmos3 import rope
from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import _apply_rotary_pos_emb
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def _reference(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Preserve the original eager multiply, round, multiply, round, add order."""
    first, second = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-second, first), dim=-1) * sin


@pytest.mark.cpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_cpu_fallback_preserves_outputs_and_gradients(dtype: torch.dtype) -> None:
    """The production dispatcher retains eager behavior, including autograd."""
    torch.manual_seed(17)
    q = torch.randn(2, 5, 7, 8, dtype=dtype, requires_grad=True)
    k = torch.randn(2, 5, 3, 8, dtype=dtype, requires_grad=True)
    cos = torch.randn(2, 5, 1, 8, dtype=dtype, requires_grad=True)
    sin = torch.randn_like(cos, requires_grad=True)
    inputs = (q, k, cos, sin)
    expected = (_reference(q, cos, sin), _reference(k, cos, sin))

    assert rope.fused_rotary_pos_emb(*inputs) is None
    actual = _apply_rotary_pos_emb(*inputs)
    for result, reference in zip(actual, expected):
        assert torch.equal(result, reference)
    actual_grads = torch.autograd.grad(sum(t.float().sum() for t in actual), inputs)
    expected_grads = torch.autograd.grad(sum(t.float().sum() for t in expected), inputs)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        assert torch.equal(actual_grad, expected_grad)


@pytest.mark.cpu
@pytest.mark.parametrize("has_triton, nvidia_platform", [(True, False), (False, True)])
def test_ineligible_platform_never_reaches_kernel(
    monkeypatch: pytest.MonkeyPatch, has_triton: bool, nvidia_platform: bool
) -> None:
    """ROCm's CUDA tensor alias and missing Triton must both retain eager RoPE."""

    class CudaLikeTensor(torch.Tensor):
        # HIP tensors report is_cuda=True too; no accelerator is needed to
        # verify that platform eligibility takes precedence over that alias.
        @property
        def is_cuda(self) -> bool:
            return True

    def unexpected_table_lookup(*args):
        pytest.fail("Ineligible platform reached fused input preparation")

    monkeypatch.setattr(rope, "HAS_TRITON", has_triton)
    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: nvidia_platform)
    monkeypatch.setattr(rope, "_broadcast_table", unexpected_table_lookup)
    q = torch.randn(1, 3, 2, 8, dtype=torch.bfloat16).as_subclass(CudaLikeTensor)
    cos = torch.randn(1, 3, 1, 8, dtype=q.dtype)
    assert rope.fused_rotary_pos_emb(q, q, cos, cos) is None


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="NVIDIA CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "batch, sequence, q_heads, k_heads, head_dim",
    [(1, 1, 1, 1, 2), (2, 17, 7, 3, 64), (1, 257, 16, 8, 128), (2, 33, 8, 8, 256)],
)
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("table_layout", ["shared_batch", "per_batch", "shared_sequence"])
def test_fused_rope_is_bit_exact(
    dtype: torch.dtype,
    batch: int,
    sequence: int,
    q_heads: int,
    k_heads: int,
    head_dim: int,
    strided: bool,
    table_layout: str,
) -> None:
    """Exercise the real kernel and call site with GQA, broadcasts and strides."""
    torch.manual_seed(17)
    step = 2 if strided else 1
    q = torch.randn(batch, sequence * step, q_heads, head_dim, device="cuda", dtype=dtype)[:, ::step]
    k = torch.randn(batch, sequence * step, k_heads, head_dim, device="cuda", dtype=dtype)[:, ::step]
    table_batch = 1 if table_layout == "shared_batch" else batch
    table_sequence = 1 if table_layout == "shared_sequence" else sequence
    # Use independent full-width columns so an incorrect frequency mapping
    # cannot pass merely because adjacent columns happened to be repeated.
    cos = torch.randn(table_batch, table_sequence * step, 1, head_dim, device="cuda", dtype=dtype)[:, ::step]
    sin = torch.randn_like(cos)

    with torch.inference_mode():
        expected = (_reference(q, cos, sin), _reference(k, cos, sin))
        fused = rope.fused_rotary_pos_emb(q, k, cos, sin)
        assert fused is not None  # An eager fallback must not count as a kernel pass.
        actual = _apply_rotary_pos_emb(q, k, cos, sin)
    for result, dispatched, reference in zip(fused, actual, expected):
        assert torch.equal(result, reference)
        assert torch.equal(dispatched, reference)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="NVIDIA CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
@pytest.mark.parametrize(
    "case",
    [
        "float32",
        "non_power_of_two",
        "strided_head_dim",
        "strided_table_dim",
        "mixed_table_dtype",
        "nonbroadcast_table",
        "empty",
        "autograd_q",
        "autograd_k",
        "autograd_cos",
        "autograd_sin",
        "compile",
    ],
)
def test_unsupported_inputs_use_fallback(monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    """Unsupported inputs must return before launch; valid eager cases stay exact."""
    torch.manual_seed(17)
    dtype = torch.float32 if case == "float32" else torch.bfloat16
    head_dim = 6 if case == "non_power_of_two" else 128
    sequence = 0 if case == "empty" else 5
    q = torch.randn(2, sequence, 7, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(2, sequence, 3, head_dim, device="cuda", dtype=dtype)
    cos = torch.randn(2, sequence, 1, head_dim, device="cuda", dtype=dtype)
    sin = torch.randn_like(cos)
    if case == "strided_head_dim":
        q = torch.randn(2, sequence, 7, head_dim * 2, device="cuda", dtype=dtype)[..., ::2]
    elif case == "strided_table_dim":
        cos = torch.randn(2, sequence, 1, head_dim * 2, device="cuda", dtype=dtype)[..., ::2]
    elif case == "mixed_table_dtype":
        cos, sin = cos.float(), sin.float()
    elif case == "nonbroadcast_table":
        cos = torch.randn(3, sequence, 1, head_dim, device="cuda", dtype=dtype)
    elif case.startswith("autograd_"):
        {"autograd_q": q, "autograd_k": k, "autograd_cos": cos, "autograd_sin": sin}[case].requires_grad_()
    elif case == "compile":
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    class UnexpectedKernel:
        def __getitem__(self, grid):
            pytest.fail("Unsupported inputs reached the fused kernel")

    monkeypatch.setattr(rope, "_rope_qk_kernel", UnexpectedKernel())
    with torch.enable_grad():
        assert rope.fused_rotary_pos_emb(q, k, cos, sin) is None
        if case != "nonbroadcast_table":
            actual = _apply_rotary_pos_emb(q, k, cos, sin)
            expected = (_reference(q, cos, sin), _reference(k, cos, sin))
            for result, reference in zip(actual, expected):
                assert torch.equal(result, reference)
