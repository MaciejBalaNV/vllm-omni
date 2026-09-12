# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for Quack dispatch settings and inference-compatible warmup."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.quantization import quack_fp8

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_scale_validation_does_not_recompile_for_each_layer(monkeypatch):
    monkeypatch.setattr(quack_fp8, "_valid_scale_ptrs", set())
    torch._dynamo.reset()
    compiled_graphs = []

    def backend(graph, example_inputs):
        compiled_graphs.append(graph)
        return graph.forward

    def forward(x, scale_a, scale_b):
        if quack_fp8._scales_valid(scale_a, scale_b):
            return x * (scale_a * scale_b)
        return x

    try:
        compiled = torch.compile(forward, backend=backend)
        # Keep all pairs alive, as separate model layers do. Both pointer
        # addresses and scale values differ; tensor metadata stays identical.
        with torch.inference_mode():
            pairs = [(torch.tensor([float(i + 1)]), torch.tensor([0.5])) for i in range(5)]
            x = torch.ones(4)
            for _ in range(2):
                for scale_a, scale_b in pairs:
                    torch.testing.assert_close(compiled(x, scale_a, scale_b), x * scale_a * scale_b)
        assert len(compiled_graphs) == 1
    finally:
        torch._dynamo.reset()


def test_unpopulated_scales_are_rechecked_after_loading(monkeypatch):
    monkeypatch.setattr(quack_fp8, "_valid_scale_ptrs", set())
    scale_a = torch.tensor([torch.finfo(torch.float32).min])
    scale_b = torch.tensor([0.5])
    assert not quack_fp8._scales_valid(scale_a, scale_b)
    scale_a.fill_(0.25)
    assert quack_fp8._scales_valid(scale_a, scale_b)


@pytest.mark.parametrize("autotune, expected", [(None, True), ("0", False), ("false", False), ("1", True)])
@pytest.mark.parametrize("with_bias", [False, True])
def test_gemm_dispatch_keeps_runtime_scales_and_respects_autotuning(monkeypatch, autotune, expected, with_bias):
    monkeypatch.delenv("VLLM_OMNI_QUACK_FP8_AUTOTUNE", raising=False)
    if autotune is not None:
        monkeypatch.setenv("VLLM_OMNI_QUACK_FP8_AUTOTUNE", autotune)
    calls = []

    def gemm(a, b, *, out, bias, alpha, tuned):
        assert isinstance(alpha, torch.Tensor)
        assert alpha.dtype == torch.float32
        assert alpha.shape == (1,)
        result = (a.float() @ b.float()) * alpha
        if bias is not None:
            result += bias
        out.copy_(result)
        calls.append(tuned)

    monkeypatch.setattr(quack_fp8, "_gemm_interface", SimpleNamespace(gemm=gemm))
    a = torch.arange(32, dtype=torch.float32).reshape(4, 8).to(torch.float8_e4m3fn)
    b = torch.ones(6, 8, dtype=torch.float8_e4m3fn).t()
    bias = torch.arange(6, dtype=torch.float32) if with_bias else None
    # Different calibrated scales must reach the kernel as runtime inputs.
    for scale in (0.25, 0.5):
        scale_a = torch.tensor([scale])
        scale_b = torch.tensor([0.5])
        out = quack_fp8.quack_scaled_fp8_mm(a, b, scale_a, scale_b, torch.float32, bias)
        reference = (a.float() @ b.float()) * scale * 0.5
        if bias is not None:
            reference += bias
        torch.testing.assert_close(out, reference)
    assert calls == [expected, expected]


def test_warmup_matches_inference_weight_layout(monkeypatch):
    calls = []

    def gemm(a, b, *, out, bias, alpha, tuned):
        assert torch.is_inference_mode_enabled()
        assert a.dtype == b.dtype == torch.float8_e4m3fn
        assert a.is_contiguous()
        assert b.stride() == (1, b.shape[0])
        assert bias is None
        assert tuned is False
        calls.append((a.shape[0], a.shape[1], b.shape[1]))
        out.zero_()

    monkeypatch.setattr(quack_fp8, "_gemm_interface", SimpleNamespace(gemm=gemm))
    monkeypatch.setenv("VLLM_OMNI_QUACK_FP8_AUTOTUNE", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    shapes = [(4, 8, 6), (2, 16, 8)]
    quack_fp8.warmup_quack_fp8(shapes, device="cpu")
    assert calls == shapes
