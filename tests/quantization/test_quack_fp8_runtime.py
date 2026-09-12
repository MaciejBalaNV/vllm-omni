# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for Quack runtime scales and inference-compatible warmup."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.quantization import quack_fp8

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_non_daemon_keeps_default_compilation(monkeypatch):
    import_module = Mock(side_effect=AssertionError("must not import or patch the pool"))
    monkeypatch.setattr(quack_fp8, "current_process", lambda: SimpleNamespace(daemon=False))
    monkeypatch.setattr(quack_fp8, "import_module", import_module)
    quack_fp8._configure_quack_compilation()
    import_module.assert_not_called()


@pytest.mark.parametrize("missing_module", ["quack.cache", "quack.cache.async_compile"])
def test_legacy_quack_needs_no_compile_pool_adapter(monkeypatch, missing_module):
    monkeypatch.setattr(quack_fp8, "current_process", lambda: SimpleNamespace(daemon=True))
    monkeypatch.setattr(quack_fp8, "import_module", Mock(side_effect=ModuleNotFoundError(name=missing_module)))
    quack_fp8._configure_quack_compilation()


def test_broken_async_compile_dependency_is_not_treated_as_legacy_quack(monkeypatch):
    monkeypatch.setattr(quack_fp8, "current_process", lambda: SimpleNamespace(daemon=True))
    monkeypatch.setattr(quack_fp8, "import_module", Mock(side_effect=ModuleNotFoundError(name="cutlass")))
    with pytest.raises(ModuleNotFoundError):
        quack_fp8._configure_quack_compilation()


@pytest.mark.parametrize("fail", [False, True])
def test_daemon_compilation_suppresses_pool_without_constructing_executor(monkeypatch, fail):
    active_pool = object()
    pool = SimpleNamespace(
        pool_scope=Mock(spec=[], side_effect=AssertionError("must not construct a compiler executor")),
        get_active_pool=lambda: active_pool,
    )

    @contextmanager
    def suppress_pool():
        nonlocal active_pool
        previous, active_pool = active_pool, None
        try:
            yield
        finally:
            active_pool = previous

    pool.suppress_pool = suppress_pool
    original_pool = active_pool
    monkeypatch.setattr(quack_fp8, "current_process", lambda: SimpleNamespace(daemon=True))
    monkeypatch.setattr(quack_fp8, "import_module", lambda name: pool)
    quack_fp8._configure_quack_compilation()
    scope = pool.pool_scope
    quack_fp8._configure_quack_compilation()
    assert pool.pool_scope is scope

    def compile_candidates():
        with pool.pool_scope():
            assert pool.get_active_pool() is None
            with pool.pool_scope():
                assert pool.get_active_pool() is None
            assert pool.get_active_pool() is None
            if fail:
                raise RuntimeError("compile failed")

    if fail:
        with pytest.raises(RuntimeError, match="compile failed"):
            compile_candidates()
    else:
        compile_candidates()
    assert pool.get_active_pool() is original_pool


def test_daemon_autotuning_benchmarks_candidates_and_reuses_disk_cache(monkeypatch, tmp_path):
    # Exercise Quack's real tuning loop with a CPU benchmark stand-in. Optional
    # because the quack extra is not installed in every CPU test environment.
    autotuner = pytest.importorskip("quack.autotuner")
    async_compile = pytest.importorskip("quack.cache.async_compile")
    monkeypatch.setenv("QUACK_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("QUACK_FORCE_CACHE_UPDATE", raising=False)
    monkeypatch.setattr(quack_fp8, "current_process", lambda: SimpleNamespace(daemon=True))
    monkeypatch.setattr(autotuner, "_gpu_warmup", lambda: None)
    monkeypatch.setattr(async_compile, "_make_executor", Mock(side_effect=AssertionError("compiler child requested")))
    # Arrange for monkeypatch to restore the process-local adapter after this test.
    monkeypatch.setattr(async_compile, "pool_scope", async_compile.pool_scope)
    quack_fp8._configure_quack_compilation()
    candidates = []

    def kernel(x, *, tile):
        assert async_compile.get_active_pool() is None
        candidates.append(tile)
        return tile

    def benchmark(fn, quantiles):
        fn()
        tile = candidates[-1]
        return [{16: 3.0, 32: 1.0, 64: 2.0}[tile]] * len(quantiles)

    def make_tuner():
        return autotuner.Autotuner(
            kernel,
            key=[],
            configs=[autotuner.AutotuneConfig(tile=tile) for tile in (16, 32, 64)],
            do_bench=benchmark,
            cache_results=True,
        )

    x = torch.ones(4)
    tuner = make_tuner()
    assert tuner(x) == 32
    assert candidates == [16, 32, 64, 32]
    candidates.clear()
    assert tuner(x) == 32
    assert candidates == [32]
    assert list(tmp_path.rglob("*.autotune.json"))
    candidates.clear()
    assert make_tuner()(x) == 32  # A fresh tuner must use the persisted winner.
    assert candidates == [32]
    async_compile._make_executor.assert_not_called()


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


@pytest.mark.parametrize("with_bias", [False, True])
def test_gemm_dispatch_keeps_runtime_scales(monkeypatch, with_bias):
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
    assert calls == [True, True]


def test_warmup_matches_inference_weight_layout(monkeypatch):
    calls = []

    def gemm(a, b, *, out, bias, alpha, tuned):
        assert torch.is_inference_mode_enabled()
        assert a.dtype == b.dtype == torch.float8_e4m3fn
        assert a.is_contiguous()
        assert b.stride() == (1, b.shape[0])
        assert bias is None
        assert tuned is True
        calls.append((a.shape[0], a.shape[1], b.shape[1]))
        out.zero_()

    monkeypatch.setattr(quack_fp8, "_gemm_interface", SimpleNamespace(gemm=gemm))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    shapes = [(4, 8, 6), (2, 16, 8)]
    quack_fp8.warmup_quack_fp8(shapes, device="cpu")
    assert calls == shapes
