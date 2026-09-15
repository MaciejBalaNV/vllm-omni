# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Input and comparison contracts for the standalone encoder benchmark."""

from types import SimpleNamespace

import pytest
import torch

from benchmarks.diffusion.bench_wan_vae_encode import differences, make_pixels, parse_args

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_benchmark_reference_always_runs_first(monkeypatch):
    monkeypatch.setattr("sys.argv", ["bench", "--fast-path", "channels_last,lossless,channels_last"])
    args = parse_args()
    assert args.levels == ["off", "channels_last", "lossless"]
    assert args.warmup == 3 and args.iters == 10


def test_seeded_pixels_and_real_input(tmp_path):
    args = SimpleNamespace(input=None, size="32x16", frames=5, seed=123)
    a, b = make_pixels(args), make_pixels(args)
    assert torch.equal(a, b) and a.shape == (1, 3, 5, 16, 32)
    args.input = tmp_path / "pixels.pt"
    torch.save(a.to(torch.bfloat16), args.input)
    assert torch.equal(make_pixels(args), a.to(torch.bfloat16))
    torch.save(torch.full_like(a, float("nan")), args.input)
    with pytest.raises(ValueError, match="finite"):
        make_pixels(args)
    torch.save(torch.full_like(a, 2), args.input)
    with pytest.raises(ValueError, match="normalized"):
        make_pixels(args)


@pytest.mark.parametrize("size,frames", [("32x16", 6), ("31x16", 5)])
def test_invalid_input_schedule(size, frames):
    with pytest.raises(ValueError, match="1\\+4k"):
        make_pixels(SimpleNamespace(input=None, size=size, frames=frames, seed=0))


def test_metrics_distinguish_signed_zero_and_measure_relative_error():
    assert differences(torch.tensor([-0.0]), torch.tensor([0.0]))["bitwise_equal"] is False
    result = differences(torch.tensor([1.01]), torch.tensor([1.0]))
    assert result["normalized_rmse"] == pytest.approx(0.01, abs=1e-6)
    assert result["max_abs_diff"] == pytest.approx(0.01, abs=1e-6)
    assert result["bitwise_equal"] is False
