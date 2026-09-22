# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _load_validator():
    path = Path(__file__).resolve().parents[2] / "tools" / "validate_cosmos3_lidar_decoder.py"
    spec = importlib.util.spec_from_file_location("validate_cosmos3_lidar_decoder", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


validator = _load_validator()


def _config(*, apply_mask: bool = True):
    return {
        "latent_channels": 2,
        "spatial_compression": [2, 2],
        "apply_validity_mask": apply_mask,
        "network_config": {"resolution": [2, 6]},
        "range_projection": {
            "native_height": 2,
            "semantic_width": 4,
            "model_width": 6,
            "min_range_m": 5.0,
            "max_range_m": 105.0,
            "validity_threshold": 0.5,
        },
    }


def test_postprocess_keeps_probabilities_and_crops_metric_output():
    raw = torch.zeros(1, 3, 1, 2, 6)
    raw[:, 0] = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0, 0.0])
    raw[:, 1] = 0.5
    raw[:, 2] = torch.tensor([-1.0, -1.0, 1.0, 1.0, -1.0, -1.0])

    output = validator._postprocess(raw, _config())

    assert output["decoder_raw"].shape == (1, 3, 1, 2, 6)
    assert output["decoder_validity"].shape == (1, 1, 1, 2, 4)
    assert output["decoder_metric"].shape == (1, 3, 1, 2, 4)
    assert output["decoder_binary_validity"].dtype == torch.bool
    assert output["decoder_binary_validity"][0, 0, 0, 0].tolist() == [False, True, True, False]
    assert output["decoder_metric"][0, 0, 0, 0].tolist() == [0.0, 55.0, 105.0, 0.0]


def test_prepare_encoder_input_uses_circular_padding_and_metric_normalization():
    frames = torch.zeros(1, 3, 1, 2, 4)
    frames[:, 0] = torch.tensor([5.0, 55.0, 105.0, 200.0])
    frames[:, 1] = torch.tensor([0.0, 0.5, 1.0, 0.25])
    frames[:, 2] = 1

    normalized = validator._prepare_encoder_input(frames, _config())

    assert normalized.shape == (1, 3, 1, 2, 6)
    assert normalized[0, 0, 0, 0].tolist() == [-1.0, -1.0, 0.0, 1.0, -1.0, -1.0]
    assert normalized[0, 2, 0, 0].tolist() == [0.0, 1.0, 1.0, 1.0, 0.0, 1.0]


def test_reference_artifact_round_trip(tmp_path: Path):
    artifact = tmp_path / "reference.safetensors"
    tensors = {
        "latents": torch.randn(1, 2, 3, 1, 2),
        "decoder_binary_validity": torch.tensor([True, False]),
    }
    metadata = {"schema_version": validator.SCHEMA_VERSION, "producer": "imaginaire4"}

    validator._write_artifact(artifact, tensors, metadata)
    loaded, loaded_metadata = validator._read_artifact(artifact)

    assert loaded_metadata == metadata
    assert loaded.keys() == tensors.keys()
    for name in tensors:
        assert torch.equal(loaded[name], tensors[name])


def test_comparison_reports_tolerance_and_exact_mismatches():
    expected = torch.tensor([1.0, 2.0])
    close = validator._comparison(expected + 5e-5, expected, rtol=1e-4, atol=1e-4)
    exact = validator._comparison(expected + 5e-5, expected, rtol=0, atol=0, exact=True)

    assert close["passed"]
    assert close["max_abs_error"] == pytest.approx(5e-5, rel=2e-3)
    assert not exact["passed"]


def test_cpu_benchmark_checks_rng_and_repeated_outputs():
    value = torch.arange(4)
    output, measurement = validator._benchmark(
        lambda: {"value": value.clone()},
        device=torch.device("cpu"),
        warmups=1,
        runs=2,
    )

    assert torch.equal(output["value"], value)
    assert measurement.rng_preserved
    assert measurement.repeated_output_identical
    assert measurement.timed_calls == 2


def test_cache_reuse_requires_hits_without_new_specializations():
    first = {
        "block_masks": {"hits": 1, "misses": 2, "maxsize": 32, "currsize": 2},
        "compiled_runners": {"hits": 0, "misses": 3, "maxsize": 32, "currsize": 3},
    }
    reused = {
        "block_masks": {"hits": 5, "misses": 2, "maxsize": 32, "currsize": 2},
        "compiled_runners": {"hits": 4, "misses": 3, "maxsize": 32, "currsize": 3},
    }
    recompiled = {
        **reused,
        "compiled_runners": {"hits": 4, "misses": 4, "maxsize": 32, "currsize": 4},
    }

    assert validator._cache_reuse_check(first, reused)["passed"]
    assert not validator._cache_reuse_check(first, recompiled)["passed"]


def test_cli_keeps_environment_specific_inputs_on_reference_side(tmp_path: Path):
    common = ["--model", "model", "--artifact", str(tmp_path / "reference.safetensors")]
    reference = validator.parse_args(["--mode", "imaginaire4", *common, "--frames", "8", "--batch-size", "2"])
    assert reference.frames == 8 and reference.batch_size == 2

    with pytest.raises(SystemExit):
        validator.parse_args(["--mode", "vllm-omni", *common, "--latents", "inputs.safetensors"])
    with pytest.raises(SystemExit):
        validator.parse_args(["--mode", "vllm-omni", *common])

    candidate = validator.parse_args(["--mode", "vllm-omni", *common, "--report", str(tmp_path / "report.json")])
    assert candidate.report == tmp_path / "report.json"
