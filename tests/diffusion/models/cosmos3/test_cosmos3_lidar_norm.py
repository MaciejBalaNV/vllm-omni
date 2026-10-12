# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""LiDAR RMSNorm numerics and compatibility with original VAE checkpoints."""

import json

import pytest
import torch
from safetensors.torch import save_file

from vllm_omni.diffusion.layers.norm import RMSNorm as FrameworkRMSNorm
from vllm_omni.diffusion.models.cosmos3.lidar import Cosmos3LidarDecoder, Cosmos3LidarEncoder
from vllm_omni.diffusion.models.cosmos3.norm import RMSNorm

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("method", ["forward", "forward_cuda", "forward_hip", "forward_npu"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_lidar_rmsnorm_matches_original_fp32_formula(method, dtype):
    norm = RMSNorm(8)
    assert isinstance(norm, FrameworkRMSNorm)
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.1234567, 1.234567, 8))
    x = torch.linspace(-100, 100, 48).reshape(2, 3, 8).to(dtype)
    x_fp32 = x.float()
    expected = (x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + 1e-6) * norm.weight).to(x)
    actual = getattr(norm, method)(x)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.fixture(params=[Cosmos3LidarEncoder, Cosmos3LidarDecoder], ids=["encoder", "decoder"])
def original_norm_artifact(request, tmp_path):
    config = {
        "dtype": "float32",
        "sample_posterior": False,
        "apply_validity_mask": True,
        "fps": 10.0,
        "latent_channels": 2,
        "spatial_compression": [16, 16],
        "temporal_compression_factor": 1,
        "streaming_chunk_frames": 2,
        "streaming_context_frames": 3,
        "range_projection": {
            "native_height": 128,
            "semantic_width": 1800,
            "model_width": 1808,
            "model_width_transform": "circular_pad",
            "intensity_encoding": "unit",
            "min_range_m": 5.0,
            "max_range_m": 100.0,
        },
        "network_config": {
            "resolution": [128, 1808],
            "patch_size": [2, 2],
            "in_channels": 3,
            "out_channels": 3,
            "z_dim": 2,
            "base_channels": 4,
            "depths": [0, 0, 0, 1],
            "num_heads": [1, 1, 1, 1],
            "dilation": [1, 1, 1, 1],
            "temporal_downsample": [False, False, False],
        },
    }
    cls = request.param
    model = cls(config).float()
    model.latent_mean.fill_(0.1234567)
    model.latent_std.fill_(0.9876543)
    norm_keys = {}
    for name, module in model.named_modules():
        if isinstance(module, RMSNorm):
            with torch.no_grad():
                module.weight.copy_(torch.linspace(0.1234567, 1.234567, module.hidden_size))
            norm_keys[f"{name}.weight"] = f"{name}.scale"
    expected = model.state_dict()
    state = {norm_keys.get(name, name): value for name, value in expected.items()}
    # The shared artifact also contains the other component's unused weights.
    state[f"{cls._unused_prefixes[0]}unused"] = torch.ones(1)
    folder = tmp_path / "lidar_vae"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps(config))
    weights_path = folder / "diffusion_pytorch_model.safetensors"
    save_file(state, weights_path)
    return cls, tmp_path, config, expected, norm_keys, state, weights_path


def test_original_norm_checkpoint_loads_exactly(original_norm_artifact):
    cls, path, config, expected, norm_keys, _, _ = original_norm_artifact
    assert norm_keys
    # Sequential norms need the same remapping as attention/MLP block norms.
    sequential_key = "encoder.head.0.weight" if cls is Cosmos3LidarEncoder else "decoder.detokenizer.0.weight"
    assert sequential_key in norm_keys
    attention_scales = [name for name in expected if name.endswith(".scale")]
    assert attention_scales
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        loaded = cls.from_pretrained(str(path), config, torch.device("cpu"))
    finally:
        torch.set_default_dtype(previous)
    assert loaded.state_dict().keys() == expected.keys()
    for name, value in loaded.state_dict().items():
        assert value.dtype == torch.float32
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


@pytest.mark.parametrize("corruption", ["missing", "shape", "dtype", "duplicate", "unexpected"])
def test_original_norm_checkpoint_rejects_invalid_state(original_norm_artifact, corruption):
    cls, path, config, _, norm_keys, state, weights_path = original_norm_artifact
    weight_key, scale_key = next(iter(norm_keys.items()))
    error, message = ValueError, ""
    if corruption == "missing":
        del state[scale_key]
        error, message = RuntimeError, "Missing key"
    elif corruption == "shape":
        state[scale_key] = torch.ones(1)
        error, message = RuntimeError, "size mismatch"
    elif corruption == "dtype":
        state[scale_key] = state[scale_key].bfloat16()
        message = "must be FP32"
    elif corruption == "duplicate":
        state[weight_key] = state[scale_key].clone()
        message = "Duplicate LiDAR"
    else:
        state[f"{cls._component_name}.unexpected.scale"] = torch.ones(1)
        message = "Unexpected LiDAR"
    save_file(state, weights_path)
    with pytest.raises(error, match=message):
        cls.from_pretrained(str(path), config, torch.device("cpu"))
