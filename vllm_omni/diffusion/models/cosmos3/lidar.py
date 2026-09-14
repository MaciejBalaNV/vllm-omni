# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V1.2 LiDAR deployment encoder. Deliberately has no decode method."""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn


def validate_lidar_config(config: dict[str, Any]) -> None:
    required = {
        "version",
        "fps",
        "latent_channels",
        "temporal_compression_factor",
        "spatial_compression",
        "network_config",
        "range_projection",
        "streaming_chunk_frames",
        "streaming_context_frames",
    }
    if missing := required - config.keys():
        raise ValueError(f"Incomplete joint artifact: missing LiDAR metadata {sorted(missing)}.")
    if config["version"] != "1.2":
        raise ValueError("Only the V1.2 LiDAR tokenizer is supported.")
    projection = config["range_projection"]
    expected = {
        "semantic_width": 1800,
        "model_width": 1808,
        "native_height": 128,
        "model_width_transform": "circular_pad",
        "intensity_encoding": "unit",
    }
    if any(projection.get(key) != value for key, value in expected.items()):
        raise ValueError("V1.2 requires the 128x1800 metric/unit-intensity grid with circular padding to 1808.")
    minimum, maximum = projection.get("min_range_m"), projection.get("max_range_m")
    if not all(isinstance(v, int | float) and math.isfinite(v) for v in (minimum, maximum)) or maximum <= minimum:
        raise ValueError("LiDAR range normalization requires finite min_range_m < max_range_m.")
    network = config["network_config"]
    if (
        network.get("resolution") != [128, 1808]
        or network.get("patch_size") != [2, 2]
        or len(network.get("depths", [])) != 4
        or config["spatial_compression"] != [16, 16]
        or config["temporal_compression_factor"] != 1
        or network.get("z_dim") != config["latent_channels"]
        or network.get("in_channels") != 3
        or any(network.get("temporal_downsample", [True]))
    ):
        raise ValueError("LiDAR architecture and V1.2 compression metadata disagree.")
    if not math.isfinite(config["fps"]) or config["fps"] <= 0:
        raise ValueError("LiDAR FPS must be finite and positive.")
    chunk, context = config["streaming_chunk_frames"], config["streaming_context_frames"]
    if type(chunk) is not int or chunk < 1 or (context is not None and (type(context) is not int or context < chunk)):
        raise ValueError("LiDAR streaming context must be at least the positive chunk length.")


def prepare_lidar_encoder_input(frames: torch.Tensor, projection: dict[str, Any]) -> torch.Tensor:
    """Reference physical normalization after symmetric circular width padding."""
    frames = frames.float().unsqueeze(0)
    padding = projection["model_width"] - projection["semantic_width"]
    if padding < 0 or padding % 2:
        raise ValueError("LiDAR model width must allow symmetric circular padding of the semantic width.")
    half_padding = padding // 2
    if half_padding:
        frames = torch.cat((frames[..., -half_padding:], frames, frames[..., :half_padding]), dim=-1)
    minimum, maximum = projection["min_range_m"], projection["max_range_m"]
    ranges, intensities, validity = frames.split(1, dim=1)
    valid = (validity >= 0.5) & (ranges >= minimum) & (ranges <= maximum)
    ranges = (ranges.clamp(minimum, maximum) - minimum) / (maximum - minimum) * 2 - 1
    intensities = intensities.clamp(0, 1) * 2 - 1
    return torch.cat((ranges.masked_fill(~valid, -1), intensities.masked_fill(~valid, -1), valid.float()), dim=1)


class Cosmos3LidarEncoder(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        validate_lidar_config(config)
        from .lidar_encoder.encoding import generate_polar_coords
        from .lidar_encoder.transformer_vae import Encoder

        self.config = config
        network = config["network_config"]
        encoder_args = set(inspect.signature(Encoder).parameters)
        self.encoder = Encoder(**{key: value for key, value in network.items() if key in encoder_args})
        channels = config["latent_channels"]
        self.quant_conv = nn.Conv2d(channels * 2, channels * 2, 1)
        self.register_buffer("coords", generate_polar_coords(*network["resolution"]))
        self.register_buffer("latent_mean", torch.empty(1, channels, 1, 1, 1))
        self.register_buffer("latent_std", torch.empty(1, channels, 1, 1, 1))

    @classmethod
    def from_pretrained(cls, model_path: str, config: dict[str, Any], device: torch.device) -> Cosmos3LidarEncoder:
        from safetensors.torch import load_file

        folder = Path(model_path) / "lidar_encoder"
        if not Path(model_path).exists():
            from huggingface_hub import snapshot_download

            folder = Path(snapshot_download(model_path, allow_patterns=["lidar_encoder/*"])) / "lidar_encoder"
        if not (folder / "config.json").is_file() or not (folder / "model.safetensors").is_file():
            raise ValueError("Incomplete joint artifact: lidar_encoder/config.json and model.safetensors are required.")
        if json.loads((folder / "config.json").read_text()) != config:
            raise ValueError("LiDAR encoder metadata disagrees with transformer deployment metadata.")
        # Pipeline construction may set the default parameter dtype to BF16.
        # Convert before loading so FP32 checkpoint values are never rounded.
        model = cls(config).float()
        model.load_state_dict(load_file(folder / "model.safetensors"), strict=True)
        if (
            not torch.isfinite(model.latent_mean).all()
            or not torch.isfinite(model.latent_std).all()
            or (model.latent_std <= 0).any()
        ):
            raise ValueError("LiDAR latent statistics must be finite with positive standard deviations.")
        return model.eval().requires_grad_(False).to(device=device, dtype=torch.float32)

    @torch.inference_mode()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # Keep normalization, network execution and the latent affine in FP32,
        # including when the parent DiT is under BF16 autocast.
        with torch.autocast(device_type=self.coords.device.type, enabled=False):
            state = prepare_lidar_encoder_input(frames.to(self.coords.device), self.config["range_projection"])
            chunk = self.config["streaming_chunk_frames"]
            context = self.config["streaming_context_frames"]
            outputs, cache = [], None
            for start in range(0, state.shape[2], chunk):
                pixels = state[:, :, start : start + chunk]
                if cache is not None and context is not None:
                    keep = context - pixels.shape[2]
                    cache = (
                        {key: (k[:, :, -keep:], v[:, :, -keep:]) for key, (k, v) in cache.items()} if keep > 0 else None
                    )
                parameters, cache = self.encoder.forward_stream(pixels, self.coords, cache)
                batch, _, time, height, width = parameters.shape
                parameters = parameters.permute(0, 2, 1, 3, 4).flatten(0, 1)
                mean = self.quant_conv(parameters).chunk(2, dim=1)[0]
                outputs.append(mean.reshape(batch, time, -1, height, width).permute(0, 2, 1, 3, 4))
            latent = torch.cat(outputs, dim=2)
            return (latent - self.latent_mean) / self.latent_std
