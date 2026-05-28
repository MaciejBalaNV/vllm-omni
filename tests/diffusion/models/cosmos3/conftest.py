# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn


class StubScheduler:
    def __init__(self, timesteps: list[int] | None = None, *, flow_shift: float = 1.0) -> None:
        self.timesteps = torch.tensor(timesteps or [9, 3], dtype=torch.int64)
        self.config = SimpleNamespace(num_train_timesteps=1000, flow_shift=flow_shift)
        self.set_timesteps_calls: list[tuple[int, torch.device]] = []
        self.step_calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def set_timesteps(self, num_steps: int, device: torch.device) -> None:
        self.set_timesteps_calls.append((num_steps, device))
        self.timesteps = torch.arange(num_steps, 0, -1, dtype=torch.int64, device=device)

    def step(self, noise_pred: torch.Tensor, timestep: torch.Tensor, latents: torch.Tensor, **kwargs):
        del kwargs
        self.step_calls.append((noise_pred.clone(), timestep.clone(), latents.clone()))
        return (latents + noise_pred,)


class _ModeLatentDist:
    def __init__(self, latents: torch.Tensor) -> None:
        self._latents = latents

    def mode(self) -> torch.Tensor:
        return self._latents


class StubCosmos3VAE:
    dtype = torch.float32

    def __init__(self, z_dim: int = 2, *, temporal: int = 4, spatial: int = 8) -> None:
        self.config = SimpleNamespace(
            z_dim=z_dim,
            scale_factor_temporal=temporal,
            scale_factor_spatial=spatial,
            latents_mean=[0.0] * z_dim,
            latents_std=[1.0] * z_dim,
        )

    def encode(self, video: torch.Tensor):
        latent_frames = (video.shape[2] - 1) // self.config.scale_factor_temporal + 1
        latent_height = video.shape[-2] // self.config.scale_factor_spatial
        latent_width = video.shape[-1] // self.config.scale_factor_spatial
        latents = torch.ones(
            video.shape[0],
            self.config.z_dim,
            latent_frames,
            latent_height,
            latent_width,
            dtype=video.dtype,
            device=video.device,
        )
        return SimpleNamespace(latent_dist=_ModeLatentDist(latents))

    def decode(self, latents: torch.Tensor, return_dict: bool = False):
        del return_dict
        return (latents,)


class StubCosmos3Transformer(nn.Module):
    def __init__(
        self,
        *,
        latent_channel_size: int = 2,
        sound_gen: bool = False,
        sound_dim: int = 3,
    ) -> None:
        super().__init__()
        self.latent_channel_size = latent_channel_size
        self.sound_gen = sound_gen
        self.sound_dim = sound_dim
        self.cached_kv: Any | None = None
        self.cached_freqs_gen: Any | None = None
        self.calls: list[dict[str, Any]] = []
        self.reset_calls = 0

    def reset_cache(self) -> None:
        self.reset_calls += 1
        self.cached_kv = None
        self.cached_freqs_gen = None

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        token = int(text_ids.reshape(-1)[0].item()) if text_ids.numel() else 0
        sound_latents = kwargs.get("sound_latents")
        self.calls.append(
            {
                "token": token,
                "timestep": timestep.clone(),
                "text_mask": text_mask.clone(),
                "cache_before": self.cached_kv,
                "kwargs": dict(kwargs),
            }
        )
        if self.cached_kv is None:
            marker = torch.tensor([token], dtype=torch.float32)
            self.cached_kv = [(marker, marker + 100)]
            self.cached_freqs_gen = (marker + 200, marker + 300)
        outputs: list[torch.Tensor] = [torch.full_like(hidden_states, float(token))]
        if sound_latents is not None:
            outputs.append(torch.full_like(sound_latents, float(token + 10)))
        return outputs[0] if len(outputs) == 1 else tuple(outputs)


def passthrough_progress_bar(iterable):
    return iterable


@pytest.fixture(autouse=True)
def fake_cosmos3_guardrails(monkeypatch: pytest.MonkeyPatch):
    module = types.ModuleType("vllm_omni.diffusion.models.cosmos3.guardrails")
    module.is_guardrails_enabled = lambda od_config, sampling_params=None: False
    module.ensure_initialized = lambda od_config: None
    module.check_text_safety = lambda text: None
    module.check_video_safety = lambda video: video
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


@pytest.fixture
def make_cosmos3_pipeline():
    def _make():
        from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import (
            Cosmos3OmniDiffusersPipeline,
        )

        pipeline = object.__new__(Cosmos3OmniDiffusersPipeline)
        nn.Module.__init__(pipeline)
        pipeline.od_config = SimpleNamespace()
        pipeline.device = torch.device("cpu")
        pipeline.dtype = torch.float32
        pipeline.transformer = StubCosmos3Transformer(latent_channel_size=2)
        pipeline.vae = StubCosmos3VAE(z_dim=2)
        pipeline.vae_scale_factor_temporal = 4
        pipeline.vae_scale_factor_spatial = 8
        pipeline.scheduler = StubScheduler([9, 3], flow_shift=1.0)
        pipeline._base_scheduler_config = pipeline.scheduler.config
        pipeline._engine_init_flow_shift = 1.0
        pipeline._current_flow_shift = 1.0
        pipeline._guidance_scale = None
        pipeline._num_timesteps = None
        pipeline.progress_bar = passthrough_progress_bar
        pipeline._sound_tokenizer = None
        return pipeline

    return _make


def make_sampling_params(**overrides: Any) -> SimpleNamespace:
    values = {
        "height": None,
        "width": None,
        "num_frames": None,
        "num_inference_steps": None,
        "guidance_scale": None,
        "generator": None,
        "seed": 123,
        "num_outputs_per_prompt": 1,
        "frame_rate": None,
        "resolved_frame_rate": None,
        "max_sequence_length": None,
        "extra_args": {},
    }
    values.update(overrides)
    return SimpleNamespace(**values)
