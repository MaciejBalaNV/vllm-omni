# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-defined fixed-step sampler for distilled Cosmos-Dreams."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch


class CosmosDreamsDistilledSampler:
    """Four-step x0-prediction sampler with ODE or SDE transitions."""

    def __init__(
        self,
        t_list: Sequence[float],
        *,
        sample_type: str = "sde",
        num_train_timesteps: int = 1000,
    ) -> None:
        self.t_list = tuple(float(value) for value in t_list)
        self.sample_type = str(sample_type)
        self.num_train_timesteps = int(num_train_timesteps)
        if not self.t_list:
            raise ValueError("Cosmos-Dreams distilled t_list must not be empty")
        if abs(self.t_list[0] - 1.0) > 1e-6:
            raise ValueError(f"Cosmos-Dreams distilled t_list must start at 1.0, got {self.t_list[0]}")
        if any(sigma <= 0.0 or sigma > 1.0 for sigma in self.t_list):
            raise ValueError(f"Cosmos-Dreams distilled t_list entries must be in (0, 1], got {self.t_list}")
        if self.sample_type not in {"ode", "sde"}:
            raise ValueError(f"Cosmos-Dreams distilled sample_type must be 'ode' or 'sde', got {sample_type!r}")
        if self.num_train_timesteps <= 0:
            raise ValueError("Cosmos-Dreams num_train_timesteps must be positive")
        if any(left <= right for left, right in zip(self.t_list, self.t_list[1:])):
            raise ValueError(f"Cosmos-Dreams t_list must be strictly descending, got {self.t_list}")

    @property
    def schedule(self) -> tuple[float, ...]:
        return self.t_list if self.t_list[-1] == 0.0 else (*self.t_list, 0.0)

    @torch.no_grad()
    def sample(
        self,
        velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        initial_noise: torch.Tensor,
        *,
        seed: int,
        frame_idx: int,
    ) -> torch.Tensor:
        """Return fp32 clean latents; callers cast only before commit forward."""
        x = initial_noise.float()
        schedule = self.schedule
        for step_idx, (sigma_cur, sigma_next) in enumerate(zip(schedule[:-1], schedule[1:])):
            timestep = torch.full(
                (x.shape[0],),
                sigma_cur * self.num_train_timesteps,
                dtype=torch.float32,
                device=x.device,
            )
            velocity = velocity_fn(x, timestep).float()
            sigma_cur_tensor = torch.as_tensor(sigma_cur, dtype=torch.float32, device=x.device)
            x0_pred = x - sigma_cur_tensor * velocity
            if sigma_next == 0.0:
                x = x0_pred
                continue
            sigma_next_tensor = torch.as_tensor(sigma_next, dtype=torch.float32, device=x.device)
            if self.sample_type == "ode":
                x = x + (sigma_next_tensor - sigma_cur_tensor) * velocity
            else:
                step_seed = int(seed) + int(frame_idx) * 1_000_003 + int(step_idx) * 9_176
                generator = torch.Generator(device=x.device).manual_seed(step_seed)
                noise = torch.empty_like(x).normal_(generator=generator)
                x = (1.0 - sigma_next_tensor) * x0_pred + sigma_next_tensor * noise
        return x
