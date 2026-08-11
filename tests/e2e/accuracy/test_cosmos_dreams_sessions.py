# SPDX-License-Identifier: Apache-2.0
"""Cosmos-Dreams real-weight AR session equivalence smoke test."""

from __future__ import annotations

import os
from typing import Any

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion]

MODEL_ENV_VAR = "VLLM_TEST_COSMOS_DREAMS_MODEL"
PROMPT = "A stationary AgiBot observes a workbench."
HEIGHT = 720
WIDTH = 1280
SEED = 42


def _model_name() -> str:
    model = os.environ.get(MODEL_ENV_VAR)
    if not model:
        pytest.skip(f"Set {MODEL_ENV_VAR} to run Cosmos-Dreams full-model session tests.")
    if not torch.cuda.is_available():
        pytest.skip("Cosmos-Dreams full-model session tests require CUDA.")
    return model


def _unwrap_latent(output: Any) -> torch.Tensor:
    if isinstance(output, list):
        if len(output) != 1:
            raise AssertionError(f"Expected one Cosmos-Dreams output, got {len(output)}")
        return _unwrap_latent(output[0])
    if isinstance(output, OmniRequestOutput):
        if output.is_pipeline_output and output.request_output is not None:
            return _unwrap_latent(output.request_output)
        return _unwrap_latent(output.images)
    if isinstance(output, dict):
        return _unwrap_latent(output.get("video", output))
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Expected a latent tensor, got {type(output).__name__}")
    return output


def _sampling_params(*, extra_args: dict[str, Any], num_frames: int) -> OmniDiffusionSamplingParams:
    return OmniDiffusionSamplingParams(
        height=HEIGHT,
        width=WIDTH,
        num_frames=num_frames,
        num_inference_steps=4,
        guidance_scale=1.0,
        frame_rate=15.0,
        seed=SEED,
        output_type="latent",
        extra_args=extra_args,
    )


@pytest.mark.benchmark
@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_full_rollout_matches_two_persistent_tick_requests() -> None:
    """One request and chunk-per-request mode must produce the same latents."""

    model = _model_name()
    initial_latent = torch.zeros(1, 48, 1, HEIGHT // 16, WIDTH // 16, dtype=torch.bfloat16)
    full_actions = torch.linspace(-0.01, 0.01, 32 * 29, dtype=torch.float32).reshape(32, 29)
    omni = Omni(
        model=model,
        model_class_name="CosmosDreamsPipeline",
        deploy_config="vllm_omni/deploy/cosmos_dreams.yaml",
        enforce_eager=True,
    )

    full = _unwrap_latent(
        omni.generate(
            PROMPT,
            _sampling_params(
                num_frames=33,
                extra_args={
                    "session_id": "cosmos-dreams-e2e-full",
                    "reset": True,
                    "close_session": True,
                    "domain_id": 15,
                    "action": full_actions,
                    "initial_latent": initial_latent,
                },
            ),
        )
    )
    first_tick = _unwrap_latent(
        omni.generate(
            PROMPT,
            _sampling_params(
                num_frames=17,
                extra_args={
                    "session_id": "cosmos-dreams-e2e-ticks",
                    "reset": True,
                    "ar_diffusion_tick": True,
                    "num_latent_frames": 4,
                    "frame_idx": 0,
                    "domain_id": 15,
                    "action": full_actions[:16],
                    "initial_latent": initial_latent,
                },
            ),
        )
    )
    second_tick = _unwrap_latent(
        omni.generate(
            PROMPT,
            _sampling_params(
                num_frames=16,
                extra_args={
                    "session_id": "cosmos-dreams-e2e-ticks",
                    "close_session": True,
                    "ar_diffusion_tick": True,
                    "num_latent_frames": 4,
                    "frame_idx": 5,
                    "domain_id": 15,
                    "action": full_actions[16:],
                },
            ),
        )
    )

    assert full.shape[2] == 9
    assert first_tick.shape[2] == 5
    assert second_tick.shape[2] == 4
    torch.testing.assert_close(torch.cat([first_tick, second_tick], dim=2), full, rtol=0, atol=0)
