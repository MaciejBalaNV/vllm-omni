# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from importlib.util import find_spec

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_FRAMEWORK_AVAILABLE = find_spec("cosmos_framework") is not None
requires_cosmos_framework = pytest.mark.skipif(
    not _FRAMEWORK_AVAILABLE,
    reason="cosmos_framework source is not present on PYTHONPATH",
)


@requires_cosmos_framework
def test_current_cosmos_framework_policy_modules_import_and_instantiate() -> None:
    from cosmos_framework.data.generator.action.utils.domain_utils import get_domain_id
    from cosmos_framework.data.generator.action.utils.pose_utils import convert_rotation
    from cosmos_framework.data.generator.action.utils.transforms import ActionTransformPipeline
    from cosmos_framework.model.generator.diffusion.samplers.fm_solvers_unipc import (
        FlowUniPCMultistepScheduler,
    )

    transform = ActionTransformPipeline(
        max_action_dim=64,
        cfg_dropout_rate=0.0,
        format_prompt_as_json=True,
    )
    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000,
        shift=1.0,
        use_dynamic_shifting=False,
    )
    scheduler.set_timesteps(4, device=torch.device("cpu"), shift=5.0)

    assert transform.prompt_json_formatter is not None
    assert len(scheduler.timesteps) == 4
    assert get_domain_id("droid_lerobot") == 8
    assert convert_rotation([0.0, 0.0, 0.0, 1.0], "quat_xyzw", "rot6d").shape == (6,)


@requires_cosmos_framework
def test_real_action_transform_pipeline_matches_droid_policy_contract() -> None:
    from cosmos_framework.data.generator.action.utils.transforms import ActionTransformPipeline

    transform = ActionTransformPipeline(
        max_action_dim=64,
        cfg_dropout_rate=0.0,
        format_prompt_as_json=True,
    )
    sample = {
        "ai_caption": "Pick up the cube.",
        "video": torch.zeros((3, 3, 192, 320), dtype=torch.uint8),
        "action": torch.zeros((3, 8), dtype=torch.float32),
        "conditioning_fps": torch.tensor(15, dtype=torch.long),
        "mode": "wam",
        "domain_id": torch.tensor(8, dtype=torch.long),
        "viewpoint": "concat_view",
        "additional_view_description": "Wrist view above two exterior views.",
    }

    transformed = transform(sample, "256")

    assert isinstance(transformed["ai_caption"], dict)
    assert transformed["action"].shape == (3, 64)
    assert int(transformed["raw_action_dim"].item()) == 8
    assert transformed["sequence_plan"].has_action is True
    assert transformed["mode"] == "wam"
    assert int(transformed["domain_id"].item()) == 8
