# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cosmos3 deploy-schema topology."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

COSMOS3_PIPELINE = PipelineConfig(
    model_type="cosmos3_omni",
    model_arch="Cosmos3ForConditionalGeneration",
    hf_architectures=("Cosmos3ForConditionalGeneration",),
    diffusers_class_name="Cosmos3OmniDiffusersPipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="image",
        ),
    ),
)
