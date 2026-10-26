# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cosmos3 opt-in topologies for policy (OpenPI) and omni deploy YAML.

Both topologies declare neither ``hf_architectures`` nor
``diffusers_class_name``. Every Cosmos3 checkpoint (T2I, video, policy)
shares HF metadata (``model_type=cosmos3_omni``, ``model_index.json``
``_class_name=Cosmos3OmniDiffusersPipeline``). Auto-registering any of them
under that metadata would capture the others. Select explicitly via a deploy
yaml ``pipeline:`` key.

Policy (``cosmos3_policy``)::

    vllm serve nvidia/Cosmos3-Nano-Policy-DROID --omni \\
      --deploy-config .../vllm_omni/deploy/cosmos3_policy_droid.yaml

Omni deploy overlay (``cosmos3_omni_deploy``) — makes ``--deploy-config``
apply stage / ``model_config`` (e.g. ``guardrails: false``) instead of being
silently dropped (#6874). Same overlay works for Super / Nano (T2I and video).
``final_output_type`` matches the CLI Cosmos3OmniDiffusersPipeline default
(``video``)::

    vllm serve nvidia/Cosmos3-Super --omni \\
      --deploy-config .../vllm_omni/deploy/cosmos3_omni.yaml

Without ``--deploy-config``, T2I/video keep the default single-stage
diffusion CLI fallback. ``--no-guardrails`` remains the CLI-only path.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

COSMOS3_POLICY_PIPELINE = PipelineConfig(
    model_type="cosmos3_policy",
    model_arch="Cosmos3OmniDiffusersPipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="action",
            model_arch="Cosmos3OmniDiffusersPipeline",
        ),
    ),
)

COSMOS3_OMNI_DEPLOY_PIPELINE = PipelineConfig(
    model_type="cosmos3_omni_deploy",
    model_arch="Cosmos3OmniDiffusersPipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="video",
            model_arch="Cosmos3OmniDiffusersPipeline",
        ),
    ),
)


_PROC = "vllm_omni.model_executor.stage_input_processors.cosmos3"

COSMOS3_SHARED_UND_PIPELINE = PipelineConfig(
    model_type="cosmos3_shared_und",
    default_output_modalities=("text",),
    default_deploy_config_name="cosmos3_shared_und.yaml",
    model_arch="OmniCosmos3ForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="und",
            execution_type=StageExecutionType.LLM_AR,
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            model_arch="OmniCosmos3ForConditionalGeneration",
            engine_output_type="text",
            tokenizer_subdir="text_tokenizer",
            prompt_transform_func=f"{_PROC}.Cosmos3PromptTransform",
            prompt_expand_func=f"{_PROC}.expand_cfg_prompts",
            omni_kv_config={"need_send_cache": True, "kv_transfer_criteria": {"type": "prefill_finished"}},
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(0,),
            final_output=True,
            final_output_type="video",
            final_output_types=("image", "video"),
            model_arch="Cosmos3OmniDiffusersPipeline",
            requires_multimodal_data=True,
            cfg_kv_collect_func=f"{_PROC}.collect_cfg_kv_caches",
            omni_kv_config={"need_recv_cache": True},
        ),
    ),
)
