# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from pathlib import Path

import pytest

from vllm_omni.config.omni_config import VllmOmniConfig
from vllm_omni.config.pipeline_registry import resolve_pipeline_config
from vllm_omni.config.stage_config import load_deploy_config, merge_pipeline_deploy
from vllm_omni.entrypoints.utils import get_final_stage_id_for_e2e
from vllm_omni.outputs.stage_capabilities import stage_output_types

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    "setting,expected",
    [("", None), ("cfg_companion_wait_timeout_s: null\n", None), ("cfg_companion_wait_timeout_s: 12.5\n", 12.5)],
)
def test_companion_timeout_preserves_disabled_default_and_fractional_values(tmp_path, setting, expected):
    path = tmp_path / "deploy.yaml"
    path.write_text("stages: []\n" + setting)
    assert load_deploy_config(path).cfg_companion_wait_timeout_s == expected


def test_shared_und_is_opt_in_and_defaults_survive_both_config_builders():
    deploy = load_deploy_config(Path(__file__).parents[2] / "vllm_omni/deploy/cosmos3_shared_und.yaml")
    pipeline = resolve_pipeline_config(deploy.pipeline)
    assert not pipeline.hf_architectures and pipeline.diffusers_class_name is None
    assert pipeline.default_output_modalities == ("text",)
    assert deploy.cfg_companion_wait_timeout_s == 300
    assert deploy.model_runner == "v1" and not deploy.async_chunk
    legacy = merge_pipeline_deploy(pipeline, deploy)
    structured = VllmOmniConfig.from_pipeline_config(pipeline, user_deploy_config=deploy)
    ar = structured.stage_by_id(0)
    assert ar.scheduler_config.max_num_seqs == legacy[0].yaml_engine_args["max_num_seqs"] == 4
    assert ar.scheduler_config.max_model_len == 8192
    assert ar.scheduler_config.max_num_batched_tokens == 4096
    assert ar.cache_config.kv_cache_memory_bytes == 8 * 1024**3
    assert not ar.cache_config.enable_prefix_caching
    assert ar.scheduler_config.enable_chunked_prefill
    assert ar.scheduler_cls.endswith("OmniARAsyncScheduler")
    assert stage_output_types(structured.stage_by_id(1)) == ("image", "video")
    for modality, stage in (("text", 0), ("image", 1), ("video", 1)):
        assert get_final_stage_id_for_e2e([modality], ["text", "image", "video"], legacy) == stage
