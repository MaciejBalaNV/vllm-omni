# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Admission must align effective CFG branches across sharded-weight ranks."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor
from vllm_omni.diffusion.executor.ray_executor import RayDiffusionExecutor
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched import RequestScheduler
from vllm_omni.diffusion.sched.interface import CachedRequestData, DiffusionSchedulerOutput, NewRequestData
from vllm_omni.diffusion.sched.request_scheduler import build_request_batch_sampling_params_key
from vllm_omni.diffusion.worker.utils import RunnerOutput
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture(params=["text", "empty_text", "embeddings"])
def negative_inputs(request):
    if request.param == "text":
        return {"negative_prompt": "blurry"}
    if request.param == "empty_text":
        return {"negative_prompt": ""}
    return {
        "negative_prompt_embeds": torch.zeros(1, 2, 4),
        "negative_prompt_embeds_mask": torch.ones(1, 2, dtype=torch.bool),
    }


def _make_request(request_id, negative_inputs=None, true_cfg_scale=4.0):
    return OmniDiffusionRequest(
        request_id=request_id,
        prompt={"prompt": f"prompt {request_id}", **(negative_inputs or {})},
        sampling_params=OmniDiffusionSamplingParams(
            guidance_scale=1.0,
            true_cfg_scale=true_cfg_scale,
            num_inference_steps=2,
        ),
    )


def _make_wave(*requests):
    return DiffusionSchedulerOutput(
        step_id=0,
        scheduled_new_reqs=[NewRequestData(request_id=req.request_id, req=req) for req in requests],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        finished_req_ids=set(),
        num_running_reqs=len(requests),
        num_waiting_reqs=0,
    )


@pytest.fixture(params=[MultiprocDiffusionExecutor, RayDiffusionExecutor], ids=["multiproc", "ray"])
def executor(request):
    executor = object.__new__(request.param)
    executor._closed = False
    executor._is_failed = False
    executor._result_mq = Mock()
    executor._broadcast_mq = Mock()
    executor.collective_rpc = Mock()
    return executor


@pytest.fixture(params=["hsdp", "dlo"])
def wave_config(request):
    config = SimpleNamespace(
        step_execution=False,
        parallel_config=SimpleNamespace(data_parallel_size=2, hsdp_data_parallel=request.param == "hsdp"),
    )
    if request.param == "dlo":
        config.diffusion_offload_config = {
            "mode": "layer",
            "components": ["dit"],
            "layer_options": {"dit": {"weight_transfer": "allgather"}},
        }
    return config


@pytest.mark.parametrize("true_cfg_scale", [None, 4.0])
def test_negative_inputs_change_qwen_cfg_admission_key(negative_inputs, true_cfg_scale):
    positive = _make_request("positive", true_cfg_scale=true_cfg_scale)
    guided = _make_request("guided", negative_inputs, true_cfg_scale)

    assert positive.sampling_params.do_classifier_free_guidance is False
    assert guided.sampling_params.do_classifier_free_guidance is False
    assert build_request_batch_sampling_params_key(positive) != build_request_batch_sampling_params_key(guided)


@pytest.mark.parametrize("true_cfg_scale", [None, 4.0])
def test_scheduler_separates_mixed_qwen_cfg_requests(negative_inputs, true_cfg_scale):
    scheduler = RequestScheduler()
    scheduler.initialize(SimpleNamespace(max_num_seqs=2))
    scheduler.add_request(_make_request("positive", true_cfg_scale=true_cfg_scale))
    scheduler.add_request(_make_request("guided", negative_inputs, true_cfg_scale))

    first = scheduler.schedule()

    assert first.scheduled_request_ids == ["positive"]
    assert first.num_waiting_reqs == 1
    scheduler.update_from_output(
        first,
        RunnerOutput(request_id="positive", finished=True, result=DiffusionOutput(output="done")),
    )
    second = scheduler.schedule()
    assert second.scheduled_request_ids == ["guided"]


@pytest.mark.parametrize("true_cfg_scale", [None, 4.0])
def test_executors_reject_mixed_qwen_cfg_wave(executor, wave_config, negative_inputs, true_cfg_scale):
    executor.od_config = wave_config
    wave = _make_wave(
        _make_request("positive", true_cfg_scale=true_cfg_scale),
        _make_request("guided", negative_inputs, true_cfg_scale),
    )

    with pytest.raises(ValueError, match="compatible shape, CFG"):
        executor.execute_request(wave)

    executor.collective_rpc.assert_not_called()


def test_matching_negative_inputs_run_after_rejected_wave(executor, wave_config, negative_inputs):
    executor.od_config = wave_config
    invalid = _make_wave(_make_request("positive"), _make_request("guided", negative_inputs))
    with pytest.raises(ValueError, match="compatible shape, CFG"):
        executor.execute_request(invalid)
    executor.collective_rpc.assert_not_called()

    wave = _make_wave(_make_request("first", negative_inputs), _make_request("second", negative_inputs))
    results = [DiffusionOutput(output="first"), DiffusionOutput(output="second")]
    executor.collective_rpc.return_value = (
        [{"dp_rank": rank, "output": out} for rank, out in enumerate(results)]
        if isinstance(executor, RayDiffusionExecutor)
        else results
    )

    output = executor.execute_request(wave)

    assert [item.result for item in output.runner_outputs] == results
    executor.collective_rpc.assert_called_once()


def test_negative_prompt_contents_are_request_local():
    first = _make_request("first", {"negative_prompt": "blurry"})
    second = _make_request("second", {"negative_prompt": "grainy"})

    assert build_request_batch_sampling_params_key(first) == build_request_batch_sampling_params_key(second)


def test_absent_and_none_negative_inputs_are_compatible():
    absent = _make_request("absent")
    explicit_none = _make_request(
        "none",
        {"negative_prompt": None, "negative_prompt_embeds": None, "negative_prompt_embeds_mask": None},
    )

    assert build_request_batch_sampling_params_key(absent) == build_request_batch_sampling_params_key(explicit_none)


def test_negative_pooled_embeds_change_admission_key():
    # Flux and HiDream require pooled negative embeddings to enable true CFG.
    embeds = {"negative_prompt_embeds": torch.zeros(1, 2, 4)}
    without_pooled = _make_request("without_pooled", embeds)
    with_pooled = _make_request(
        "with_pooled",
        {**embeds, "negative_pooled_prompt_embeds": torch.zeros(1, 4)},
    )

    assert build_request_batch_sampling_params_key(without_pooled) != build_request_batch_sampling_params_key(
        with_pooled
    )
