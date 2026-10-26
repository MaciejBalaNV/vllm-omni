# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.request import Request, RequestStatus

from tests.helpers.omni_scheduler import bind_omits_transfer_helpers
from vllm import SamplingParams
from vllm_omni.core.sched.omni_ar_scheduler import OmniARAsyncScheduler, OmniARScheduler
from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin
from vllm_omni.engine.async_engine_utils import apply_omni_final_stage_metadata
from vllm_omni.engine.serialization import deserialize_additional_information

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_engine_request() -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id="req",
        prompt_token_ids=[1],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


def _omit_flags(*, force_kv_transfer: bool) -> tuple[bool, bool, dict]:
    tagged = apply_omni_final_stage_metadata(
        _make_engine_request(),
        final_stage_id=0,
        force_kv_transfer=force_kv_transfer,
    )
    scheduler = OmniARScheduler.__new__(OmniARScheduler)
    scheduler._omits_kv_transfer_cache = {}
    request = SimpleNamespace(
        request_id="req",
        additional_information=tagged.additional_information,
    )
    omits_kv = scheduler._request_omits_kv_transfer_to_next_stage(request)
    omits_chunk = scheduler._request_omits_chunk_transfer_to_next_stage(request)
    metadata = deserialize_additional_information(tagged.additional_information)
    return omits_kv, omits_chunk, metadata


def test_stage_zero_request_omits_kv_transfer():
    omits_kv, omits_chunk, metadata = _omit_flags(force_kv_transfer=False)

    assert omits_kv
    assert omits_chunk
    assert "omni_force_kv_transfer" not in metadata


def test_cfg_companion_forces_kv_transfer_without_downstream_payload():
    omits_kv, omits_chunk, metadata = _omit_flags(force_kv_transfer=True)

    assert not omits_kv
    assert omits_chunk
    assert metadata["omni_final_stage_id"] == 0
    assert metadata["omni_force_kv_transfer"] is True


class _ChunkRequest(SimpleNamespace):
    def __hash__(self):
        return hash(self.request_id)

    def __eq__(self, other):
        return isinstance(other, _ChunkRequest) and other.request_id == self.request_id


def _make_chunk_request(final_stage_id: int, *, force_kv_transfer: bool = False):
    tagged = apply_omni_final_stage_metadata(
        _make_engine_request(),
        final_stage_id=final_stage_id,
        force_kv_transfer=force_kv_transfer,
    )
    request = _ChunkRequest(
        request_id="req-shm-leak",
        additional_information=tagged.additional_information,
        num_in_flight_tokens=1,
        num_stale_output_tokens=0,
        sampling_params=SimpleNamespace(num_logprobs=None),
        has_encoder_inputs=False,
        pooling_params=None,
        status=RequestStatus.RUNNING,
        resumable=False,
        num_output_placeholders=0,
        spec_token_ids=[],
        _output_token_ids=[],
        num_computed_tokens=1,
        stop_reason=None,
        num_nans_in_logits=None,
        client_index=0,
        trace_headers=None,
    )
    request.is_finished = lambda: False
    request.get_finished_reason = lambda: None
    request.take_prefill_stats = lambda: None
    request.take_events = lambda: None
    return request


def _run_finished_save_step(mocker, request, *, inter_stage_outputs=None):
    adapter = mocker.MagicMock()
    adapter._confirmed_num_computed_tokens.return_value = 1

    sched = mocker.MagicMock()
    sched.requests = {request.request_id: request}
    sched.perf_metrics = None
    sched.defer_block_free = False
    sched.structured_output_manager.should_advance.return_value = False
    sched._update_request_with_output.return_value = ([42], True)
    sched._process_kv_transfer_trigger.return_value = False
    sched._handle_stopped_request.return_value = True

    def _free_request(req, delay_free_blocks=False):
        sched._omits_kv_transfer_cache.pop(req.request_id, None)
        return None, None

    sched._free_request.side_effect = _free_request
    sched.chunk_transfer_adapter = adapter
    sched.running = [request]
    sched.waiting_for_transfer_free = set()
    sched.transfer_triggered_requests = set()
    sched.active_kv_transfers = set()
    sched.pending_stop_after_extraction = set()
    sched.connector = None
    sched.kv_cache_manager.take_events.return_value = None
    sched.kv_cache_manager.estimate_cached_tokens.return_value = 0
    sched.finished_req_ids_dict = {}
    sched.make_stats.return_value = None
    bind_omits_transfer_helpers(sched)

    scheduler_output = mocker.MagicMock()
    scheduler_output.num_scheduled_tokens = {request.request_id: 1}
    scheduler_output.total_num_scheduled_tokens = 1
    scheduler_output.scheduled_spec_decode_tokens = {}
    scheduler_output.num_invalid_spec_tokens = 0

    model_runner_output = mocker.MagicMock()
    model_runner_output.sampled_token_ids = [[42]]
    model_runner_output.logprobs = None
    model_runner_output.prompt_logprobs_dict = {}
    model_runner_output.pooler_output = None
    model_runner_output.num_nans_in_logits = None
    model_runner_output.kv_connector_output = None
    model_runner_output.cudagraph_stats = None
    model_runner_output.req_id_to_index = {request.request_id: 0}
    model_runner_output.routed_experts = None
    model_runner_output.inter_stage_outputs = inter_stage_outputs

    OmniARScheduler.update_from_output(sched, scheduler_output, model_runner_output)
    return adapter


@pytest.mark.parametrize(
    ("final_stage_id", "force_kv_transfer", "expect_save"),
    [
        pytest.param(0, False, False, id="stage-zero-final"),
        pytest.param(1, False, True, id="downstream-final"),
    ],
)
def test_finish_saves_async_only_with_downstream_consumer(mocker, final_stage_id, force_kv_transfer, expect_save):
    request = _make_chunk_request(final_stage_id, force_kv_transfer=force_kv_transfer)
    adapter = _run_finished_save_step(mocker, request)

    assert adapter.save_async.called is expect_save


def test_stage_zero_final_skips_inter_stage_output(mocker):
    warn = mocker.patch("vllm_omni.core.sched.omni_ar_scheduler.logger.warning")
    request = _make_chunk_request(0)
    adapter = _run_finished_save_step(mocker, request, inter_stage_outputs=[{"codes": 1}])

    adapter.save_async.assert_not_called()
    assert any("inter_stage_output is present" in str(call) for call in warn.call_args_list)


def test_streaming_session_update_invalidates_omits_kv_transfer_cache():
    scheduler = OmniARScheduler.__new__(OmniARScheduler)
    scheduler._omits_kv_transfer_cache = {}
    scheduler.deferred_waiting = set()
    scheduler.kv_holding_waiting = []
    scheduler.log_stats = False
    tagged = apply_omni_final_stage_metadata(_make_engine_request(), final_stage_id=0)

    class Session(SimpleNamespace):
        __hash__ = object.__hash__

    session = Session(
        request_id="req",
        additional_information=tagged.additional_information,
        status=RequestStatus.WAITING,
    )
    assert scheduler._request_omits_chunk_transfer_to_next_stage(session)
    assert "req" in scheduler._omits_kv_transfer_cache
    update = SimpleNamespace(
        additional_information={"omni_final_stage_id": 1},
        model_intermediate_buffer=None,
        arrival_time=0.0,
        sampling_params=None,
    )

    OmniSchedulerMixin._finish_streaming_session_update(scheduler, session, update)

    assert scheduler._omits_kv_transfer_cache == {}
    assert session.additional_information == {"omni_final_stage_id": 1}
    assert not scheduler._request_omits_chunk_transfer_to_next_stage(session)
    assert not scheduler._request_omits_kv_transfer_to_next_stage(session)


def test_omits_cache_misses_when_payload_object_is_replaced():
    scheduler = OmniARScheduler.__new__(OmniARScheduler)
    scheduler._omits_kv_transfer_cache = {}
    tagged = apply_omni_final_stage_metadata(_make_engine_request(), final_stage_id=0)
    request = SimpleNamespace(
        request_id="req",
        additional_information=tagged.additional_information,
    )
    assert scheduler._request_omits_chunk_transfer_to_next_stage(request)
    request.additional_information = {"omni_final_stage_id": 1}

    assert not scheduler._request_omits_chunk_transfer_to_next_stage(request)


@pytest.mark.parametrize("length,confirmed", [(15, 15), (16, 17), (17, 20), (31, 33), (32, 34), (33, 33)])
def test_conditioning_snapshot_selects_exact_prompt_without_clamping_reasoning(length, confirmed):
    from dataclasses import dataclass

    @dataclass
    class RequestView:
        num_prompt_tokens: int
        additional_information: dict

    @dataclass
    class CacheConfig:
        block_size: int = 16

    class Blocks:
        def get_block_ids(self, _):
            return ([1, 2, 3],)

    scheduler = OmniARScheduler.__new__(OmniARScheduler)
    scheduler.requests_needing_kv_transfer = {}
    scheduler.requests = {
        "conditioning": RequestView(length, {"cosmos3_conditioning_length": length}),
        "reasoning": RequestView(length, {}),
    }
    scheduler._should_transfer_kv_for_request = lambda _: True
    scheduler.cache_config = CacheConfig()
    scheduler.kv_cache_manager = Blocks()
    scheduler._mark_request_for_kv_transfer("conditioning", confirmed)
    scheduler._mark_request_for_kv_transfer("reasoning", confirmed)
    selected = scheduler.requests_needing_kv_transfer["conditioning"]
    assert selected["seq_len"] == length
    assert selected["custom_metadata"]["confirmed_computed"] == confirmed
    assert len(selected["block_ids"]) == (length + 15) // 16
    assert scheduler.requests_needing_kv_transfer["reasoning"]["seq_len"] == confirmed
    scheduler.requests_needing_kv_transfer.clear()
    with pytest.raises(ValueError, match="does not cover"):
        scheduler._mark_request_for_kv_transfer("conditioning", length - 1)


def _conditioning_abort_scheduler(mocker, scheduler_cls, *, computed, placeholders, companion=False):
    """Use real request queues and finish/free methods; fake only cache storage."""
    scheduler = scheduler_cls.__new__(scheduler_cls)
    request = Request(
        request_id="conditioning",
        prompt_token_ids=list(range(32)),
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
    )
    request.additional_information = {
        "cosmos3_conditioning_length": 32,
        "omni_final_stage_id": 0 if companion else 1,
        "omni_force_kv_transfer": companion,
    }
    request.num_computed_tokens = computed + placeholders
    request.num_output_placeholders = placeholders
    request.status = RequestStatus.RUNNING if computed else RequestStatus.WAITING
    scheduler.requests = {request.request_id: request}
    scheduler.running = [request] if computed else []
    scheduler.waiting = create_request_queue(SchedulingPolicy.FCFS)
    scheduler.kv_holding_waiting = create_request_queue(SchedulingPolicy.FCFS)
    if not computed:
        scheduler.waiting.add_request(request)
    scheduler.deferred_waiting = set()
    scheduler.num_waiting_for_streaming_input = 0
    scheduler.chunk_transfer_adapter = None
    scheduler.input_coordinator = mocker.Mock()
    scheduler._omits_kv_transfer_cache = {}
    scheduler._omni_kv_config = {"need_send_cache": True}
    scheduler.vllm_config = SimpleNamespace(model_config=SimpleNamespace(stage_id=0), kv_transfer_config=None)
    scheduler.connector = None
    scheduler.ec_connector = None
    scheduler.encoder_cache_manager = mocker.Mock()
    scheduler.kv_cache_manager = mocker.Mock()
    scheduler.kv_cache_manager.get_block_ids.return_value = ([1, 2],)
    scheduler.cache_config = SimpleNamespace(block_size=16)
    scheduler.defer_block_free = False
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = None
    scheduler._new_prompt_len_snapshot = {}
    scheduler.transfer_triggered_requests = set()
    scheduler.active_kv_transfers = set()
    scheduler.waiting_for_transfer_free = set()
    scheduler.pending_stop_after_extraction = set()
    scheduler.requests_needing_kv_transfer = {}
    scheduler._kv_wait_start_ts = {}
    return scheduler, request


@pytest.mark.parametrize("scheduler_cls", [OmniARScheduler, OmniARAsyncScheduler])
@pytest.mark.parametrize("computed,placeholders", [(0, 0), (15, 0), (16, 1), (17, 2), (32, 0)])
@pytest.mark.parametrize("companion", [False, True])
def test_abort_conditioning_before_export_frees_request(mocker, scheduler_cls, computed, placeholders, companion):
    scheduler, request = _conditioning_abort_scheduler(
        mocker, scheduler_cls, computed=computed, placeholders=placeholders, companion=companion
    )

    assert scheduler.finish_requests([request.request_id], RequestStatus.FINISHED_ABORTED) == [request]

    assert request.status == RequestStatus.FINISHED_ABORTED
    assert not scheduler.requests
    assert not scheduler.running
    assert not scheduler.waiting
    assert not scheduler.waiting_for_transfer_free
    assert not scheduler._has_pending_kv_work()
    assert not scheduler._kv_wait_start_ts
    assert scheduler.finished_req_ids == {request.request_id}
    scheduler.kv_cache_manager.free.assert_called_once_with(request)
    scheduler.encoder_cache_manager.free.assert_called_once_with(request)
    scheduler.input_coordinator.free_finished_request.assert_any_call(request.request_id)

    # A subsequent request still takes the normal prefill-finished export path.
    next_request = Request(
        request_id="next-conditioning",
        prompt_token_ids=list(range(32)),
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
    )
    next_request.additional_information = request.additional_information
    scheduler.requests[next_request.request_id] = next_request
    scheduler._commit_kv_transfer_trigger(next_request.request_id, 32, True)
    assert scheduler.requests_needing_kv_transfer[next_request.request_id]["seq_len"] == 32


@pytest.mark.parametrize("scheduler_cls", [OmniARScheduler, OmniARAsyncScheduler])
@pytest.mark.parametrize("extraction_started", [False, True])
def test_abort_conditioning_preserves_triggered_extraction_hold(mocker, scheduler_cls, extraction_started):
    scheduler, request = _conditioning_abort_scheduler(mocker, scheduler_cls, computed=32, placeholders=0)
    scheduler._commit_kv_transfer_trigger(request.request_id, 32, True)
    if extraction_started:
        scheduler.get_finished_requests_needing_kv_transfer()

    scheduler.finish_requests([request.request_id], RequestStatus.FINISHED_ABORTED)

    if extraction_started:
        assert request.request_id in scheduler.waiting_for_transfer_free
        assert request.request_id in scheduler.active_kv_transfers
        scheduler.kv_cache_manager.free.assert_not_called()
    else:
        # A trigger queued for extraction also owns the blocks until the runner
        # has consumed it, even if cancellation precedes the runner's next step.
        assert request.request_id in scheduler.waiting_for_transfer_free
        assert request.request_id in scheduler.requests_needing_kv_transfer
        scheduler.kv_cache_manager.free.assert_not_called()
