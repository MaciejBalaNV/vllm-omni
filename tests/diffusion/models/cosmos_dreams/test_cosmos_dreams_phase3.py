# SPDX-License-Identifier: Apache-2.0
"""CPU acceptance contracts for the Cosmos-Dreams Phase-3 serving surface."""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.cosmos_dreams.controller import (
    ACTION_COORDINATE_VERSION,
    AgiBotControllerLimits,
    AgiBotKeyboardController,
    AgiBotKeyboardResampler,
    AgiBotSceneState,
    build_scheduled_action_chunk,
    parse_key_schedule,
    schedule_to_controller_inputs,
)
from vllm_omni.diffusion.models.cosmos_dreams.realtime import (
    BoundedRGBFrameQueue,
    CosmosDreamsLatencyRecorder,
)
from vllm_omni.diffusion.models.cosmos_dreams.runtime import (
    CosmosDreamsTickResult,
    CosmosDreamsTickRuntime,
)
from vllm_omni.diffusion.models.cosmos_dreams.state_cosmos_dreams import CosmosDreamsSessionState
from vllm_omni.diffusion.models.cosmos_dreams.streaming_vae import decode_wan_causal_chunk
from vllm_omni.diffusion.models.cosmos_dreams.tick_adapter import (
    COSMOS_DREAMS_ACTION_SCHEMA,
    build_cosmos_dreams_action_control,
    parse_cosmos_dreams_tick,
)
from vllm_omni.errors import client_error_metadata
from vllm_omni.experimental.ar_diffusion.capability import (
    AR_DIFFUSION_REQUEST_REJECTED_ERROR_TYPE,
    ARDiffusionRequestRejectedError,
)
from vllm_omni.experimental.ar_diffusion.tick_protocol import (
    ARDiffusionChunkMetadata,
    ARDiffusionControlInput,
    ARDiffusionTickRequest,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_IDENTITY_ACTION_POSE = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


def _scene(*, fps: float = 16.0, limits: AgiBotControllerLimits | None = None) -> AgiBotSceneState:
    return AgiBotSceneState(
        seed_rgb=torch.zeros(3, 8, 12),
        prompt="A calibrated AgiBot scene",
        fps=fps,
        head_transform=torch.eye(4),
        right_wrist_transform=torch.eye(4),
        left_wrist_transform=torch.eye(4),
        limits=limits or AgiBotControllerLimits(),
    )


def test_scene_contract_rejects_wrong_domain_coordinate_version_and_idle_rotations() -> None:
    with pytest.raises(ValueError, match="domain 15"):
        AgiBotSceneState(
            seed_rgb=torch.zeros(3, 8, 12),
            prompt="scene",
            fps=30,
            domain_id=2,
            head_transform=torch.eye(4),
            right_wrist_transform=torch.eye(4),
            left_wrist_transform=torch.eye(4),
        )
    with pytest.raises(ValueError, match="coordinate contract"):
        AgiBotSceneState(
            seed_rgb=torch.zeros(3, 8, 12),
            prompt="scene",
            fps=30,
            action_coordinate_version="wrong",
            head_transform=torch.eye(4),
            right_wrist_transform=torch.eye(4),
            left_wrist_transform=torch.eye(4),
        )

    action = build_scheduled_action_chunk(AgiBotKeyboardController(_scene()), "none:16")
    assert tuple(action.shape) == (16, 29)
    assert action.dtype == torch.float32
    torch.testing.assert_close(action[:, :9], _IDENTITY_ACTION_POSE.expand(16, -1))
    torch.testing.assert_close(action[:, 9:18], _IDENTITY_ACTION_POSE.expand(16, -1))
    torch.testing.assert_close(action[:, 19:28], _IDENTITY_ACTION_POSE.expand(16, -1))
    assert torch.equal(action[:, 18], torch.ones(16))
    assert torch.equal(action[:, 28], torch.ones(16))
    assert ACTION_COORDINATE_VERSION.endswith("backward_framewise.rot6d.opencv.v1")


def test_joint_only_scene_uses_the_authoritative_kinematics_seam() -> None:
    class IdentityKinematics:
        def inverse(self, *, targets, joint_state):
            return joint_state

        def forward(self, joint_state):
            assert tuple(joint_state.shape) == (3,)
            return {"head": torch.eye(4), "right": torch.eye(4), "left": torch.eye(4)}

    scene = AgiBotSceneState(
        seed_rgb=torch.zeros(3, 8, 12),
        prompt="joint-calibrated scene",
        fps=16,
        joint_state=torch.zeros(3),
    )
    controller = AgiBotKeyboardController(scene, kinematics=IdentityKinematics())
    action = build_scheduled_action_chunk(controller, "none:16")
    torch.testing.assert_close(action[:, :9], _IDENTITY_ACTION_POSE.expand(16, -1))


def test_selected_arm_translation_rotation_and_gripper_use_the_raw_29d_layout() -> None:
    limits = AgiBotControllerLimits(linear_velocity_m_s=0.16, angular_velocity_rad_s=np.pi / 2)
    controller = AgiBotKeyboardController(_scene(fps=16, limits=limits))
    action = build_scheduled_action_chunk(controller, "right+w+j:16")

    # 0.16 m/s at 16 fps -> 1 cm forward per raw action row in the
    # selected wrist's local OpenCV frame. No normalizer or 64-D pad appears.
    torch.testing.assert_close(action[:, 9:12], torch.tensor([0.0, 0.0, 0.01]).expand(16, -1), atol=1e-6, rtol=0)
    torch.testing.assert_close(action[:, :9], _IDENTITY_ACTION_POSE.expand(16, -1))
    torch.testing.assert_close(action[:, 19:28], _IDENTITY_ACTION_POSE.expand(16, -1))
    assert action.shape[-1] == 29
    assert not torch.equal(action[0, 12:18], _IDENTITY_ACTION_POSE[3:])

    controller.reset()
    toggled = build_scheduled_action_chunk(controller, "left+space:1,left:15")
    assert torch.equal(toggled[:, 18], torch.ones(16))
    assert torch.equal(toggled[:, 28], torch.zeros(16))


def test_live_and_headless_key_schedules_use_the_same_resampler_and_controller() -> None:
    schedule = parse_key_schedule("right+w:4,right+w+j:4,space:1,none:7")
    headless_controller = AgiBotKeyboardController(_scene())
    headless_segments, headless_times = schedule_to_controller_inputs(schedule, fps=16)
    headless = headless_controller.build_action_chunk(headless_segments, headless_times)

    live = AgiBotKeyboardResampler(fps=16, start_time=0.0)
    live.on_edge(arrival_time=0.0, event="keydown", key="2")
    live.on_edge(arrival_time=0.0, event="keydown", key="w")
    live.on_edge(arrival_time=4 / 16, event="keydown", key="j")
    live.on_edge(arrival_time=8 / 16, event="keyup", key="2")
    live.on_edge(arrival_time=8 / 16, event="keyup", key="w")
    live.on_edge(arrival_time=8 / 16, event="keyup", key="j")
    live.on_edge(arrival_time=8 / 16, event="keydown", key="space")
    live.on_edge(arrival_time=9 / 16, event="keyup", key="space")
    live_segments, live_times = live.sample_chunk(16)
    browser = AgiBotKeyboardController(_scene()).build_action_chunk(live_segments, live_times)

    torch.testing.assert_close(browser, headless)


def test_controller_snapshot_restores_state_after_a_rejected_tick() -> None:
    controller = AgiBotKeyboardController(_scene())
    before = controller.snapshot()
    build_scheduled_action_chunk(controller, "right+w:16")
    controller.restore(before)
    retried = build_scheduled_action_chunk(controller, "none:16")
    torch.testing.assert_close(retried[:, 9:18], _IDENTITY_ACTION_POSE.expand(16, -1))


class _FakeWanDecoder:
    def __call__(self, x, *, feat_cache, feat_idx, first_chunk):
        assert feat_idx == [0]
        previous = torch.zeros_like(x) if feat_cache[0] is None else feat_cache[0]
        value = x + previous
        feat_cache[0] = value[:, :, -1:].clone()
        return value.repeat(1, 1, 1 if first_chunk else 4, 1, 1)


class _FakeWanVAE:
    def __init__(self) -> None:
        self.post_quant_conv = nn.Identity()
        self.decoder = _FakeWanDecoder()
        self.config = SimpleNamespace(patch_size=None)
        self.clear_cache()

    def clear_cache(self) -> None:
        self._feat_map = [None, "Rep"]
        self._conv_idx = [0]


def test_incremental_wan_decode_matches_one_shot_and_keeps_5_then_4_inputs() -> None:
    latents = torch.arange(9, dtype=torch.float32).view(1, 1, 9, 1, 1)
    one_shot = decode_wan_causal_chunk(
        _FakeWanVAE(),
        latents,
        feature_cache=None,
        initialized=False,
    ).video

    streaming_vae = _FakeWanVAE()
    first = decode_wan_causal_chunk(
        streaming_vae,
        latents[:, :, :5],
        feature_cache=None,
        initialized=False,
    )
    second = decode_wan_causal_chunk(
        streaming_vae,
        latents[:, :, 5:],
        feature_cache=first.feature_cache,
        initialized=True,
    )
    assert first.video.shape[2] == 17
    assert second.video.shape[2] == 16
    assert first.feature_cache[1] == second.feature_cache[1] == "Rep"
    torch.testing.assert_close(torch.cat([first.video, second.video], dim=2), one_shot)

    state = CosmosDreamsSessionState("live")
    state.append_chunk(latents[:, :, :5], frame_start=0, retain_latent=False)
    state.record_incremental_decode(input_frames=5, feature_cache=first.feature_cache)
    state.append_chunk(latents[:, :, 5:], frame_start=5, retain_latent=False)
    state.record_incremental_decode(input_frames=4, feature_cache=second.feature_cache)
    assert state.accumulated_latents is None
    assert state.next_frame_idx == 9
    assert state.last_vae_decode_input_frames == 4
    assert state.max_vae_decode_input_frames == 5


class _FakeTickBackend:
    def __init__(
        self,
        *,
        wrong_frames_at: int | None = None,
        reject_once: bool = False,
        fail_close_once: bool = False,
    ) -> None:
        self.engine = SimpleNamespace(stage_pools=[SimpleNamespace(num_replicas=1)])
        self.default_sampling_params_list = [OmniDiffusionSamplingParams()]
        self.requests: list[dict[str, object]] = []
        self.lifecycle_calls: list[tuple[str, str]] = []
        self.wrong_frames_at = wrong_frames_at
        self.reject_once = reject_once
        self.fail_close_once = fail_close_once

    async def generate(self, prompt, *, request_id, sampling_params_list, output_modalities=None):
        params = sampling_params_list[0]
        tick = ARDiffusionTickRequest.from_extra_args(params.extra_args)
        assert tick is not None
        inputs = parse_cosmos_dreams_tick(tick)
        self.requests.append(
            {
                "prompt": prompt,
                "request_id": request_id,
                "params": params,
                "tick": tick,
                "inputs": inputs,
                "output_modalities": output_modalities,
            }
        )
        if self.reject_once:
            self.reject_once = False
            raise ARDiffusionRequestRejectedError("session fingerprint mismatch")
        tick_index = len(self.requests) - 1
        frames = 17 if inputs.frame_idx == 0 else 16
        if tick_index == self.wrong_frames_at:
            frames -= 1
        yield OmniRequestOutput(
            request_id=request_id,
            stage_id=0,
            final_output_type="video",
            images=[torch.zeros(1, 3, frames, 8, 12)],  # type: ignore[list-item]
            _multimodal_output={"metadata": {"ar_diffusion": ARDiffusionChunkMetadata.from_tick(tick).to_dict()}},
            stage_durations={"denoise_s": 0.1, "clean_cache_commit_s": 0.02, "vae_decode_s": 0.03},
            finished=True,
        )

    async def collective_rpc(self, method, timeout=None, args=(), kwargs=None, stage_ids=None):
        self.lifecycle_calls.append((method, args[0]))
        if method == "close_ar_diffusion_session" and self.fail_close_once:
            self.fail_close_once = False
            raise RuntimeError("transient lifecycle failure")
        return [True]


class _CancelledTickBackend:
    def __init__(self) -> None:
        self.engine = SimpleNamespace(stage_pools=[SimpleNamespace(num_replicas=1)])
        self.default_sampling_params_list = [OmniDiffusionSamplingParams()]
        self.lifecycle_calls: list[tuple[str, str]] = []

    async def generate(self, prompt, *, request_id, sampling_params_list, output_modalities=None):
        raise asyncio.CancelledError
        yield  # pragma: no cover

    async def collective_rpc(self, method, timeout=None, args=(), kwargs=None, stage_ids=None):
        self.lifecycle_calls.append((method, args[0]))
        return [True]


def test_typed_action_adapter_round_trips_raw_agibot_tensor_and_rejects_wrong_schema() -> None:
    action = torch.arange(16 * 29, dtype=torch.float32).reshape(16, 29)
    control = build_cosmos_dreams_action_control(action, frame_idx=5, measure_tick_latency=True)
    tick = ARDiffusionTickRequest(
        session_id="s",
        request_id="r",
        chunk_index=1,
        controls=(control,),
    )
    parsed = parse_cosmos_dreams_tick(tick)
    torch.testing.assert_close(parsed.action, action)
    assert parsed.frame_idx == 5
    assert parsed.domain_name == "agibotworld" and parsed.domain_id == 15
    assert parsed.measure_tick_latency is True

    bad_tick = ARDiffusionTickRequest(
        session_id="s",
        request_id="bad",
        chunk_index=1,
        controls=(ARDiffusionControlInput(track="robot_action", schema="wrong.v1", data=control.data),),
    )
    with pytest.raises(ValueError, match="schema"):
        parse_cosmos_dreams_tick(bad_tick)


@pytest.mark.asyncio
async def test_tick_runtime_sends_typed_ticks_and_advances_0_5_9_for_eight_chunks() -> None:
    backend = _FakeTickBackend()
    runtime = CosmosDreamsTickRuntime(backend, _scene(fps=30), session_id="stable", seed=7)
    action = torch.zeros(16, 29)
    results = [await runtime.tick_async(action) for _ in range(8)]

    assert [result.frames.shape[2] for result in results] == [17] + [16] * 7
    assert [result.frame_idx for result in results] == [0, 5, 9, 13, 17, 21, 25, 29]
    assert [result.next_frame_idx for result in results] == [5, 9, 13, 17, 21, 25, 29, 33]
    assert backend.requests[0]["prompt"]["multi_modal_data"]["image"] is runtime.scene.seed_rgb
    assert all("multi_modal_data" not in request["prompt"] for request in backend.requests[1:])
    assert len({request["request_id"] for request in backend.requests}) == 8
    for index, request in enumerate(backend.requests):
        params = request["params"]
        typed_tick = request["tick"]
        inputs = request["inputs"]
        assert typed_tick.session_id == "stable" and typed_tick.chunk_index == index
        assert typed_tick.applied_event_ids == (index,)
        assert typed_tick.controls[0].schema == COSMOS_DREAMS_ACTION_SCHEMA
        assert inputs.num_latent_frames == 4
        assert tuple(inputs.action.shape) == (16, 29)
        assert params.num_inference_steps == 4 and params.guidance_scale == 1.0
    await runtime.close_async()
    assert backend.lifecycle_calls[-1] == ("close_ar_diffusion_session", "stable")
    assert len(backend.requests) == 8


@pytest.mark.asyncio
async def test_tick_runtime_does_not_advance_on_bad_output_and_resets_by_rpc() -> None:
    backend = _FakeTickBackend(wrong_frames_at=0)
    runtime = CosmosDreamsTickRuntime(backend, _scene())
    with pytest.raises(RuntimeError, match="expected 17, got 16"):
        await runtime.tick_async(torch.zeros(16, 29))
    assert runtime.next_frame_idx == 0
    assert not runtime.started
    with pytest.raises(RuntimeError, match="reset the session before retrying"):
        await runtime.tick_async(torch.zeros(16, 29))
    await runtime.reset_async()
    assert backend.lifecycle_calls[-1][0] == "reset_ar_diffusion_session"


@pytest.mark.asyncio
async def test_tick_failure_is_fail_closed_until_explicit_reset() -> None:
    backend = _FakeTickBackend(reject_once=True)
    runtime = CosmosDreamsTickRuntime(backend, _scene(), session_id="original")
    action = torch.zeros(16, 29)

    with pytest.raises(ARDiffusionRequestRejectedError, match="fingerprint mismatch"):
        await runtime.tick_async(action)
    assert not runtime.started and runtime.next_frame_idx == 0
    assert backend.lifecycle_calls[-1] == ("close_ar_diffusion_session", "original")
    with pytest.raises(RuntimeError, match="reset the session before retrying"):
        await runtime.tick_async(action)

    await runtime.reset_async(session_id="replacement")
    assert runtime.session_id == "replacement"
    after_reset = await runtime.tick_async(action)
    assert after_reset.session_id == "replacement" and after_reset.frame_idx == 0


@pytest.mark.asyncio
async def test_sync_entrypoint_rejects_an_active_event_loop() -> None:
    backend = _FakeTickBackend()
    runtime = CosmosDreamsTickRuntime(backend, _scene(), session_id="async")
    with pytest.raises(TypeError, match="use the async method"):
        runtime.tick(torch.zeros(16, 29))

    first = await runtime.tick_async(torch.zeros(16, 29))
    second = await runtime.tick_async(torch.zeros(16, 29))
    assert [first.frames.shape[2], second.frames.shape[2]] == [17, 16]
    assert [first.frame_idx, second.frame_idx] == [0, 5]
    await runtime.close_async()


def test_admission_error_keeps_stable_client_error_metadata() -> None:
    assert client_error_metadata(ARDiffusionRequestRejectedError("rejected")) == (
        400,
        AR_DIFFUSION_REQUEST_REJECTED_ERROR_TYPE,
    )


@pytest.mark.asyncio
async def test_cancelled_async_submission_is_fail_closed_but_can_release_the_session() -> None:
    backend = _CancelledTickBackend()
    runtime = CosmosDreamsTickRuntime(backend, _scene())
    with pytest.raises(asyncio.CancelledError):
        await runtime.tick_async(torch.zeros(16, 29))
    with pytest.raises(RuntimeError, match="reset the session before retrying"):
        await runtime.tick_async(torch.zeros(16, 29))
    await runtime.close_async()
    assert runtime.closed
    assert backend.lifecycle_calls[-1][0] == "close_ar_diffusion_session"


@pytest.mark.asyncio
async def test_failed_close_keeps_the_lifecycle_tombstone_retryable() -> None:
    backend = _FakeTickBackend(fail_close_once=True)
    runtime = CosmosDreamsTickRuntime(backend, _scene(), session_id="retry-close")
    await runtime.tick_async(torch.zeros(16, 29))

    with pytest.raises(RuntimeError, match="transient lifecycle failure"):
        await runtime.close_async()
    assert not runtime.closed

    await runtime.close_async()
    assert runtime.closed
    assert backend.lifecycle_calls[-2:] == [
        ("close_ar_diffusion_session", "retry-close"),
        ("close_ar_diffusion_session", "retry-close"),
    ]


@pytest.mark.asyncio
async def test_bounded_rgb_queue_applies_backpressure_and_never_exceeds_depth() -> None:
    queue = BoundedRGBFrameQueue(max_frames=2)
    result = CosmosDreamsTickResult(
        frames=torch.zeros(1, 3, 3, 2, 2),
        session_id="s",
        frame_idx=0,
        next_frame_idx=5,
        chunk_index=0,
        stage_durations={},
    )
    handed_off = False

    def on_handoff() -> None:
        nonlocal handed_off
        handed_off = True

    producer = asyncio.create_task(queue.enqueue_chunk(result, on_encoder_handoff=on_handoff))
    await asyncio.sleep(0.01)
    assert handed_off
    assert not producer.done()
    assert queue.qsize() == 2
    first_frame = await queue.get()
    assert first_frame.generation_id == 0
    assert await producer == 3
    assert queue.max_observed_depth == 2
    await queue.close()


class _FakeDataChannel:
    readyState = "open"  # noqa: N815 - mirrors aiortc's browser-compatible API

    def __init__(self) -> None:
        self.messages: list[dict[str, object]] = []

    def send(self, message: str) -> None:
        self.messages.append(json.loads(message))


def _load_webrtc_demo(monkeypatch):
    demo_path = Path(__file__).parents[4] / "examples/online_serving/cosmos_dreams/webrtc_demo.py"
    monkeypatch.syspath_prepend(str(demo_path.parent))
    monkeypatch.setitem(
        sys.modules,
        "demo_support",
        SimpleNamespace(
            load_replay_actions=lambda path: None,
            load_scene_bundle=lambda path: None,
        ),
    )
    spec = importlib.util.spec_from_file_location(f"cosmos_dreams_webrtc_test_{id(monkeypatch)}", demo_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_make_video_track", lambda *args, **kwargs: object())
    return module


async def _wait_until(predicate, *, timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("Timed out waiting for Cosmos-Dreams WebRTC test state.")
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_webrtc_reset_restarts_chunk_zero_and_rejects_late_old_generation_ack(monkeypatch) -> None:
    demo = _load_webrtc_demo(monkeypatch)
    channel = _FakeDataChannel()
    session = demo.CosmosDreamsWebRTCSession(
        backend=_FakeTickBackend(),
        scene=_scene(fps=1),
        seed=7,
        queue_frames=32,
        replay_actions=None,
    )
    session._channel = channel

    await session._handle_message(json.dumps({"type": "action", "action": {"event": "keydown", "key": "w"}}))
    await _wait_until(
        lambda: any(
            message.get("type") == "chunk_done" and message.get("generation_id") == 0 for message in channel.messages
        )
    )
    first_recorder = session.recorder

    await session._handle_message(json.dumps({"type": "reset"}))
    assert session._generation_id == 1
    assert session.recorder is not first_recorder
    assert any(
        message.get("type") == "reset_done" and message.get("generation_id") == 1 for message in channel.messages
    )
    await session._handle_message(json.dumps({"type": "action", "action": {"event": "keydown", "key": "w"}}))
    await _wait_until(
        lambda: any(
            message.get("type") == "chunk_done"
            and message.get("generation_id") == 1
            and message.get("chunk_index") == 0
            for message in channel.messages
        )
    )
    assert not session._worker.done()

    await session._handle_message(json.dumps({"type": "presented", "generation_id": 0, "chunk_index": 0}))
    assert session.recorder.record(0)["key_arrival_to_first_presented_s"] is None
    await session._handle_message(json.dumps({"type": "presented", "generation_id": 1, "chunk_index": 0}))
    assert session.recorder.record(0)["key_arrival_to_first_presented_s"] is not None
    await session._handle_message(json.dumps({"type": "reset"}))
    assert session._generation_id == 2
    assert not session._worker.done()
    await session.close()


@pytest.mark.asyncio
async def test_webrtc_worker_reports_recorder_failure_without_retaining_a_task_exception(monkeypatch) -> None:
    demo = _load_webrtc_demo(monkeypatch)
    channel = _FakeDataChannel()
    session = demo.CosmosDreamsWebRTCSession(
        backend=_FakeTickBackend(),
        scene=_scene(fps=1),
        seed=7,
        queue_frames=32,
        replay_actions=None,
    )
    session._channel = channel

    def fail_begin(**kwargs) -> None:
        raise ValueError("synthetic recorder collision")

    monkeypatch.setattr(session.recorder, "begin", fail_begin)
    await session._handle_message(json.dumps({"type": "action", "action": {"event": "keydown", "key": "w"}}))
    await _wait_until(lambda: session._worker.done())

    assert session._worker.exception() is None
    assert any(
        message.get("type") == "error"
        and "synthetic recorder collision" in str(message.get("message"))
        and message.get("session_reset_required") is True
        for message in channel.messages
    )
    await session.close()


def test_latency_recorder_reports_cold_separately_and_uses_16_frame_budget() -> None:
    recorder = CosmosDreamsLatencyRecorder()
    for index, duration in enumerate((1.0, 0.4, 0.5)):
        start = float(index * 2)
        recorder.begin(chunk_index=index, key_arrival_time=start, request_dispatch_time=start + 0.01)
        recorder.attach_tick(
            CosmosDreamsTickResult(
                frames=torch.empty(1, 3, 1, 1, 1),
                session_id="s",
                frame_idx=index,
                next_frame_idx=index + 1,
                chunk_index=index,
                stage_durations={"denoise_s": 0.1},
            ),
            timestamp=start + 0.15,
        )
        recorder.mark_encoder_handoff(index, timestamp=start + 0.2)
        recorder.mark_enqueue_done(index, timestamp=start + 0.3)
        recorder.mark_presented(index, timestamp=start + duration)
    summary = recorder.warm_summary(playback_fps=30)
    assert summary["warm_chunks"] == 2
    assert summary["p50_s"] == pytest.approx(0.45)
    assert summary["p90_s"] == pytest.approx(0.5)
    assert summary["playback_budget_s"] == pytest.approx(16 / 30)
    assert summary["sustains_playback_at_p90"] is True
    assert recorder.record(0)["gpu_to_encoder_handoff_s"] == pytest.approx(0.05)


def test_key_arrival_clock_is_validated_before_runtime_state_changes() -> None:
    runtime = CosmosDreamsTickRuntime(_FakeTickBackend(), _scene())
    future = time.perf_counter() + 10
    with pytest.raises(ValueError, match="not be in the future"):
        runtime.tick(torch.zeros(16, 29), key_arrival_time=future)
    assert runtime.next_frame_idx == 0


def test_live_generation_worker_has_no_offline_or_file_video_io() -> None:
    demo_path = Path(__file__).parents[4] / "examples/online_serving/cosmos_dreams/webrtc_demo.py"
    module = ast.parse(demo_path.read_text(encoding="utf-8"))
    worker = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_generation_worker"
    )
    worker_source = ast.unparse(worker).lower()
    forbidden = ("pickle", "jsonl", ".mp4", "torch.load", "video_reader", "load_scene_bundle")
    assert all(token not in worker_source for token in forbidden)
