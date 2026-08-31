# SPDX-License-Identifier: Apache-2.0
"""Typed one-session runtime for Cosmos-Dreams interactive ticks."""

from __future__ import annotations

import asyncio
import math
import time
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch
from controller import AgiBotSceneState

from vllm_omni.diffusion.models.cosmos_dreams.tick_adapter import (
    build_cosmos_dreams_action_control,
)
from vllm_omni.experimental.ar_diffusion.consumer import ARDiffusionOmniTickConsumer
from vllm_omni.experimental.ar_diffusion.session import (
    ARDiffusionSession,
    ARDiffusionSessionEvent,
    ARDiffusionSessionManager,
    ARDiffusionWorkerLifecycle,
)
from vllm_omni.experimental.ar_diffusion.tick_protocol import ARDiffusionTickRequest
from vllm_omni.inputs.data import (
    OmniDiffusionSamplingParams,
    OmniPromptType,
    OmniSamplingParams,
)
from vllm_omni.outputs import OmniRequestOutput


class CosmosDreamsTickBackend(Protocol):
    """AsyncOmni subset used by the typed consumer and worker lifecycle."""

    engine: Any

    def generate(
        self,
        prompt: OmniPromptType,
        *,
        request_id: str,
        sampling_params_list: Sequence[OmniSamplingParams],
        output_modalities: list[str] | None = None,
    ) -> AsyncIterator[OmniRequestOutput]: ...

    async def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        stage_ids: list[int] | None = None,
    ) -> list[Any]: ...


@dataclass(frozen=True, slots=True)
class CosmosDreamsTickResult:
    frames: torch.Tensor
    session_id: str
    frame_idx: int
    next_frame_idx: int
    chunk_index: int
    stage_durations: dict[str, float]


@dataclass(frozen=True, slots=True)
class _PendingTick:
    first_tick: bool
    frame_idx: int
    chunk_index: int
    request_started: float
    key_arrival_delay: float | None


def _unwrap_omni_result(result: Any) -> tuple[Any, dict[str, float]]:
    if isinstance(result, list):
        if not result:
            raise ValueError("Cosmos-Dreams tick backend returned an empty result list.")
        result = result[-1]

    stage_durations = getattr(result, "stage_durations", None)
    durations = dict(stage_durations) if isinstance(stage_durations, dict) else {}
    unwrap = getattr(result, "unwrap", None)
    if callable(unwrap):
        result = unwrap()
        nested_durations = getattr(result, "stage_durations", None)
        if isinstance(nested_durations, dict):
            durations.update(nested_durations)
    images = getattr(result, "images", None)
    if images:
        result = images
    if isinstance(result, Mapping):
        payload = result.get("payload")
        if isinstance(payload, Mapping):
            result = payload
        if isinstance(result, Mapping):
            result = result.get("video", result.get("frames", result))
    if isinstance(result, list) and len(result) == 1:
        result = result[0]
    return result, {str(key): float(value) for key, value in durations.items()}


def _video_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        video = value.detach()
    else:
        if isinstance(value, list):
            converted = []
            for frame in value:
                if hasattr(frame, "convert"):
                    frame = np.asarray(frame.convert("RGB"))
                converted.append(np.asarray(frame))
            value = np.stack(converted) if converted else np.empty((0, 0, 0, 3), dtype=np.uint8)
        video = torch.as_tensor(np.asarray(value))

    if video.ndim == 4:
        if video.shape[0] in (3, 4) and video.shape[-1] not in (3, 4):
            video = video.unsqueeze(0)
        elif video.shape[-1] in (3, 4):
            video = video.permute(3, 0, 1, 2).unsqueeze(0)
    elif video.ndim == 5 and video.shape[-1] in (3, 4):
        video = video.permute(0, 4, 1, 2, 3)
    if video.ndim != 5 or video.shape[0] != 1 or video.shape[1] not in (3, 4):
        raise ValueError(
            f"Cosmos-Dreams tick output must be video shaped [1,C,T,H,W] or [T,H,W,C], got {tuple(video.shape)}."
        )
    if video.shape[1] == 4:
        video = video[:, :3]
    return video.contiguous()


class CosmosDreamsTickRuntime:
    """Drive one typed AR-Diffusion session from raw AgiBot action chunks."""

    def __init__(
        self,
        backend: CosmosDreamsTickBackend,
        scene: AgiBotSceneState,
        *,
        session_id: str | None = None,
        seed: int = 42,
        measure_latency: bool = True,
        diffusion_stage_id: int = 0,
        sampling_params_list: Sequence[OmniSamplingParams] | None = None,
        lifecycle_timeout: float | None = None,
    ) -> None:
        self.backend = backend
        self.scene = scene
        self.session_id = session_id or f"cosmos-dreams-{uuid.uuid4().hex}"
        self.seed = int(seed)
        self.measure_latency = bool(measure_latency)
        self.diffusion_stage_id = int(diffusion_stage_id)

        template = self._sampling_params_template()
        templates = list(
            sampling_params_list
            if sampling_params_list is not None
            else getattr(backend, "default_sampling_params_list", ())
        )
        if not templates:
            if self.diffusion_stage_id != 0:
                raise ValueError("sampling_params_list is required when diffusion_stage_id is not zero.")
            templates = [template]
        elif not 0 <= self.diffusion_stage_id < len(templates):
            raise ValueError("diffusion_stage_id must index sampling_params_list.")
        else:
            templates[self.diffusion_stage_id] = template

        self._consumer = ARDiffusionOmniTickConsumer(
            backend,
            prompt_provider=self._prompt_for_tick,
            sampling_params_list=templates,
            diffusion_stage_id=self.diffusion_stage_id,
            output_modalities=["video"],
        )
        self._lifecycle = ARDiffusionWorkerLifecycle(
            backend,
            stage_ids=[self.diffusion_stage_id],
            timeout=lifecycle_timeout,
        )
        self._session_manager: ARDiffusionSessionManager[OmniRequestOutput] = ARDiffusionSessionManager(
            tick_consumer=self._consumer,
            lifecycle=self._lifecycle,
            max_pending_events=1,
        )
        self._session: ARDiffusionSession[OmniRequestOutput] | None = None
        self._next_frame_idx = 0
        self._chunk_index = 0
        self._next_event_id = 0
        self._started = False
        self._requires_reset = False
        self._in_flight = False
        self._closed = False

    @property
    def next_frame_idx(self) -> int:
        return self._next_frame_idx

    @property
    def started(self) -> bool:
        return self._started

    @property
    def closed(self) -> bool:
        return self._closed

    def _sampling_params_template(self) -> OmniDiffusionSamplingParams:
        height, width = self.scene.resolution
        return OmniDiffusionSamplingParams(
            height=height,
            width=width,
            num_frames=17,
            num_inference_steps=4,
            guidance_scale=1.0,
            frame_rate=float(self.scene.fps),
            seed=self.seed,
        )

    def _prompt_for_tick(self, tick: ARDiffusionTickRequest) -> dict[str, Any]:
        prompt: dict[str, Any] = {"prompt": tick.prompt or self.scene.prompt}
        if tick.chunk_index == 0:
            prompt["multi_modal_data"] = {"image": self.scene.seed_rgb}
        return prompt

    async def _ensure_session(self) -> ARDiffusionSession[OmniRequestOutput]:
        if self._session is None:
            self._session = await self._session_manager.create_session(self.session_id)
        return self._session

    def _prepare_tick(
        self,
        action: torch.Tensor | np.ndarray,
        *,
        key_arrival_time: float | None,
    ) -> tuple[_PendingTick, Any]:
        if self._closed:
            raise RuntimeError("Cosmos-Dreams tick runtime is closed; create or reset a session explicitly.")
        if self._requires_reset:
            raise RuntimeError(
                "The previous Cosmos-Dreams tick failed after submission; reset the session before retrying."
            )
        if self._in_flight:
            raise RuntimeError("A Cosmos-Dreams tick is already in flight for this session.")
        if key_arrival_time is not None:
            pre_dispatch_delay = time.perf_counter() - float(key_arrival_time)
            if not math.isfinite(pre_dispatch_delay) or pre_dispatch_delay < 0:
                raise ValueError("key_arrival_time must use the same monotonic clock and not be in the future.")
        control = build_cosmos_dreams_action_control(
            torch.as_tensor(action),
            measure_tick_latency=self.measure_latency,
        )
        request_started = time.perf_counter()
        pending = _PendingTick(
            first_tick=not self._started,
            frame_idx=self._next_frame_idx,
            chunk_index=self._chunk_index,
            request_started=request_started,
            key_arrival_delay=(None if key_arrival_time is None else request_started - float(key_arrival_time)),
        )
        self._in_flight = True
        return pending, control

    def _complete_tick(self, result: OmniRequestOutput, pending: _PendingTick) -> CosmosDreamsTickResult:
        request_finished = time.perf_counter()
        try:
            value, stage_durations = _unwrap_omni_result(result)
            frames = _video_tensor(value)
            expected_frames = 17 if pending.first_tick else 16
            if int(frames.shape[2]) != expected_frames:
                raise RuntimeError(
                    "Cosmos-Dreams tick returned the wrong number of new RGB frames: "
                    f"expected {expected_frames}, got {frames.shape[2]}. Session was not advanced locally; "
                    "an explicit reset is required."
                )
        except Exception:
            self._requires_reset = True
            self._in_flight = False
            raise

        if pending.key_arrival_delay is not None:
            stage_durations["key_arrival_to_request_s"] = pending.key_arrival_delay
        stage_durations["client_request_wall_s"] = request_finished - pending.request_started

        self._next_frame_idx = 5 if pending.first_tick else self._next_frame_idx + 4
        self._started = True
        self._in_flight = False
        self._chunk_index += 1
        return CosmosDreamsTickResult(
            frames=frames,
            session_id=self.session_id,
            frame_idx=pending.frame_idx,
            next_frame_idx=self._next_frame_idx,
            chunk_index=pending.chunk_index,
            stage_durations=stage_durations,
        )

    async def tick_async(
        self,
        action: torch.Tensor | np.ndarray,
        *,
        key_arrival_time: float | None = None,
    ) -> CosmosDreamsTickResult:
        """Accept one action event and transactionally execute its typed tick."""

        pending, control = self._prepare_tick(action, key_arrival_time=key_arrival_time)
        try:
            session = await self._ensure_session()
            event = ARDiffusionSessionEvent(
                event_id=self._next_event_id,
                prompt=self.scene.prompt if not self._started else None,
                controls=(control,),
            )
            await session.accept_event(event)
            self._next_event_id += 1
            result = await session.next_chunk()
        except BaseException:
            self._requires_reset = True
            self._in_flight = False
            raise
        return self._complete_tick(result, pending)

    @staticmethod
    def _require_sync_context(operation: str) -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        raise TypeError(f"{operation} cannot run inside an event loop; use the async method instead.")

    def tick(
        self,
        action: torch.Tensor | np.ndarray,
        *,
        key_arrival_time: float | None = None,
    ) -> CosmosDreamsTickResult:
        self._require_sync_context("CosmosDreamsTickRuntime.tick")
        return asyncio.run(self.tick_async(action, key_arrival_time=key_arrival_time))

    def _reset_local(self, scene: AgiBotSceneState | None) -> None:
        if scene is not None:
            self.scene = scene
        self._next_frame_idx = 0
        self._chunk_index = 0
        self._started = False
        self._requires_reset = False
        self._in_flight = False
        self._closed = False

    async def reset_async(
        self,
        scene: AgiBotSceneState | None = None,
        *,
        session_id: str | None = None,
    ) -> None:
        if self._in_flight:
            raise RuntimeError("Cannot reset a Cosmos-Dreams session while a tick is in flight.")
        target_session_id = session_id or self.session_id
        if self._session is not None:
            if target_session_id == self.session_id:
                await self._session.reset()
            else:
                await self._session_manager.close_session(self.session_id)
                self._session = None
                self._next_event_id = 0
        self.session_id = target_session_id
        self._reset_local(scene)

    def reset(
        self,
        scene: AgiBotSceneState | None = None,
        *,
        session_id: str | None = None,
    ) -> None:
        self._require_sync_context("CosmosDreamsTickRuntime.reset")
        asyncio.run(self.reset_async(scene, session_id=session_id))

    async def close_async(self) -> None:
        if self._closed:
            return
        if self._in_flight:
            raise RuntimeError("Cannot close a Cosmos-Dreams session while a tick is in flight.")
        if self._session is not None:
            await self._session_manager.close_session(self.session_id)
            self._session = None
        self._closed = True

    def close(self) -> None:
        self._require_sync_context("CosmosDreamsTickRuntime.close")
        asyncio.run(self.close_async())


__all__ = [
    "CosmosDreamsTickBackend",
    "CosmosDreamsTickResult",
    "CosmosDreamsTickRuntime",
]
