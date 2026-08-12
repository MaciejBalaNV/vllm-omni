# SPDX-License-Identifier: Apache-2.0
"""Model-neutral bounded delivery and latency records for the v1 demo."""

from __future__ import annotations

import asyncio
import math
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch

from vllm_omni.diffusion.models.cosmos_dreams.runtime import CosmosDreamsTickResult


def video_tensor_to_rgb_frames(video: torch.Tensor) -> list[np.ndarray]:
    """Convert one decoded ``[1,3,T,H,W]`` tensor to contiguous RGB8 frames."""

    if video.ndim != 5 or video.shape[0] != 1 or video.shape[1] != 3:
        raise ValueError(f"Expected decoded video [1,3,T,H,W], got {tuple(video.shape)}.")
    frames = video.detach().permute(0, 2, 3, 4, 1).squeeze(0)
    if frames.dtype == torch.uint8:
        frames = frames.cpu()
    else:
        frames = ((frames.float().clamp(-1, 1) * 0.5 + 0.5) * 255.0).round().to(torch.uint8).cpu()
    return [np.ascontiguousarray(frame.numpy()) for frame in frames]


@dataclass(frozen=True, slots=True)
class QueuedRGBFrame:
    frame: np.ndarray
    generation_id: int
    chunk_index: int
    frame_in_chunk: int


class BoundedRGBFrameQueue:
    """A blocking bounded queue: slow presentation backpressures generation."""

    def __init__(self, *, max_frames: int) -> None:
        if max_frames <= 0:
            raise ValueError("max_frames must be positive.")
        self.max_frames = int(max_frames)
        self._queue: asyncio.Queue[QueuedRGBFrame | None] = asyncio.Queue(maxsize=self.max_frames)
        self._closed = False
        self.max_observed_depth = 0

    @property
    def closed(self) -> bool:
        return self._closed

    def qsize(self) -> int:
        return self._queue.qsize()

    async def enqueue_chunk(
        self,
        result: CosmosDreamsTickResult,
        *,
        generation_id: int = 0,
        on_encoder_handoff: Callable[[], None] | None = None,
    ) -> int:
        if self._closed:
            return 0
        if isinstance(generation_id, bool) or not isinstance(generation_id, int) or generation_id < 0:
            raise ValueError("generation_id must be a non-negative integer.")
        frames = await asyncio.to_thread(video_tensor_to_rgb_frames, result.frames)
        if on_encoder_handoff is not None:
            on_encoder_handoff()
        for frame_idx, frame in enumerate(frames):
            if self._closed:
                return frame_idx
            await self._queue.put(
                QueuedRGBFrame(
                    frame=frame,
                    generation_id=generation_id,
                    chunk_index=result.chunk_index,
                    frame_in_chunk=frame_idx,
                )
            )
            self.max_observed_depth = max(self.max_observed_depth, self._queue.qsize())
        return len(frames)

    async def get(self) -> QueuedRGBFrame:
        if self._closed and self._queue.empty():
            raise EOFError("RGB frame queue is closed.")
        item = await self._queue.get()
        if item is None:
            raise EOFError("RGB frame queue is closed.")
        return item

    def clear(self) -> None:
        """Drop queued presentation frames during an explicit scene reset."""

        if self._closed:
            raise RuntimeError("Cannot clear a closed RGB frame queue.")
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._queue.put_nowait(None)


@dataclass(slots=True)
class CosmosDreamsChunkTiming:
    chunk_index: int
    key_arrival_time: float
    request_dispatch_time: float
    stage_durations: dict[str, float] = field(default_factory=dict)
    model_ready_time: float | None = None
    encoder_handoff_time: float | None = None
    enqueue_done_time: float | None = None
    first_presented_time: float | None = None

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "chunk_index": self.chunk_index,
            "key_arrival_to_request_s": self.request_dispatch_time - self.key_arrival_time,
            **self.stage_durations,
            "gpu_to_encoder_handoff_s": (
                None
                if self.model_ready_time is None or self.encoder_handoff_time is None
                else self.encoder_handoff_time - self.model_ready_time
            ),
            "enqueue_s": (
                None
                if self.encoder_handoff_time is None or self.enqueue_done_time is None
                else self.enqueue_done_time - self.encoder_handoff_time
            ),
            "key_arrival_to_first_presented_s": (
                None if self.first_presented_time is None else self.first_presented_time - self.key_arrival_time
            ),
        }


class CosmosDreamsLatencyRecorder:
    """Collect cold/warm per-chunk latency without hiding playback lag."""

    def __init__(self) -> None:
        self._timings: dict[int, CosmosDreamsChunkTiming] = {}

    def begin(self, *, chunk_index: int, key_arrival_time: float, request_dispatch_time: float) -> None:
        if chunk_index in self._timings:
            raise ValueError(f"Timing for chunk {chunk_index} already exists.")
        if request_dispatch_time < key_arrival_time:
            raise ValueError("request_dispatch_time cannot precede key_arrival_time.")
        self._timings[chunk_index] = CosmosDreamsChunkTiming(
            chunk_index=chunk_index,
            key_arrival_time=float(key_arrival_time),
            request_dispatch_time=float(request_dispatch_time),
        )

    def attach_tick(self, result: CosmosDreamsTickResult, *, timestamp: float | None = None) -> None:
        timing = self._timings[result.chunk_index]
        timing.stage_durations.update(result.stage_durations)
        timing.model_ready_time = time.perf_counter() if timestamp is None else float(timestamp)

    def discard(self, chunk_index: int) -> None:
        self._timings.pop(chunk_index, None)

    def mark_encoder_handoff(self, chunk_index: int, *, timestamp: float | None = None) -> None:
        self._timings[chunk_index].encoder_handoff_time = time.perf_counter() if timestamp is None else float(timestamp)

    def mark_enqueue_done(self, chunk_index: int, *, timestamp: float | None = None) -> None:
        self._timings[chunk_index].enqueue_done_time = time.perf_counter() if timestamp is None else float(timestamp)

    def mark_presented(self, chunk_index: int, *, timestamp: float | None = None) -> None:
        timing = self._timings[chunk_index]
        if timing.first_presented_time is None:
            timing.first_presented_time = time.perf_counter() if timestamp is None else float(timestamp)

    def records(self) -> list[dict[str, float | int | None]]:
        return [self._timings[index].as_dict() for index in sorted(self._timings)]

    def record(self, chunk_index: int) -> dict[str, float | int | None]:
        return self._timings[chunk_index].as_dict()

    def warm_summary(self, *, playback_fps: float, skip_cold_chunks: int = 1) -> dict[str, float | int | bool]:
        if not math.isfinite(playback_fps) or playback_fps <= 0:
            raise ValueError("playback_fps must be positive and finite.")
        warm = [
            timing
            for index, timing in sorted(self._timings.items())
            if index >= skip_cold_chunks and timing.first_presented_time is not None
        ]
        values = [timing.first_presented_time - timing.key_arrival_time for timing in warm]
        if not values:
            return {
                "warm_chunks": 0,
                "p50_s": math.nan,
                "p90_s": math.nan,
                "playback_budget_s": 16.0 / playback_fps,
                "sustains_playback_at_p90": False,
            }
        ordered = sorted(values)
        p90_index = min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)
        budget = 16.0 / playback_fps
        return {
            "warm_chunks": len(values),
            "p50_s": statistics.median(values),
            "p90_s": ordered[p90_index],
            "playback_budget_s": budget,
            "sustains_playback_at_p90": ordered[p90_index] <= budget,
        }


__all__ = [
    "BoundedRGBFrameQueue",
    "CosmosDreamsChunkTiming",
    "CosmosDreamsLatencyRecorder",
    "QueuedRGBFrame",
    "video_tensor_to_rgb_frames",
]
