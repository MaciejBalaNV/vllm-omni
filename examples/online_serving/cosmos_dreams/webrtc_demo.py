# SPDX-License-Identifier: Apache-2.0
"""Single-user keyboard/WebRTC demo for stateful Cosmos-Dreams ticks."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import time
from collections import deque
from fractions import Fraction
from functools import partial
from pathlib import Path
from typing import Any

import torch
from demo_support import load_replay_actions, load_scene_bundle

from vllm_omni.diffusion.models.cosmos_dreams.controller import (
    ACTION_STEPS_PER_TICK,
    AgiBotKeyboardController,
    AgiBotKeyboardResampler,
)
from vllm_omni.diffusion.models.cosmos_dreams.realtime import (
    BoundedRGBFrameQueue,
    CosmosDreamsLatencyRecorder,
)
from vllm_omni.diffusion.models.cosmos_dreams.runtime import CosmosDreamsTickRuntime

logger = logging.getLogger("cosmos_dreams.webrtc")


def _require_webrtc():
    try:
        from aiohttp import web
        from aiortc import RTCPeerConnection, RTCRtpSender, RTCSessionDescription
        from aiortc.mediastreams import MediaStreamError, MediaStreamTrack
        from av import VideoFrame
    except ImportError as exc:
        raise RuntimeError(
            "The Cosmos-Dreams WebRTC demo requires aiohttp, aiortc, and av. "
            "Install the 'cosmos-dreams-demo' optional dependencies."
        ) from exc
    return web, RTCPeerConnection, RTCRtpSender, RTCSessionDescription, MediaStreamError, MediaStreamTrack, VideoFrame


def _make_video_track(
    queue: BoundedRGBFrameQueue,
    *,
    fps: float,
    on_chunk_start,
):
    _, _, _, _, MediaStreamError, MediaStreamTrack, VideoFrame = _require_webrtc()

    class CosmosDreamsVideoTrack(MediaStreamTrack):
        kind = "video"

        def __init__(self) -> None:
            super().__init__()
            self._time_base = Fraction(1, 90_000)
            self._pts = 0
            self._pts_step = max(1, round(90_000 / fps))
            self._next_deadline: float | None = None
            self._frame_interval = 1.0 / fps

        async def recv(self):
            try:
                queued = await queue.get()
            except EOFError as exc:
                raise MediaStreamError from exc
            loop = asyncio.get_running_loop()
            now = loop.time()
            if self._next_deadline is None:
                self._next_deadline = now
            else:
                deadline = self._next_deadline + self._frame_interval
                if deadline > now:
                    await asyncio.sleep(deadline - now)
                    self._next_deadline = deadline
                else:
                    self._next_deadline = now
            first_new_frame = 1 if queued.chunk_index == 0 else 0
            if queued.frame_in_chunk == first_new_frame:
                on_chunk_start(queued.generation_id, queued.chunk_index)
            frame = VideoFrame.from_ndarray(queued.frame, format="rgb24")
            frame.pts = self._pts
            frame.time_base = self._time_base
            self._pts += self._pts_step
            return frame

    return CosmosDreamsVideoTrack()


class CosmosDreamsWebRTCSession:
    def __init__(
        self,
        *,
        backend,
        scene,
        seed: int,
        queue_frames: int,
        replay_actions: list[torch.Tensor] | None,
    ) -> None:
        self.scene = scene
        self.runtime = CosmosDreamsTickRuntime(backend, scene, seed=seed, measure_latency=True)
        self.controller = AgiBotKeyboardController(scene)
        self.resampler = AgiBotKeyboardResampler(fps=scene.fps, start_time=time.perf_counter())
        self.queue = BoundedRGBFrameQueue(max_frames=queue_frames)
        self.recorder = CosmosDreamsLatencyRecorder()
        self.video_track = _make_video_track(
            self.queue,
            fps=scene.fps,
            on_chunk_start=self._notify_chunk_streaming,
        )
        self.replay_actions = replay_actions
        self._replay_index = 0
        self._channel = None
        self._generation_id = 0
        self._first_action = asyncio.Event()
        self._pending_arrivals: deque[float] = deque()
        self._worker = asyncio.create_task(self._generation_worker(self._generation_id))
        self._lifecycle_lock = asyncio.Lock()
        self._closed = False

    def bind_channel(self, channel) -> None:
        self._channel = channel

        @channel.on("message")
        def on_message(message: Any) -> None:
            asyncio.create_task(self._handle_message(message))

    def _send(self, payload: dict[str, Any]) -> None:
        if self._channel is not None and getattr(self._channel, "readyState", "") == "open":
            self._channel.send(json.dumps(payload, separators=(",", ":")))

    def _notify_chunk_streaming(self, generation_id: int, chunk_index: int) -> None:
        self._send(
            {
                "type": "chunk_streaming",
                "generation_id": generation_id,
                "chunk_index": chunk_index,
            }
        )

    async def _handle_message(self, message: Any) -> None:
        try:
            await self._handle_message_impl(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Cosmos-Dreams control message failed")
            self._send(
                {
                    "type": "error",
                    "message": str(exc),
                    "session_reset_required": True,
                }
            )

    async def _handle_message_impl(self, message: Any) -> None:
        try:
            payload = json.loads(message) if isinstance(message, str) else {}
        except json.JSONDecodeError:
            self._send({"type": "error", "message": "Invalid JSON control message."})
            return
        message_type = str(payload.get("type", "")).lower()
        if message_type == "heartbeat":
            self._send({"type": "heartbeat", "server_time": time.time()})
            return
        if message_type == "reset":
            await self.reset()
            self._send(
                {
                    "type": "reset_done",
                    "session_id": self.runtime.session_id,
                    "generation_id": self._generation_id,
                }
            )
            return
        if message_type == "disconnect":
            await self.close()
            return
        if message_type == "presented":
            generation_id = payload.get("generation_id")
            if (
                isinstance(generation_id, bool)
                or not isinstance(generation_id, int)
                or generation_id != self._generation_id
            ):
                # A frame may already be inside the browser or encoder when a
                # reset clears the queue. Never let its late acknowledgement
                # mark the new generation's identically numbered chunk.
                return
            chunk_index = int(payload.get("chunk_index", -1))
            if chunk_index >= 0:
                try:
                    self.recorder.mark_presented(chunk_index)
                    timing = self.recorder.record(chunk_index)
                except KeyError:
                    pass
                else:
                    self._send(
                        {
                            "type": "presentation_timing",
                            "generation_id": generation_id,
                            "timing": timing,
                        }
                    )
            return
        if message_type != "action":
            self._send({"type": "error", "message": "Expected action, reset, heartbeat, or disconnect."})
            return
        action = payload.get("action", payload)
        event = str(action.get("event", "")) if isinstance(action, dict) else ""
        key = str(action.get("key", "")) if isinstance(action, dict) else ""
        arrival = time.perf_counter()
        if not self.resampler.on_edge(arrival_time=arrival, event=event, key=key):
            self._send({"type": "error", "message": f"Unsupported keyboard event {event!r}/{key!r}."})
            return
        self._pending_arrivals.append(arrival)
        self._first_action.set()

    def _next_replay_action(self) -> torch.Tensor:
        if not self.replay_actions:
            raise RuntimeError("Replay mode has no actions.")
        if self._replay_index >= len(self.replay_actions):
            raise EOFError("Replay action fixture is exhausted.")
        action = self.replay_actions[self._replay_index]
        self._replay_index += 1
        return action

    def _report_worker_failure(self, generation_id: int, exc: Exception) -> None:
        logger.error(
            "Cosmos-Dreams generation worker failed in generation %d",
            generation_id,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        self._send(
            {
                "type": "error",
                "generation_id": generation_id,
                "message": str(exc),
                "session_reset_required": True,
            }
        )

    async def _generation_worker(self, generation_id: int) -> None:
        try:
            await self._first_action.wait()
            self.resampler.next_chunk_start = time.perf_counter()
            while not self._closed:
                snapshot = None
                timing_started = False
                tick_committed = False
                chunk_index = -1
                try:
                    chunk_duration = ACTION_STEPS_PER_TICK * self.resampler.dt
                    trigger = self.resampler.next_chunk_start
                    delay = trigger - time.perf_counter()
                    if delay > 0:
                        await asyncio.sleep(delay)
                    if self._closed:
                        return
                    now = time.perf_counter()
                    if now - trigger > chunk_duration:
                        self.resampler.next_chunk_start = now
                    segments, frame_times = self.resampler.sample_chunk(ACTION_STEPS_PER_TICK)
                    chunk_end = self.resampler.next_chunk_start
                    arrivals: list[float] = []
                    while self._pending_arrivals and self._pending_arrivals[0] <= chunk_end:
                        arrivals.append(self._pending_arrivals.popleft())
                    key_arrival = min(arrivals) if arrivals else frame_times[0] - self.resampler.dt
                    chunk_index = 0 if not self.runtime.started else (self.runtime.next_frame_idx - 1) // 4
                    snapshot = self.controller.snapshot()
                    self.recorder.begin(
                        chunk_index=chunk_index,
                        key_arrival_time=key_arrival,
                        request_dispatch_time=time.perf_counter(),
                    )
                    timing_started = True
                    if self.replay_actions is None:
                        action = self.controller.build_action_chunk(segments, frame_times)
                    else:
                        action = self._next_replay_action()
                    result = await self.runtime.tick_async(action, key_arrival_time=key_arrival)
                    tick_committed = True
                    self.recorder.attach_tick(result)
                    enqueued = await self.queue.enqueue_chunk(
                        result,
                        generation_id=generation_id,
                        on_encoder_handoff=partial(self.recorder.mark_encoder_handoff, result.chunk_index),
                    )
                    self.recorder.mark_enqueue_done(result.chunk_index)
                    timing = self.recorder.record(result.chunk_index)
                    self._send(
                        {
                            "type": "chunk_done",
                            "generation_id": generation_id,
                            "chunk_index": result.chunk_index,
                            "frame_idx": result.frame_idx,
                            "next_frame_idx": result.next_frame_idx,
                            "frames": enqueued,
                            "queue_depth": self.queue.qsize(),
                            "timing": timing,
                        }
                    )
                except EOFError:
                    self._send({"type": "replay_done"})
                    return
                except Exception as exc:
                    if snapshot is not None and not tick_committed:
                        self.controller.restore(snapshot)
                    if timing_started:
                        self.recorder.discard(chunk_index)
                    self._report_worker_failure(generation_id, exc)
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._report_worker_failure(generation_id, exc)

    async def reset(self) -> None:
        async with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("Cannot reset a closed Cosmos-Dreams WebRTC session.")
            previous_generation = self._generation_id
            # Advance before the first await so late browser acknowledgements
            # are rejected throughout reset, including frames already handed
            # to the encoder before the queue is cleared.
            self._generation_id += 1
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker
            summary = self.recorder.warm_summary(playback_fps=self.scene.fps)
            logger.info(
                "Cosmos-Dreams generation %d latency summary before reset: %s",
                previous_generation,
                summary,
            )
            self.recorder = CosmosDreamsLatencyRecorder()
            await self.runtime.reset_async()
            self.controller.reset()
            self.resampler.reset(start_time=time.perf_counter())
            self.queue.clear()
            self._pending_arrivals.clear()
            self._replay_index = 0
            self._first_action = asyncio.Event()
            self._worker = asyncio.create_task(self._generation_worker(self._generation_id))

    async def close(self) -> None:
        async with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker
            with contextlib.suppress(Exception):
                await self.runtime.close_async()
            await self.queue.close()
            summary = self.recorder.warm_summary(playback_fps=self.scene.fps)
            logger.info(
                "Cosmos-Dreams generation %d final latency summary: %s",
                self._generation_id,
                summary,
            )


class SingleUserWebRTCManager:
    def __init__(self, *, backend, scene, seed: int, queue_frames: int, replay_actions) -> None:
        self.backend = backend
        self.scene = scene
        self.seed = seed
        self.queue_frames = queue_frames
        self.replay_actions = replay_actions
        self.pc = None
        self.session: CosmosDreamsWebRTCSession | None = None

    async def offer(self, request):
        web, RTCPeerConnection, RTCRtpSender, RTCSessionDescription, _, _, _ = _require_webrtc()
        if self.session is not None:
            raise web.HTTPConflict(text="The v1 Cosmos-Dreams demo supports one active user.")
        payload = await request.json()
        pc = RTCPeerConnection()
        session = CosmosDreamsWebRTCSession(
            backend=self.backend,
            scene=self.scene,
            seed=self.seed,
            queue_frames=self.queue_frames,
            replay_actions=self.replay_actions,
        )
        transceiver = pc.addTransceiver(session.video_track, direction="sendonly")
        capabilities = RTCRtpSender.getCapabilities("video")
        preferred = [codec for codec in capabilities.codecs if codec.mimeType.lower() == "video/h264"]
        if preferred:
            transceiver.setCodecPreferences(preferred)

        @pc.on("datachannel")
        def on_datachannel(channel) -> None:
            session.bind_channel(channel)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            if pc.connectionState in {"failed", "closed", "disconnected"}:
                await self.close()

        await pc.setRemoteDescription(RTCSessionDescription(sdp=payload["sdp"], type=payload["type"]))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        self.pc = pc
        self.session = session
        return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

    async def close(self) -> None:
        session, pc = self.session, self.pc
        self.session = None
        self.pc = None
        if session is not None:
            await session.close()
        if pc is not None:
            await pc.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--deploy-config", default="vllm_omni/deploy/cosmos_dreams.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--queue-frames", type=int, default=32)
    parser.add_argument("--replay-actions", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    web, *_ = _require_webrtc()
    from vllm_omni.entrypoints.async_omni import AsyncOmni

    scene = load_scene_bundle(args.scene)
    backend = AsyncOmni(
        model=args.model,
        model_class_name="CosmosDreamsPipeline",
        deploy_config=args.deploy_config,
        enforce_eager=True,
    )
    replay_actions = None if args.replay_actions is None else load_replay_actions(args.replay_actions)
    manager = SingleUserWebRTCManager(
        backend=backend,
        scene=scene,
        seed=args.seed,
        queue_frames=args.queue_frames,
        replay_actions=replay_actions,
    )
    web_root = Path(__file__).with_name("web")
    app = web.Application()
    app.router.add_post("/offer", manager.offer)
    app.router.add_get("/", lambda _: web.FileResponse(web_root / "index.html"))
    app.router.add_static("/static", web_root)

    async def cleanup(_app) -> None:
        await manager.close()
        await asyncio.to_thread(backend.shutdown)

    app.on_cleanup.append(cleanup)
    logging.basicConfig(level=logging.INFO)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
