# SPDX-License-Identifier: Apache-2.0
"""AgiBot keyboard control and raw action construction for the WebRTC demo.

The controller deliberately stops at the exported artifact boundary: it emits
raw physical ``float32[16, 29]`` actions and never reads normalizer statistics
or pads actions to the model's 64-dimensional internal width.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import torch

AGIBOT_ACTION_DIM = 29
AGIBOT_DOMAIN_ID = 15
AGIBOT_EMBODIMENT = "agibotworld"
ACTION_COORDINATE_VERSION = "agibotworld.backward_framewise.rot6d.opencv.v1"
ACTION_STEPS_PER_TICK = 16

PoseTarget = Literal["head", "right", "left"]
PoseSegment = tuple[float, float, frozenset[str]]

_TARGET_KEYS: dict[str, PoseTarget] = {"1": "left", "2": "right", "3": "head"}
_KEY_ALIASES = {
    "arrowup": "i",
    "arrowdown": "k",
    "arrowleft": "j",
    "arrowright": "l",
    "left": "1",
    "right": "2",
    "head": "3",
}
AGIBOT_SUPPORTED_KEYS = frozenset(
    {
        "1",
        "2",
        "3",
        "w",
        "s",
        "a",
        "d",
        "r",
        "f",
        "i",
        "k",
        "j",
        "l",
        "u",
        "o",
        "space",
    }
)


def normalize_agibot_key(key: str) -> str:
    raw_key = str(key).lower()
    normalized = raw_key.strip()
    if raw_key == " ":
        normalized = "space"
    return _KEY_ALIASES.get(normalized, normalized)


def _as_transform(value: torch.Tensor | Sequence[Sequence[float]], name: str) -> torch.Tensor:
    transform = torch.as_tensor(value, dtype=torch.float32).detach().cpu().clone()
    if tuple(transform.shape) != (4, 4):
        raise ValueError(f"{name} must have shape [4,4], got {tuple(transform.shape)}.")
    if not torch.isfinite(transform).all():
        raise ValueError(f"{name} contains NaN or Inf values.")
    expected_bottom = torch.tensor([0.0, 0.0, 0.0, 1.0])
    if not torch.allclose(transform[3], expected_bottom, atol=1e-5, rtol=0):
        raise ValueError(f"{name} must be a homogeneous rigid transform.")
    rotation = transform[:3, :3]
    identity = torch.eye(3, dtype=torch.float32)
    if not torch.allclose(rotation.T @ rotation, identity, atol=1e-4, rtol=1e-4):
        raise ValueError(f"{name} rotation must be orthonormal.")
    if not torch.allclose(torch.det(rotation), torch.tensor(1.0), atol=1e-4, rtol=1e-4):
        raise ValueError(f"{name} rotation must have determinant +1.")
    return transform


def _seed_resolution(seed_rgb: Any) -> tuple[int, int]:
    if isinstance(seed_rgb, torch.Tensor):
        shape = tuple(seed_rgb.shape)
    else:
        shape = tuple(getattr(seed_rgb, "shape", ()))
    if len(shape) == 4 and shape[0] == 1:
        shape = shape[1:]
    if len(shape) == 3:
        if shape[-1] in (3, 4):
            return int(shape[0]), int(shape[1])
        if shape[0] in (3, 4):
            return int(shape[1]), int(shape[2])
    size = getattr(seed_rgb, "size", None)
    if isinstance(size, tuple) and len(size) == 2:
        return int(size[1]), int(size[0])
    raise ValueError("seed_rgb must expose RGB tensor/array shape [H,W,3], [3,H,W], or PIL-style size.")


@dataclass(frozen=True, slots=True)
class AgiBotControllerLimits:
    """Rate and workspace limits applied to model-only Cartesian targets."""

    linear_velocity_m_s: float = 0.12
    angular_velocity_rad_s: float = math.radians(30.0)
    workspace_min_xyz: tuple[float, float, float] | None = None
    workspace_max_xyz: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.linear_velocity_m_s) or self.linear_velocity_m_s <= 0:
            raise ValueError("linear_velocity_m_s must be positive and finite.")
        if not math.isfinite(self.angular_velocity_rad_s) or self.angular_velocity_rad_s <= 0:
            raise ValueError("angular_velocity_rad_s must be positive and finite.")
        if (self.workspace_min_xyz is None) != (self.workspace_max_xyz is None):
            raise ValueError("workspace_min_xyz and workspace_max_xyz must be supplied together.")
        if self.workspace_min_xyz is not None and self.workspace_max_xyz is not None:
            if len(self.workspace_min_xyz) != 3 or len(self.workspace_max_xyz) != 3:
                raise ValueError("Workspace bounds must contain exactly three coordinates.")
            if any(lo >= hi for lo, hi in zip(self.workspace_min_xyz, self.workspace_max_xyz, strict=True)):
                raise ValueError("Every workspace minimum must be smaller than its maximum.")


@dataclass(frozen=True, slots=True)
class AgiBotSceneState:
    """Immutable scene bundle used to create one interactive model session.

    The three transforms must already be calibrated into the training/OpenCV
    action coordinate system. A physical-robot integration should provide them
    through its authoritative FK implementation; this visual demo never closes
    a hardware control loop through the generated video.
    """

    seed_rgb: Any
    prompt: str
    fps: float
    head_transform: torch.Tensor | Sequence[Sequence[float]] | None = None
    right_wrist_transform: torch.Tensor | Sequence[Sequence[float]] | None = None
    left_wrist_transform: torch.Tensor | Sequence[Sequence[float]] | None = None
    right_gripper: float = 1.0
    left_gripper: float = 1.0
    domain_id: int = AGIBOT_DOMAIN_ID
    embodiment: str = AGIBOT_EMBODIMENT
    limits: AgiBotControllerLimits = field(default_factory=AgiBotControllerLimits)
    joint_state: torch.Tensor | None = None
    action_coordinate_version: str = ACTION_COORDINATE_VERSION

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValueError("AgiBotSceneState prompt must be non-empty.")
        if not math.isfinite(float(self.fps)) or float(self.fps) <= 0:
            raise ValueError("AgiBotSceneState fps must be positive and finite.")
        if int(self.domain_id) != AGIBOT_DOMAIN_ID or self.embodiment != AGIBOT_EMBODIMENT:
            raise ValueError(
                "The v1 Cosmos-Dreams controller is pinned to agibotworld/domain 15; "
                f"got {self.embodiment!r}/domain {self.domain_id}."
            )
        if self.action_coordinate_version != ACTION_COORDINATE_VERSION:
            raise ValueError(
                "Unsupported AgiBot action coordinate contract: "
                f"{self.action_coordinate_version!r}; expected {ACTION_COORDINATE_VERSION!r}."
            )
        for name, value in (("right_gripper", self.right_gripper), ("left_gripper", self.left_gripper)):
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be an absolute open fraction in [0,1].")
        if self.joint_state is not None:
            joint_state = torch.as_tensor(self.joint_state, dtype=torch.float32).detach().cpu().clone()
            if joint_state.ndim != 1 or not torch.isfinite(joint_state).all():
                raise ValueError("joint_state must be a finite one-dimensional tensor.")
            object.__setattr__(self, "joint_state", joint_state)
        transforms = (self.head_transform, self.right_wrist_transform, self.left_wrist_transform)
        if any(transform is None for transform in transforms):
            if not all(transform is None for transform in transforms):
                raise ValueError("Supply all three calibrated transforms or omit all three and provide joint_state.")
            if self.joint_state is None:
                raise ValueError("AgiBotSceneState requires calibrated transforms or an authoritative joint_state.")
        else:
            object.__setattr__(self, "head_transform", _as_transform(self.head_transform, "head_transform"))
            object.__setattr__(
                self,
                "right_wrist_transform",
                _as_transform(self.right_wrist_transform, "right_wrist_transform"),
            )
            object.__setattr__(
                self,
                "left_wrist_transform",
                _as_transform(self.left_wrist_transform, "left_wrist_transform"),
            )
        _seed_resolution(self.seed_rgb)

    @property
    def resolution(self) -> tuple[int, int]:
        return _seed_resolution(self.seed_rgb)


class AgiBotKinematics(Protocol):
    """Optional calibrated IK/FK seam for joint-space controller variants."""

    def inverse(
        self,
        *,
        targets: dict[PoseTarget, torch.Tensor],
        joint_state: torch.Tensor,
    ) -> torch.Tensor: ...

    def forward(self, joint_state: torch.Tensor) -> dict[PoseTarget, torch.Tensor]: ...


@dataclass(slots=True)
class AgiBotKeyboardState:
    pressed_keys: set[str] = field(default_factory=set)
    _press_order: dict[str, int] = field(default_factory=dict)
    _counter: int = 0

    def apply_event(self, *, event: str, key: str) -> bool:
        normalized_key = normalize_agibot_key(key)
        if normalized_key not in AGIBOT_SUPPORTED_KEYS:
            return False
        normalized_event = str(event).strip().lower()
        if normalized_event == "keydown":
            if normalized_key not in self.pressed_keys:
                self._counter += 1
                self._press_order[normalized_key] = self._counter
            self.pressed_keys.add(normalized_key)
            return True
        if normalized_event == "keyup":
            self.pressed_keys.discard(normalized_key)
            self._press_order.pop(normalized_key, None)
            return True
        return False

    def _latest(self, pair: tuple[str, str]) -> str | None:
        active = [key for key in pair if key in self.pressed_keys]
        if not active:
            return None
        return max(active, key=lambda key: self._press_order.get(key, -1))

    def effective_keys(self) -> frozenset[str]:
        effective = {key for key in _TARGET_KEYS if key in self.pressed_keys}
        if "space" in self.pressed_keys:
            effective.add("space")
        for pair in (("w", "s"), ("a", "d"), ("r", "f"), ("i", "k"), ("j", "l"), ("u", "o")):
            latest = self._latest(pair)
            if latest is not None:
                effective.add(latest)
        return frozenset(effective)


class AgiBotKeyboardResampler:
    """Turn sparse browser key edges into a pixel-rate piecewise timeline."""

    def __init__(self, *, fps: float, start_time: float = 0.0) -> None:
        if not math.isfinite(float(fps)) or float(fps) <= 0:
            raise ValueError("fps must be positive and finite.")
        self.fps = float(fps)
        self.dt = 1.0 / self.fps
        self.next_chunk_start = float(start_time)
        self._events: deque[tuple[float, str, str]] = deque()
        self._state = AgiBotKeyboardState()

    def on_edge(self, *, arrival_time: float, event: str, key: str) -> bool:
        normalized_key = normalize_agibot_key(key)
        if normalized_key not in AGIBOT_SUPPORTED_KEYS:
            return False
        if self._events and float(arrival_time) < self._events[-1][0]:
            raise ValueError("Keyboard edge arrival times must be monotonic.")
        if str(event).strip().lower() not in {"keydown", "keyup"}:
            return False
        self._events.append((float(arrival_time), str(event), normalized_key))
        return True

    def sample_chunk(self, num_frames: int = ACTION_STEPS_PER_TICK) -> tuple[list[PoseSegment], list[float]]:
        if num_frames <= 0:
            raise ValueError("num_frames must be positive.")
        chunk_start = self.next_chunk_start
        chunk_end = chunk_start + num_frames * self.dt
        while self._events and self._events[0][0] < chunk_start:
            _, event, key = self._events.popleft()
            self._state.apply_event(event=event, key=key)

        segments: list[PoseSegment] = []
        segment_start = chunk_start
        keys = self._state.effective_keys()
        while self._events and self._events[0][0] <= chunk_end:
            event_time, event, key = self._events.popleft()
            if event_time > segment_start:
                segments.append((segment_start, event_time, keys))
            self._state.apply_event(event=event, key=key)
            keys = self._state.effective_keys()
            segment_start = event_time
        if segment_start < chunk_end:
            segments.append((segment_start, chunk_end, keys))
        elif not segments:
            segments.append((chunk_start, chunk_end, keys))

        frame_times = [chunk_start + (index + 1) * self.dt for index in range(num_frames)]
        self.next_chunk_start = chunk_end
        return segments, frame_times

    def reset(self, *, start_time: float = 0.0) -> None:
        self.next_chunk_start = float(start_time)
        self._events.clear()
        self._state = AgiBotKeyboardState()


@dataclass(frozen=True, slots=True)
class AgiBotControllerSnapshot:
    head_transform: torch.Tensor
    right_wrist_transform: torch.Tensor
    left_wrist_transform: torch.Tensor
    right_gripper: float
    left_gripper: float
    selected_target: PoseTarget
    held_keys: frozenset[str]
    joint_state: torch.Tensor | None


def _axis_rotation(axis: Literal["x", "y", "z"], angle: float) -> torch.Tensor:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    if axis == "x":
        values = ((1.0, 0.0, 0.0), (0.0, cosine, -sine), (0.0, sine, cosine))
    elif axis == "y":
        values = ((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine))
    else:
        values = ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))
    return torch.tensor(values, dtype=torch.float32)


def _rigid_inverse(transform: torch.Tensor) -> torch.Tensor:
    inverse = torch.eye(4, dtype=torch.float32)
    inverse[:3, :3] = transform[:3, :3].T
    inverse[:3, 3] = -(inverse[:3, :3] @ transform[:3, 3])
    return inverse


def _rotation_to_column_rot6d(rotation: torch.Tensor) -> torch.Tensor:
    return rotation[:, :2].T.reshape(6)


class AgiBotKeyboardController:
    """Integrate held keys into calibrated transforms and raw AgiBot actions.

    Key map (in the selected target's local OpenCV frame): ``w/s`` forward and
    back, ``a/d`` left and right, ``r/f`` up and down, ``i/k`` pitch,
    ``j/l`` yaw, and ``u/o`` roll. ``1``/``2``/``3`` select left wrist, right
    wrist, or head. A rising ``space`` edge toggles the selected wrist gripper.
    """

    def __init__(
        self,
        scene: AgiBotSceneState,
        *,
        kinematics: AgiBotKinematics | None = None,
        selected_target: PoseTarget = "right",
    ) -> None:
        self.scene = scene
        self.kinematics = kinematics
        self._initial_selected_target = selected_target
        self._right_gripper = float(scene.right_gripper)
        self._left_gripper = float(scene.left_gripper)
        self._selected_target: PoseTarget = selected_target
        self._held_keys = frozenset[str]()
        self._joint_state = None if scene.joint_state is None else scene.joint_state.clone()
        if kinematics is not None:
            if self._joint_state is None:
                raise ValueError("AgiBotKinematics requires scene.joint_state.")
            self._set_transforms(kinematics.forward(self._joint_state))
        else:
            if (
                scene.head_transform is None
                or scene.right_wrist_transform is None
                or scene.left_wrist_transform is None
            ):
                raise ValueError("A joint-only AgiBotSceneState requires an AgiBotKinematics implementation.")
            self._head = scene.head_transform.clone()
            self._right = scene.right_wrist_transform.clone()
            self._left = scene.left_wrist_transform.clone()

    @property
    def selected_target(self) -> PoseTarget:
        return self._selected_target

    def snapshot(self) -> AgiBotControllerSnapshot:
        return AgiBotControllerSnapshot(
            head_transform=self._head.clone(),
            right_wrist_transform=self._right.clone(),
            left_wrist_transform=self._left.clone(),
            right_gripper=self._right_gripper,
            left_gripper=self._left_gripper,
            selected_target=self._selected_target,
            held_keys=self._held_keys,
            joint_state=None if self._joint_state is None else self._joint_state.clone(),
        )

    def restore(self, snapshot: AgiBotControllerSnapshot) -> None:
        self._head = snapshot.head_transform.clone()
        self._right = snapshot.right_wrist_transform.clone()
        self._left = snapshot.left_wrist_transform.clone()
        self._right_gripper = float(snapshot.right_gripper)
        self._left_gripper = float(snapshot.left_gripper)
        self._selected_target = snapshot.selected_target
        self._held_keys = frozenset(snapshot.held_keys)
        self._joint_state = None if snapshot.joint_state is None else snapshot.joint_state.clone()

    def reset(self) -> None:
        self._right_gripper = float(self.scene.right_gripper)
        self._left_gripper = float(self.scene.left_gripper)
        self._selected_target = self._initial_selected_target
        self._held_keys = frozenset()
        self._joint_state = None if self.scene.joint_state is None else self.scene.joint_state.clone()
        if self.kinematics is not None and self._joint_state is not None:
            self._set_transforms(self.kinematics.forward(self._joint_state))
        else:
            if (
                self.scene.head_transform is None
                or self.scene.right_wrist_transform is None
                or self.scene.left_wrist_transform is None
            ):
                raise RuntimeError("A joint-only AgiBot scene lost its kinematics implementation.")
            self._head = self.scene.head_transform.clone()
            self._right = self.scene.right_wrist_transform.clone()
            self._left = self.scene.left_wrist_transform.clone()

    def _transforms(self) -> dict[PoseTarget, torch.Tensor]:
        return {"head": self._head, "right": self._right, "left": self._left}

    def _set_transforms(self, transforms: dict[PoseTarget, torch.Tensor]) -> None:
        required = {"head", "right", "left"}
        if set(transforms) != required:
            raise ValueError(f"AgiBot kinematics must return transforms {sorted(required)}, got {sorted(transforms)}.")
        self._head = _as_transform(transforms["head"], "kinematics head transform")
        self._right = _as_transform(transforms["right"], "kinematics right transform")
        self._left = _as_transform(transforms["left"], "kinematics left transform")

    def _toggle_gripper(self) -> None:
        if self._selected_target == "right":
            self._right_gripper = 0.0 if self._right_gripper >= 0.5 else 1.0
        elif self._selected_target == "left":
            self._left_gripper = 0.0 if self._left_gripper >= 0.5 else 1.0

    def _clamp_workspace(self, pose: torch.Tensor) -> None:
        limits = self.scene.limits
        if limits.workspace_min_xyz is None or limits.workspace_max_xyz is None:
            return
        minimum = torch.tensor(limits.workspace_min_xyz, dtype=torch.float32)
        maximum = torch.tensor(limits.workspace_max_xyz, dtype=torch.float32)
        pose[:3, 3] = pose[:3, 3].clamp(min=minimum, max=maximum)

    def _integrate(self, keys: frozenset[str], duration: float) -> None:
        rising = keys - self._held_keys
        for key in ("1", "2", "3"):
            if key in rising:
                self._selected_target = _TARGET_KEYS[key]
        if "space" in rising:
            self._toggle_gripper()
        self._held_keys = keys
        if duration <= 0:
            return

        translation = torch.tensor(
            [
                float("d" in keys) - float("a" in keys),
                float("f" in keys) - float("r" in keys),
                float("w" in keys) - float("s" in keys),
            ],
            dtype=torch.float32,
        )
        rotation_rate = self.scene.limits.angular_velocity_rad_s
        pitch = (float("k" in keys) - float("i" in keys)) * rotation_rate * duration
        yaw = (float("l" in keys) - float("j" in keys)) * rotation_rate * duration
        roll = (float("o" in keys) - float("u" in keys)) * rotation_rate * duration
        if torch.count_nonzero(translation) == 0 and pitch == 0 and yaw == 0 and roll == 0:
            return

        norm = float(torch.linalg.vector_norm(translation))
        if norm > 1.0:
            translation /= norm
        delta = torch.eye(4, dtype=torch.float32)
        delta[:3, 3] = translation * (self.scene.limits.linear_velocity_m_s * duration)
        delta[:3, :3] = _axis_rotation("x", pitch) @ _axis_rotation("y", yaw) @ _axis_rotation("z", roll)
        targets = {name: transform.clone() for name, transform in self._transforms().items()}
        targets[self._selected_target] = targets[self._selected_target] @ delta
        self._clamp_workspace(targets[self._selected_target])
        if self.kinematics is not None:
            if self._joint_state is None:
                raise RuntimeError("AgiBot kinematics state was not initialized.")
            self._joint_state = self.kinematics.inverse(targets=targets, joint_state=self._joint_state)
            self._set_transforms(self.kinematics.forward(self._joint_state))
        else:
            self._set_transforms(targets)

    @staticmethod
    def _segment_keys_at(segments: Sequence[PoseSegment], timestamp: float) -> frozenset[str]:
        for start, end, keys in segments:
            if start < timestamp <= end or (timestamp == start == end):
                return keys
        if segments and math.isclose(timestamp, segments[0][0]):
            return segments[0][2]
        raise ValueError(f"Frame timestamp {timestamp} is not covered by the controller segments.")

    def _integrate_interval(self, segments: Sequence[PoseSegment], start: float, end: float) -> None:
        cursor = start
        for segment_start, segment_end, keys in segments:
            overlap_start = max(cursor, segment_start)
            overlap_end = min(end, segment_end)
            if overlap_end > overlap_start:
                self._integrate(keys, overlap_end - overlap_start)
                cursor = overlap_end
            elif math.isclose(overlap_end, overlap_start) and math.isclose(overlap_start, segment_start):
                # Process zero-duration rising selection/toggle edges exactly at
                # a frame boundary without adding any Cartesian motion.
                self._integrate(keys, 0.0)
            if cursor >= end:
                break
        if cursor < end and not math.isclose(cursor, end):
            raise ValueError(f"Controller segments do not cover interval [{start}, {end}].")

    @staticmethod
    def _pose_delta(previous: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        delta = _rigid_inverse(previous) @ current
        return torch.cat([delta[:3, 3], _rotation_to_column_rot6d(delta[:3, :3])])

    def build_action_chunk(
        self,
        segments: Sequence[PoseSegment],
        frame_times: Sequence[float],
    ) -> torch.Tensor:
        """Build and commit one raw ``float32[16,29]`` controller chunk."""

        if len(frame_times) != ACTION_STEPS_PER_TICK:
            raise ValueError(
                f"Cosmos-Dreams ticks require exactly {ACTION_STEPS_PER_TICK} pixel-rate steps, got {len(frame_times)}."
            )
        if not segments:
            raise ValueError("Controller segments must not be empty.")
        if any(end < start for start, end, _ in segments):
            raise ValueError("Controller segments must have non-negative durations.")
        if any(next_time <= current for current, next_time in zip(frame_times, frame_times[1:])):
            raise ValueError("frame_times must be strictly increasing.")

        rows: list[torch.Tensor] = []
        interval_start = float(segments[0][0])
        for frame_time in frame_times:
            previous = {name: transform.clone() for name, transform in self._transforms().items()}
            self._integrate_interval(segments, interval_start, float(frame_time))
            current = self._transforms()
            row = torch.cat(
                [
                    self._pose_delta(previous["head"], current["head"]),
                    self._pose_delta(previous["right"], current["right"]),
                    torch.tensor([self._right_gripper], dtype=torch.float32),
                    self._pose_delta(previous["left"], current["left"]),
                    torch.tensor([self._left_gripper], dtype=torch.float32),
                ]
            )
            if tuple(row.shape) != (AGIBOT_ACTION_DIM,):
                raise RuntimeError(f"Internal AgiBot action layout error: got {tuple(row.shape)}.")
            rows.append(row)
            interval_start = float(frame_time)
        return torch.stack(rows).to(dtype=torch.float32)


@dataclass(frozen=True, slots=True)
class KeyScheduleSegment:
    keys: frozenset[str]
    frames: int

    def __post_init__(self) -> None:
        if self.frames <= 0:
            raise ValueError("Key schedule segments must contain at least one frame.")
        unknown = self.keys - AGIBOT_SUPPORTED_KEYS
        if unknown:
            raise ValueError(f"Unknown AgiBot key schedule keys: {sorted(unknown)}.")


def parse_key_schedule(value: str) -> list[KeyScheduleSegment]:
    """Parse ``right+w:8,space:1,none:7`` into deterministic held states."""

    schedule: list[KeyScheduleSegment] = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        key_text, separator, frame_text = item.rpartition(":")
        if not separator:
            raise ValueError(f"Key schedule item {item!r} must end in ':frames'.")
        try:
            frames = int(frame_text)
        except ValueError as exc:
            raise ValueError(f"Invalid key schedule frame count in {item!r}.") from exc
        raw_keys = [] if key_text.strip().lower() in {"", "idle", "none"} else key_text.split("+")
        keys = frozenset(normalize_agibot_key(key) for key in raw_keys)
        schedule.append(KeyScheduleSegment(keys=keys, frames=frames))
    if not schedule:
        raise ValueError("Key schedule must contain at least one segment.")
    return schedule


def schedule_to_controller_inputs(
    schedule: Iterable[KeyScheduleSegment],
    *,
    fps: float,
    start_time: float = 0.0,
) -> tuple[list[PoseSegment], list[float]]:
    """Use the live resampler itself to materialize a deterministic schedule."""

    schedule = list(schedule)
    num_frames = sum(segment.frames for segment in schedule)
    resampler = AgiBotKeyboardResampler(fps=fps, start_time=start_time)
    held: frozenset[str] = frozenset()
    cursor = float(start_time)
    dt = 1.0 / float(fps)
    for segment in schedule:
        for key in sorted(held - segment.keys):
            resampler.on_edge(arrival_time=cursor, event="keyup", key=key)
        for key in sorted(segment.keys - held):
            resampler.on_edge(arrival_time=cursor, event="keydown", key=key)
        held = segment.keys
        cursor += segment.frames * dt
    return resampler.sample_chunk(num_frames=num_frames)


def build_scheduled_action_chunk(
    controller: AgiBotKeyboardController,
    schedule: Iterable[KeyScheduleSegment] | str,
) -> torch.Tensor:
    parsed = parse_key_schedule(schedule) if isinstance(schedule, str) else list(schedule)
    segments, frame_times = schedule_to_controller_inputs(parsed, fps=controller.scene.fps)
    return controller.build_action_chunk(segments, frame_times)


__all__ = [
    "ACTION_COORDINATE_VERSION",
    "ACTION_STEPS_PER_TICK",
    "AGIBOT_ACTION_DIM",
    "AGIBOT_DOMAIN_ID",
    "AGIBOT_EMBODIMENT",
    "AGIBOT_SUPPORTED_KEYS",
    "AgiBotControllerLimits",
    "AgiBotControllerSnapshot",
    "AgiBotKeyboardController",
    "AgiBotKeyboardResampler",
    "AgiBotKeyboardState",
    "AgiBotKinematics",
    "AgiBotSceneState",
    "KeyScheduleSegment",
    "PoseSegment",
    "build_scheduled_action_chunk",
    "normalize_agibot_key",
    "parse_key_schedule",
    "schedule_to_controller_inputs",
]
