# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Camera ownership, ordering and failure agreement with real Gloo collectives."""

from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm_omni.diffusion.distributed.group_coordinator import GroupCoordinator
from vllm_omni.diffusion.models.cosmos3_nano_sim_auto.parallel import for_each_camera_tensor

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _camera_tensor(camera: int) -> torch.Tensor:
    # Non-contiguous results, like a first-frame slice, must retain their values.
    return (torch.arange(12).reshape(3, 4) + camera * 100).t()


def test_camera_vae_single_rank() -> None:
    calls: list[int] = []
    outputs: list[torch.Tensor] = []

    def compute(camera: int) -> torch.Tensor:
        calls.append(camera)
        return _camera_tensor(camera)

    for_each_camera_tensor(7, compute, lambda _, tensor: outputs.append(tensor), None)
    assert calls == list(range(7))
    for camera, output in enumerate(outputs):
        assert torch.equal(output, _camera_tensor(camera))
    with pytest.raises(ValueError, match="at least one camera"):
        for_each_camera_tensor(0, compute, lambda *_: None, None)


def _distributed_camera_worker(rank: int, rendezvous: str) -> None:
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=30))
    try:
        # Use the real coordinator collective with CPU groups, without requiring
        # an accelerator or constructing unrelated denoiser process groups.
        group = GroupCoordinator.__new__(GroupCoordinator)
        group.world_size = 4
        group.rank_in_group = rank
        group.device = torch.device("cpu")
        group.cpu_group = group.device_group = dist.group.WORLD
        for cameras in (1, 3, 4, 7, 11):
            calls: list[int] = []
            outputs: list[torch.Tensor] = []

            def compute(camera: int) -> torch.Tensor:
                calls.append(camera)
                return _camera_tensor(camera)

            def consume(camera: int, tensor: torch.Tensor) -> None:
                assert camera == len(outputs)
                outputs.append(tensor)

            for_each_camera_tensor(cameras, compute, consume, group)
            assert calls == list(range(rank, cameras, 4))
            assert len(outputs) == cameras
            for camera, output in enumerate(outputs):
                assert torch.equal(output, _camera_tensor(camera))

        for failure in ("producer", "local_shape", "remote_shape", "remote_dtype"):
            published: list[int] = []

            def fail(camera: int) -> torch.Tensor:
                value = _camera_tensor(camera)
                if failure == "producer" and camera == 6:
                    raise OSError("bad camera media")
                if (failure == "local_shape" and camera == 4) or (failure == "remote_shape" and camera % 4 == 1):
                    return value[:1]
                return value.float() if failure == "remote_dtype" and camera % 4 == 1 else value

            expected = RuntimeError if failure in ("producer", "local_shape") else ValueError
            with pytest.raises(expected, match="Camera VAE producer failed|matching shapes"):
                for_each_camera_tensor(7, fail, lambda camera, _: published.append(camera), group)
            assert not published
            # Every peer must have raised; the group remains usable afterward.
            dist.barrier()
    finally:
        dist.destroy_process_group()


def test_camera_vae_distributed(tmp_path: Path) -> None:
    mp.spawn(_distributed_camera_worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=4, join=True)
