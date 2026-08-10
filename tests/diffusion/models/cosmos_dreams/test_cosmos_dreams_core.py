# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.cosmos3.action import ActionNormalizer
from vllm_omni.diffusion.models.cosmos_dreams.config import CosmosDreamsManifest
from vllm_omni.diffusion.models.cosmos_dreams.sampler import CosmosDreamsDistilledSampler
from vllm_omni.diffusion.models.cosmos_dreams.state_cosmos_dreams import (
    CosmosDreamsSessionFingerprint,
    CosmosDreamsSessionState,
)
from vllm_omni.diffusion.models.cosmos_dreams.utils import (
    build_interleaved_mrope_position_ids,
    estimate_kv_memory_bytes,
    interleave_action_vision_tokens,
    iter_ar_chunk_ranges,
    iter_clean_commit_frames,
    split_interleaved_action_vision_tokens,
    zero_null_action_values,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _fingerprint(**overrides) -> CosmosDreamsSessionFingerprint:
    values = {
        "prompt_hash": "prompt",
        "real_text_kv_lengths": (("main", 12),),
        "height": 720,
        "width": 1280,
        "fps": 15.0,
        "domain_id": 15,
        "normalizer_id": "normalizer",
        "checkpoint_id": "checkpoint",
        "manifest_id": "manifest",
        "sampler_id": "sampler",
    }
    values.update(overrides)
    return CosmosDreamsSessionFingerprint(**values)


def test_manifest_reads_exported_nested_causal_fields() -> None:
    artifact = {
        "chunk_size": 4,
        "checkpoint_hash": "a" * 64,
        "checkpoint_id": "checkpoint",
        "checkpoint_iteration": 1600,
        "normalizer_id": "normalizer",
        "normalizer_source": "config/normalizer.json",
        "action_normalizer": {
            "mean": [0.0],
            "std": [1.0],
            "source": "config/normalizer.json",
        },
        "embodiment_to_domain": {"agibotworld": 15, "camera": 2},
        "window_frames": 64,
        "deploy_resolution": [720, 1280],
        "fixed_step_sampler_config": {
            "sample_type": "sde",
            "t_list": [1.0, 15 / 16, 5 / 6, 5 / 8],
        },
    }
    config = SimpleNamespace(
        model_config={
            "cosmos_dreams": {
                "chunk_size": 4,
                "window_frames": 64,
                "deploy_resolution": [720, 1280],
            }
        },
        tf_model_config=SimpleNamespace(to_dict=lambda: {"cosmos_dreams": artifact}),
        custom_pipeline_args={},
    )

    manifest = CosmosDreamsManifest.from_od_config(config)

    assert manifest.window_frames == 64
    assert manifest.patch_grid == (23, 40)
    assert manifest.vision_tokens_per_frame == 920
    assert manifest.tokens_per_frame == 924
    assert manifest.checkpoint_hash == "a" * 64
    manifest.require_exported_artifact()
    assert manifest.resolve_domain_name(" CAMERA ") == 2

    config.model_config["cosmos_dreams"]["chunk_size"] = 8
    with pytest.raises(ValueError, match="contradicts the exported artifact"):
        CosmosDreamsManifest.from_od_config(config, require_explicit=True)


def test_deploy_manifest_cannot_replace_missing_transformer_artifact() -> None:
    config = SimpleNamespace(
        model_config={
            "cosmos_dreams": {
                "checkpoint_id": "deploy-only",
                "checkpoint_iteration": 1600,
                "checkpoint_hash": "a" * 64,
                "normalizer_id": "deploy-only",
                "normalizer_source": "deploy-only",
                "action_normalizer": {
                    "mean": [0.0],
                    "std": [1.0],
                    "source": "deploy-only",
                },
            }
        },
        tf_model_config={},
        custom_pipeline_args={},
    )

    with pytest.raises(ValueError, match="deployment defaults are not a validated artifact"):
        CosmosDreamsManifest.from_od_config(config, require_explicit=True)


def test_manifest_rejects_deployment_defaults_and_placeholder_hash() -> None:
    with pytest.raises(ValueError, match="checkpoint_id.*checkpoint_iteration.*checkpoint_hash"):
        CosmosDreamsManifest().require_exported_artifact()

    with pytest.raises(ValueError, match="all-zero template"):
        CosmosDreamsManifest(checkpoint_hash="0" * 64)

    manifest = CosmosDreamsManifest(
        checkpoint_id="checkpoint",
        checkpoint_iteration=1600,
        checkpoint_hash="a" * 64,
        normalizer_id="normalizer",
        normalizer_source="config/normalizer.json",
    )
    with pytest.raises(ValueError, match="Unknown Cosmos-Dreams domain_name"):
        manifest.resolve_domain_name("unknown")


def test_chunk_partition_is_singleton_then_four_frame_chunks() -> None:
    assert list(iter_ar_chunk_ranges(0, 11, 4)) == [(0, 1), (1, 5), (5, 9), (9, 11)]
    assert list(iter_ar_chunk_ranges(5, 10, 4)) == [(5, 9), (9, 10)]


def test_clean_commit_order_is_per_frame_and_skips_only_global_terminal_frame() -> None:
    assert list(
        iter_clean_commit_frames(1, 5, target_frame=5, terminal_request=False)
    ) == [(0, 1), (1, 2), (2, 3), (3, 4)]
    assert list(
        iter_clean_commit_frames(1, 5, target_frame=5, terminal_request=True)
    ) == [(0, 1), (1, 2), (2, 3)]
    assert list(
        iter_clean_commit_frames(5, 7, target_frame=7, terminal_request=True)
    ) == [(0, 5)]


def test_action_and_vision_tokens_round_trip_in_per_frame_order() -> None:
    action = torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]])
    vision = torch.tensor([[[[10.0], [11.0], [12.0]], [[20.0], [21.0], [22.0]]]])

    packed = interleave_action_vision_tokens(action, vision)
    unpacked_action, unpacked_vision = split_interleaved_action_vision_tokens(
        packed,
        num_frames=2,
        action_tokens_per_frame=2,
        vision_tokens_per_frame=3,
    )

    assert packed.flatten().tolist() == [1, 2, 10, 11, 12, 3, 4, 20, 21, 22]
    torch.testing.assert_close(unpacked_action, action)
    torch.testing.assert_close(unpacked_vision, vision)


def test_interleaved_mrope_uses_margin_fps_and_action_substeps() -> None:
    ids = build_interleaved_mrope_position_ids(
        frame_start=2,
        num_frames=2,
        grid_h=1,
        grid_w=2,
        text_temporal_offset=10,
        temporal_modality_margin=100,
        fps=12.0,
        base_fps=24.0,
        action_tokens_per_frame=4,
        null_action_frames=(3,),
    )

    # Frame stride is 2. Frame 2 vision time is 114; real action substeps
    # cover (112, 114]. A later all-null frame retains those architectural
    # action IDs even though its action V is zeroed.
    torch.testing.assert_close(ids[0, :4], torch.tensor([112.5, 113.0, 113.5, 114.0]))
    torch.testing.assert_close(ids[0, 4:6], torch.tensor([114.0, 114.0]))
    torch.testing.assert_close(ids[0, 6:10], torch.tensor([114.5, 115.0, 115.5, 116.0]))
    torch.testing.assert_close(ids[0, 10:12], torch.tensor([116.0, 116.0]))

    singleton_null = build_interleaved_mrope_position_ids(
        frame_start=3,
        num_frames=1,
        grid_h=1,
        grid_w=2,
        text_temporal_offset=10,
        temporal_modality_margin=100,
        fps=12.0,
        base_fps=24.0,
        action_tokens_per_frame=4,
        null_action_frames=(3,),
    )
    torch.testing.assert_close(singleton_null[0, :4], torch.full((4,), 116.0))


def test_null_action_zeroes_values_but_not_other_tokens() -> None:
    value = torch.arange(2 * 6, dtype=torch.float32).view(1, 12, 1, 1)
    result = zero_null_action_values(
        value,
        num_frames=2,
        tokens_per_frame=6,
        action_tokens_per_frame=2,
        null_frame_indexes=(1,),
    )

    torch.testing.assert_close(result[:, :6], value[:, :6])
    torch.testing.assert_close(result[:, 6:8], torch.zeros_like(result[:, 6:8]))
    torch.testing.assert_close(result[:, 8:], value[:, 8:])
    assert torch.count_nonzero(value[:, 6:8]) > 0


def test_text_padding_must_be_sliced_before_joint_softmax() -> None:
    query = torch.tensor([[[[1.0]]]])
    real_key = torch.tensor([[[[2.0]]]])
    real_value = torch.tensor([[[[3.0]]]])
    padded_key = torch.cat([real_key, torch.zeros(1, 1, 1, 1)], dim=2)
    padded_value = torch.cat([real_value, torch.zeros(1, 1, 1, 1)], dim=2)

    sliced = torch.nn.functional.scaled_dot_product_attention(
        query,
        padded_key[:, :, :1],
        padded_value[:, :, :1],
    )
    unsliced = torch.nn.functional.scaled_dot_product_attention(query, padded_key, padded_value)

    torch.testing.assert_close(sliced, torch.tensor([[[[3.0]]]]))
    assert not torch.allclose(sliced, unsliced)


def test_kv_estimator_counts_managed_scratch_and_text_pools() -> None:
    manifest = CosmosDreamsManifest(window_frames=2, text_cache_max_len=3, height=32, width=32)
    estimate = estimate_kv_memory_bytes(
        manifest,
        num_layers=2,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.float16,
    )

    assert estimate.managed_blocks == 5  # window + in-flight + two allocator sentinels
    assert estimate.scratch_blocks == 1
    assert estimate.total_bytes == (
        estimate.self_attention_bytes + estimate.scratch_bytes + estimate.cross_attention_bytes
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prompt_hash", "other-prompt"),
        ("real_text_kv_lengths", (("main", 13),)),
        ("height", 704),
        ("width", 1216),
        ("fps", 24.0),
        ("domain_id", 2),
        ("normalizer_id", "other-normalizer"),
        ("checkpoint_id", "other-checkpoint"),
        ("manifest_id", "other-manifest"),
        ("sampler_id", "other-sampler"),
    ],
)
def test_session_fingerprint_rejects_every_conditioning_mutation(field: str, value) -> None:
    state = CosmosDreamsSessionState("session")
    fingerprint = _fingerprint()
    state.initialize(fingerprint)
    state.validate_request(fingerprint, frame_idx=0)

    with pytest.raises(ValueError, match=rf"{field}.*session reset required"):
        state.validate_request(_fingerprint(**{field: value}), frame_idx=0)


def test_session_fingerprint_rejects_order_terminal_reuse_and_resets() -> None:
    state = CosmosDreamsSessionState("session")
    fingerprint = _fingerprint()
    state.initialize(fingerprint)

    with pytest.raises(ValueError, match="out of order"):
        state.validate_request(fingerprint, frame_idx=4)

    state.terminal = True
    with pytest.raises(ValueError, match="completed a full rollout"):
        state.validate_request(fingerprint, frame_idx=0)

    state.reset()
    assert state.fingerprint is None
    assert state.next_frame_idx == 0
    assert not state.terminal


def test_action_normalizer_affine_transform_and_identity() -> None:
    normalizer = ActionNormalizer.from_config(
        {"mean": [1.0, -1.0], "std": [2.0, 4.0], "id": "stats"}
    )

    normalized = normalizer.normalize(torch.tensor([[3.0, 3.0]]))

    torch.testing.assert_close(normalized, torch.tensor([[1.0, 1.0]]))
    assert normalizer.identity == "stats"


def test_distilled_sampler_uses_sigma_times_training_timesteps_and_is_seeded() -> None:
    sampler = CosmosDreamsDistilledSampler([1.0, 0.5], sample_type="sde", num_train_timesteps=1000)
    timesteps: list[float] = []

    def velocity(x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        timesteps.append(float(timestep.item()))
        return torch.zeros_like(x)

    initial = torch.ones(1, 1, 1, 1, 1)
    first = sampler.sample(velocity, initial, seed=7, frame_idx=3)
    timesteps.clear()
    second = sampler.sample(velocity, initial, seed=7, frame_idx=3)

    assert timesteps == [1000.0, 500.0]
    torch.testing.assert_close(first, second)
    assert first.dtype == torch.float32


def _action_layout_pipeline_stub():
    from vllm_omni.diffusion.models.cosmos_dreams.pipeline_cosmos_dreams import (
        CosmosDreamsPipeline,
    )

    stub = SimpleNamespace(
        manifest=CosmosDreamsManifest(),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    return CosmosDreamsPipeline, stub


def _numbered_action_rows(rows: int, dim: int = 64) -> torch.Tensor:
    return torch.arange(rows, dtype=torch.float32).unsqueeze(1).repeat(1, dim)


def test_action_layout_resolution_is_per_request_and_rejects_short_actions() -> None:
    from vllm_omni.experimental.ar_diffusion.capability import ARDiffusionRequestRejectedError

    pipeline_cls, stub = _action_layout_pipeline_stub()
    resolve = pipeline_cls._resolve_action_layout

    assert resolve(stub, None, start_frame=0, target_frame=5) is None
    # Whole-rollout jsonl layout: rows cover frames 1..target-1 globally.
    assert resolve(stub, _numbered_action_rows(16), start_frame=0, target_frame=5) == "global"
    # Tick continuation with exactly this request's rows is local, and must
    # stay local for EVERY chunk of the request (rows < global requirement).
    assert resolve(stub, _numbered_action_rows(16), start_frame=5, target_frame=9) == "local"
    with pytest.raises(ARDiffusionRequestRejectedError, match="cannot cover"):
        resolve(stub, _numbered_action_rows(20), start_frame=5, target_frame=9)


def test_actions_for_frames_local_layout_is_stable_across_chunks() -> None:
    """Regression: a two-chunk local-layout tick must consume rows in request
    order; per-chunk global/local re-detection previously flipped the first
    chunk to global indexing and misconditioned every frame."""
    pipeline_cls, stub = _action_layout_pipeline_stub()
    slice_actions = pipeline_cls._actions_for_frames
    raw = _numbered_action_rows(32)  # request [5, 13): 8 frames x 4 rows

    first_chunk, first_null = slice_actions(
        stub, raw, layout="local", request_start_frame=5, frame_start=5, frame_end=9
    )
    second_chunk, second_null = slice_actions(
        stub, raw, layout="local", request_start_frame=5, frame_start=9, frame_end=13
    )

    assert first_null == () and second_null == ()
    torch.testing.assert_close(first_chunk[0, :, 0], torch.arange(16, dtype=torch.float32))
    torch.testing.assert_close(
        second_chunk[0, :, 0], torch.arange(16, 32, dtype=torch.float32)
    )


def test_actions_for_frames_global_layout_and_frame_zero_null() -> None:
    pipeline_cls, stub = _action_layout_pipeline_stub()
    slice_actions = pipeline_cls._actions_for_frames
    raw = _numbered_action_rows(32)  # global rows for frames 1..8

    chunk, nulls = slice_actions(
        stub, raw, layout="global", request_start_frame=0, frame_start=5, frame_end=9
    )
    torch.testing.assert_close(chunk[0, :, 0], torch.arange(16, 32, dtype=torch.float32))
    assert nulls == ()

    prefix, prefix_nulls = slice_actions(
        stub, raw, layout="global", request_start_frame=0, frame_start=0, frame_end=1
    )
    assert prefix_nulls == (0,)
    torch.testing.assert_close(prefix, torch.zeros_like(prefix))
