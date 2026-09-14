# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace

import pytest
import torch
from safetensors.torch import save_file

from vllm_omni.diffusion.models.cosmos3.lidar import (
    Cosmos3LidarEncoder,
    prepare_lidar_encoder_input,
    validate_lidar_config,
)
from vllm_omni.diffusion.models.cosmos3.multiview_flex_attention import (
    MaskItem,
    MultiviewAttentionContext,
    MultiviewLayout,
    build_multiview_flex_metadata,
    get_multiview_attention_plan,
    multiview_pair_predicate,
)
from vllm_omni.diffusion.models.cosmos3.multiview_packing import (
    pack_state,
    packed_position_ids,
    patchify_sensor,
    unpack_state,
    unpatchify_sensor,
)
from vllm_omni.diffusion.models.cosmos3.multiview_prompts import control_emphasis, format_camera_caption
from vllm_omni.model_extras.cosmos3_lidar import load_lidar_frames, required_lidar_sweeps, validate_lidar_header

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def lidar_config() -> dict:
    return {
        "version": "1.2",
        "fps": 10.0,
        "latent_channels": 128,
        "spatial_compression": [16, 16],
        "temporal_compression_factor": 1,
        "streaming_chunk_frames": 2,
        "streaming_context_frames": 3,
        "range_projection": {
            "native_height": 128,
            "semantic_width": 1800,
            "model_width": 1808,
            "model_width_transform": "circular_pad",
            "intensity_encoding": "unit",
            "min_range_m": 5.0,
            "max_range_m": 100.0,
        },
        "network_config": {
            "resolution": [128, 1808],
            "patch_size": [2, 2],
            "depths": [3, 3, 3, 3],
            "temporal_downsample": [False, False, False],
            "z_dim": 128,
            "in_channels": 3,
        },
    }


def numeric_frames(sweeps=3):
    frames = torch.zeros(3, sweeps, 128, 1800)
    frames[0] = 52.5
    frames[1:] = 1
    return frames


@pytest.mark.parametrize("frames,fps,sweeps", [(301, 30, 100), (5, 30, 2), (9, 30, 3), (9, 10, 9)])
def test_sweep_count_tracks_resolved_camera_duration(frames, fps, sweeps):
    assert required_lidar_sweeps(frames, fps, 10) == sweeps


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf")])
def test_rejects_invalid_sweep_rates(fps):
    with pytest.raises(ValueError):
        required_lidar_sweeps(9, fps, 10)


def test_numeric_file_truncates_excess_and_rejects_short_input(tmp_path):
    path = tmp_path / "input.safetensors"
    frames = numeric_frames()
    frames[:, 2] = float("nan")  # Unused sweeps must not be read or value-validated at admission.
    save_file({"frames": frames}, path)
    assert validate_lidar_header(path) == (3, 3, 128, 1800)
    actual = load_lidar_frames(path, num_sweeps=2)
    assert actual.is_contiguous() and actual.dtype == torch.float32
    torch.testing.assert_close(actual, frames[:, :2])
    with pytest.raises(ValueError, match="requires 4 sweeps"):
        load_lidar_frames(path, num_sweeps=4)
    with pytest.raises(ValueError, match="finite"):
        load_lidar_frames(path, num_sweeps=3)


def test_lidar_header_admission_does_not_materialize_tensor_values(monkeypatch):
    import safetensors

    class HeaderOnlySlice:
        def get_shape(self):
            return [3, 100, 128, 1800]

        def get_dtype(self):
            return "F32"

        def __getitem__(self, key):
            pytest.fail("Admission must not read tensor values")

    class HeaderOnlyFile:
        def keys(self):
            return ["frames"]

        def get_slice(self, key):
            assert key == "frames"
            return HeaderOnlySlice()

    monkeypatch.setattr(safetensors, "safe_open", lambda *args, **kwargs: nullcontext(HeaderOnlyFile()))
    assert validate_lidar_header("large.safetensors") == (3, 100, 128, 1800)


@pytest.mark.parametrize("invalid", ["dtype", "shape", "name", "nan", "intensity", "validity", "range", "corrupt"])
def test_rejects_invalid_numeric_uploads(tmp_path, invalid):
    path = tmp_path / "input.safetensors"
    frames = numeric_frames(1)
    if invalid == "dtype":
        frames = frames.half()
    elif invalid == "shape":
        frames = frames[..., :1799].contiguous()
    elif invalid in {"nan", "intensity", "validity", "range"}:
        channel = {"nan": 0, "intensity": 1, "validity": 2, "range": 0}[invalid]
        frames[channel, 0, 0, 0] = float("nan") if invalid == "nan" else -1
    if invalid == "corrupt":
        path.write_bytes(b"not a tensor file")
    else:
        save_file({"wrong" if invalid == "name" else "frames": frames}, path)
    if invalid in {"dtype", "shape", "name", "corrupt"}:
        with pytest.raises(ValueError):
            validate_lidar_header(path)
    else:
        assert validate_lidar_header(path) == (3, 1, 128, 1800)
    with pytest.raises(ValueError):
        load_lidar_frames(path)


def test_physical_normalization_circular_padding_and_validity():
    frames = numeric_frames(1)
    frames[0, 0, 0, :4] = torch.tensor([0.0, 4.9, 5.0, 100.1])
    frames[2, 0, 1, 0] = 0
    frames[1, 0, 2, 0] = 0.25
    normalized = prepare_lidar_encoder_input(frames, lidar_config()["range_projection"])
    assert normalized.shape == (1, 3, 1, 128, 1808)
    torch.testing.assert_close(normalized[..., :4], normalized[..., 1800:1804])
    torch.testing.assert_close(normalized[..., -4:], normalized[..., 4:8])
    assert normalized[0, :, 0, 0, 4].tolist() == [-1, -1, 0]
    assert normalized[0, :, 0, 0, 6].tolist() == [-1, 1, 1]
    assert normalized[0, :, 0, 0, 7].tolist() == [-1, -1, 0]
    assert normalized[0, :, 0, 1, 4].tolist() == [-1, -1, 0]
    assert normalized[0, 1, 0, 2, 4].item() == -0.5


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", "1"),
        ("fps", 0),
        ("spatial_compression", [8, 8]),
        ("temporal_compression_factor", 4),
        ("streaming_context_frames", 1),
    ],
)
def test_rejects_incompatible_encoder_metadata(field, value):
    config = lidar_config()
    validate_lidar_config(config)
    config[field] = value
    with pytest.raises(ValueError):
        validate_lidar_config(config)


def test_encoder_uses_fp32_posterior_mean_chunk_context_and_latent_affine():
    model = object.__new__(Cosmos3LidarEncoder)
    torch.nn.Module.__init__(model)
    model.config = lidar_config()
    model.coords = torch.zeros(1, 2, 128, 1808)
    model.latent_mean = torch.tensor(0.25)
    model.latent_std = torch.tensor(0.5)
    calls = []

    class Encoder(torch.nn.Module):
        def forward_stream(self, pixels, coords, cache):
            assert pixels.dtype == coords.dtype == torch.float32
            assert not torch.is_autocast_enabled("cpu")
            calls.append((pixels.shape[2], 0 if cache is None else cache["temporal"][0].shape[2]))
            channels = torch.cat((pixels[:, :1, :, :1, :1], torch.full_like(pixels[:, :1, :, :1, :1], 999)), dim=1)
            # Fake log variance is deliberately huge: only the posterior mean is encoded.
            return channels, {"temporal": (torch.zeros(1, 1, 9, 1), torch.zeros(1, 1, 9, 1))}

    model.encoder = Encoder()
    model.quant_conv = torch.nn.Identity()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = model(numeric_frames(5))
    assert calls == [(2, 0), (2, 1), (1, 2)]
    assert result.shape == (1, 1, 5, 1, 1)
    torch.testing.assert_close(result, torch.full_like(result, -0.5))
    assert not hasattr(model, "decode")


def mixed_items():
    return (
        MaskItem((6, 1, 2), 2, is_control=True, seconds_per_frame=4 / 30),
        MaskItem((6, 1, 2), 2, seconds_per_frame=4 / 30),
        MaskItem((7, 1, 3), 1, view_offset=2, is_control=True, is_lidar=True, seconds_per_frame=0.1),
        MaskItem((7, 1, 3), 1, view_offset=2, is_lidar=True, seconds_per_frame=0.1),
    )


@pytest.mark.parametrize("scope", ["all_views", "same_view", "decomposed"])
def test_mixed_boundaries_caption_isolation_sensor_controls_and_time_window(scope):
    items = mixed_items()
    offsets = [6]
    for item in items:
        offsets.append(offsets[-1] + item.num_tokens)
    metadata = build_multiview_flex_metadata(
        offsets[-1],
        offsets,
        items,
        "cpu",
        6,
        attention_scope=scope,
        decomposed_temporal_window_seconds=0.4,
        caption_lengths=(2, 4),
        control_attends_sensor=True,
    )
    allowed = multiview_pair_predicate(
        metadata, torch.arange(metadata.q_len)[:, None], torch.arange(metadata.kv_len)[None, :]
    )
    # Camera control and target read only their own caption, including all_views.
    assert allowed[0, :2].all() and not allowed[0, 2:6].any()
    assert not allowed[6, :2].any() and allowed[6, 2:6].all()
    assert allowed[24:, :6].all()  # Both LiDAR items read every camera caption.
    assert not allowed[12, offsets[2] : offsets[3]].any()  # Camera cannot read LiDAR controls.
    assert not allowed[45, offsets[0] : offsets[1]].any()  # LiDAR cannot read camera controls.
    if scope == "decomposed":
        # Last LiDAR sweep (0.6 s) sees camera capture 0.267 s but not 0 s.
        assert not allowed[63, offsets[1]]
        assert allowed[63, offsets[1] + 4]
        # A camera at t=0 cannot see future LiDAR targets.
        assert not allowed[12, offsets[3] + 3]


def test_layout_cache_keys_include_geometry_and_caption_boundaries():
    layout = MultiviewLayout(
        2,
        6,
        1,
        2,
        items=mixed_items(),
        caption_lengths=(2, 4),
        max_und_tokens=8,
        decomposed_temporal_window_seconds=0.4,
    )
    context = MultiviewAttentionContext(layout, {})
    plan, _ = get_multiview_attention_plan(context, device=torch.device("cpu"), real_q_len=66, real_und_len=6)
    assert plan is not None
    changed = replace(layout, caption_lengths=(3, 3))
    assert changed.cache_key() != layout.cache_key()
    other, _ = get_multiview_attention_plan(
        replace(context, layout=changed), device=torch.device("cpu"), real_q_len=66, real_und_len=6
    )
    assert other is not plan


def test_lidar_rewinds_to_camera_origin_and_advances_to_furthest_endpoint():
    items = mixed_items()
    positions, endpoint = packed_position_ids(items, text_origin=100, base_fps=30, camera_compression=4)
    torch.testing.assert_close(positions[:, :12], positions[:, 12:24])
    torch.testing.assert_close(positions[:, 24:45], positions[:, 45:66])
    assert positions[0, 24::3][:7].tolist() == [100, 100.75, 101.5, 102.25, 103, 103.75, 104.5]
    assert endpoint == 105
    assert positions[0, 6].item() == 100  # Second camera shares the origin.
    wide_camera = replace(items[0], token_shape=(6, 1, 200))
    _, endpoint = packed_position_ids((wide_camera, *items[1:]), text_origin=100, base_fps=30, camera_compression=4)
    assert endpoint == 200  # The cursor accounts for spatial endpoints too.


def test_sensor_patching_and_shared_scheduler_state_roundtrip():
    tensors = [torch.randn(1, 3, 4, 3, 5), torch.randn(1, 7, 9, 2, 3)]
    for tensor in tensors:
        torch.testing.assert_close(unpatchify_sensor(patchify_sensor(tensor, 2), tuple(tensor.shape[1:]), 2), tensor)
    packed = pack_state(tensors)
    shapes = [tuple(tensor.shape[1:]) for tensor in tensors]
    updated = unpack_state(packed + 0.125, shapes)
    for result, original in zip(updated, tensors):
        torch.testing.assert_close(result, original + 0.125)


def test_exact_reference_camera_labels_and_mode_emphasis():
    caption = format_camera_caption("Driving.", "camera_front_wide_120fov")
    assert caption == "The video is captured from a camera mounted on a car. The camera is facing forward. Driving."
    assert control_emphasis("wsm", joint=True) == (
        "Follow the wsm and lidar control videos precisely: every camera view must align with its world-scenario map, "
        "and the LiDAR rangemap must align with the HD-map rangemap, at every frame."
    )
    assert "silhouette" in control_emphasis("depth", joint=False)
    assert "lidar" not in control_emphasis("wsm", joint=False)


def test_packed_transformer_isolates_causal_captions_and_only_times_noisy_targets(monkeypatch):
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3_multiview import Cosmos3MultiviewVFMTransformer

    model = object.__new__(Cosmos3MultiviewVFMTransformer)
    torch.nn.Module.__init__(model)
    model.lidar_config = lidar_config()
    model.latent_patch_size = 1
    model.temporal_compression_factor = 4
    model.base_fps = 30
    model.enable_fps_modulation = True
    model.temporal_modality_margin = 10
    model.timestep_scale = 1
    model.cached_kv = model.cached_freqs_gen = None
    model._multiview_mask_cache, model._multiview_buffer_cache = {}, {}
    model._offload_context = lambda _: nullcontext()
    model.gen_sp_prepare = lambda hidden, cos, sin: (hidden, cos, sin)
    model.gen_sp_gather = torch.nn.Identity()
    model.proj_in = torch.nn.Linear(1, 2, bias=False)
    model.proj_in.weight.data.fill_(1)
    model.lidar_proj_in = model.lidar_proj_out = torch.nn.Identity()
    model._project_video_tokens = lambda hidden: hidden[:, :, :1]
    model.norm_moe_gen = torch.nn.Identity()
    model.time_embedder = lambda time: time[:, None].expand(-1, 2)
    origins, texts, layers = [], [], []

    class Language(torch.nn.Module):
        def rotary_emb(self, dummy, position_ids):
            origins.append(position_ids.clone())
            value = position_ids[0].unsqueeze(-1).expand(-1, -1, 2).float()
            return value, value

        def forward(self, ids, freqs):
            texts.append(ids.clone())
            key = ids.cumsum(1).float().unsqueeze(-1).unsqueeze(-1)
            return [(key, key)]

    class Layer(torch.nn.Module):
        def forward(self, hidden, **kwargs):
            layers.append((hidden.clone(), kwargs))
            return hidden

    model.language_model = Language()
    model.gen_layers = torch.nn.ModuleList([Layer()])
    camera = torch.zeros(1, 1, 4, 1, 1)
    lidar = torch.zeros(1, 2, 3, 1, 1)
    controls = [torch.full_like(camera, 3)]
    items = (
        MaskItem((4, 1, 1), 2, is_control=True, seconds_per_frame=4 / 30),
        MaskItem((4, 1, 1), 2, seconds_per_frame=4 / 30),
        MaskItem((3, 1, 1), 1, view_offset=2, is_lidar=True, is_control=True, seconds_per_frame=0.1),
        MaskItem((3, 1, 1), 1, view_offset=2, is_lidar=True, seconds_per_frame=0.1),
    )
    shapes = (tuple(camera.shape[1:]), tuple(lidar.shape[1:]))
    kwargs = dict(
        hidden_states=pack_state([camera, lidar]),
        timestep=torch.tensor([10.0]),
        text_ids=torch.tensor([[2, 3, 7, 11, 13]]),
        text_mask=torch.tensor([[1, 1, 2, 2, 2]]),
        caption_lengths=(2, 3),
        packed_shapes=shapes,
        control_latents=controls,
        lidar_control_latents=torch.full_like(lidar, 2),
        noisy_frame_mask=torch.tensor([0, 1, 0, 1]).reshape(1, 1, 4, 1, 1),
        temporal_position_period=2,
        multiview_layout=MultiviewLayout(
            2, 4, 1, 1, items=items, max_und_tokens=8, decomposed_temporal_window_seconds=0.4
        ),
    )
    prediction = model(**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(
            torch.Tensor, "item", lambda *args: pytest.fail("Denoising steps must not synchronize text lengths")
        )
        model(**kwargs)  # CPU position metadata was computed during cache initialization.
    assert [ids.tolist() for ids in texts] == [[[2, 3]], [[7, 11, 13]]]
    assert layers[0][1]["k_und"].flatten().tolist() == [2, 5, 7, 18, 31]
    assert origins[0][0, 0].tolist() == [0, 1]
    assert origins[1][0, 0].tolist() == [0, 1, 2]
    assert origins[2][0, 0, 0].item() == 13  # Longest caption + margin.
    assert layers[0][0][0, :, 0].tolist() == [3, 3, 3, 3, 0, 10, 0, 10, 2, 2, 2, 10, 10, 10]
    video_pred, lidar_pred = unpack_state(prediction, shapes)
    assert video_pred.flatten().tolist() == [0, 10, 0, 10]
    assert (lidar_pred == 10).all()
    model(**kwargs)
    assert len(texts) == 2  # Text cached once per branch.
    model.reset_cache()
    model(**{**kwargs, "control_latents": None})
    assert layers[-1][0].shape[1] == 7  # Both control streams are removed together for control CFG.
    assert all(not item.is_control for item in layers[-1][1]["multiview_layout"].layout.items)


@pytest.mark.parametrize("model_width", [6, 10])
def test_encoder_padding_uses_projection_widths(model_width):
    frames = torch.ones(3, 1, 1, 6)
    frames[0] = 52.5
    frames[1] = torch.linspace(0, 1, 6)
    projection = {"model_width": model_width, "semantic_width": 6, "min_range_m": 5, "max_range_m": 100}
    actual = prepare_lidar_encoder_input(frames, projection)
    half_padding = (model_width - 6) // 2
    columns = [(index - half_padding) % 6 for index in range(model_width)]
    assert actual.shape == (1, 3, 1, 1, model_width)
    torch.testing.assert_close(actual[0, 1, 0, 0], (frames[1, 0, 0] * 2 - 1)[columns])
    assert actual[0, 0].eq(0).all() and actual[0, 2].eq(1).all()


def test_missing_projection_weights_fail_even_when_some_lidar_weights_are_present():
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3_multiview import Cosmos3MultiviewVFMTransformer

    model = object.__new__(Cosmos3MultiviewVFMTransformer)
    torch.nn.Module.__init__(model)
    model.lidar_config = lidar_config()
    loaded = {
        f"transformer.lidar_proj_{direction}.{parameter}"
        for direction in ("in", "out")
        for parameter in ("weight", "bias")
    }
    model.validate_loaded_weights(loaded)
    with pytest.raises(ValueError, match="lidar_proj_out.bias"):
        model.validate_loaded_weights(loaded - {"transformer.lidar_proj_out.bias"})
    model.lidar_config = None
    model.validate_loaded_weights(set())  # Legacy WSM has no LiDAR modules.


@pytest.mark.parametrize(
    "mode,noise_source",
    [
        ("joint", "seed"),
        ("joint", "generator"),
        ("joint", "advanced_generator"),
        ("joint", "injected"),
        ("transfer", "seed"),
        ("ordinary", "seed"),
        ("completion", "seed"),
    ],
)
@pytest.mark.parametrize("emphasis", [False, True])
def test_pipeline_shares_schedule_preserves_conditions_and_returns_only_rgb(
    tmp_path, monkeypatch, mode, noise_source, emphasis
):
    from types import SimpleNamespace

    import vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview as module
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    pipeline = object.__new__(module.Cosmos3MultiviewPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.device, pipeline.dtype = torch.device("cpu"), torch.float32
    pipeline.vae_scale_factor_temporal, pipeline.vae_scale_factor_spatial = 4, 16
    pipeline.is_distilled_model = False
    cameras = tuple(module.COSMOS3_MADS_CAMERAS[1::-1])
    pipeline.multiview_cameras = cameras
    pipeline.multiview_config = {
        "schema_version": 2,
        "variable_view_count": True,
        "separate_view_text_tokenization": True,
        "inference_defaults": {"num_steps": 2, "guidance": 1},
    }
    pipeline.multiview_align_temporal_positions_across_views = True
    pipeline.multiview_attention_scope = "decomposed"
    pipeline.multiview_decomposed_temporal_window_seconds = 0.4
    pipeline.multiview_control_attends_sensor = True
    pipeline.multiview_backend = "triton"
    pipeline.transformer = SimpleNamespace(
        latent_channel_size=1,
        latent_patch_size=2,
        _pad_to_patch_size=lambda h, w: ((h + 1) // 2, (w + 1) // 2, 0, 0),
        reset_cache=lambda: None,
    )
    monkeypatch.setattr(module, "_resolve_multiview_geometry", lambda *args, **kwargs: ("480", "1,1", 32, 32))
    pipeline._set_timesteps = lambda *args, **kwargs: None
    pipeline._set_mixed_precision_step = lambda *args: None
    pipeline._reset_mixed_precision = lambda: None
    pipeline.progress_bar = lambda steps: steps
    pipeline._encode_video_tensor = lambda pixels: pixels[:, :1, ::4, ::16, ::16].clone()
    pipeline._prepare_camera_major_pixels = lambda views, **kwargs: torch.full((1, 3, 10, 32, 32), 0.5)
    decoded = []

    def decode(latents):
        decoded.append(latents.clone())
        assert latents.shape == (1, 1, 2, 2, 2)
        return torch.zeros(1, 3, 5, 32, 32)

    pipeline._decode_latents = decode
    texts = []

    def tokenize(prompt, *args, **kwargs):
        texts.append(prompt)
        length = 2 + len(prompt) % 3
        return torch.ones(1, length, dtype=torch.long), torch.ones(1, length, dtype=torch.long)

    pipeline._tokenize_prompt = tokenize
    extra = {
        "emphasize_control_in_prompt": emphasis,
        "multiview": {"condition_video_as_image": mode != "completion", "views": []},
    }
    for index, camera in enumerate(cameras):
        view = {"camera_key": camera, "prompt": f"Scene {index}."}
        if mode != "ordinary":
            view["control_path"] = "control.mp4"
        if mode != "completion" or index == 0:
            view["vision_path"] = "condition.mp4"
        extra["multiview"]["views"].append(view)
    if mode != "ordinary":
        extra["wsm"] = {}
    if mode == "joint":
        pipeline.multiview_config["lidar"] = lidar_config()
        path = tmp_path / "map.safetensors"
        save_file({"frames": numeric_frames(3)}, path)
        extra["lidar"] = {"control_path": str(path)}

        def encode(frames):
            assert frames.shape[1] == 2  # round(5 * 10 / 30); extra sweep discarded.
            return torch.full((1, 2, 2, 1, 3), 7.0)

        pipeline.lidar_encoder = encode
    calls, scheduler_calls = [], []

    def predict(**kwargs):
        calls.append(kwargs)
        targets = unpack_state(kwargs["hidden_states"], kwargs["packed_shapes"])
        if mode == "joint":
            assert len(targets) == 2 and kwargs["lidar_control_latents"].eq(7).all()
            assert kwargs["multiview_layout"].items[-1].is_lidar
        if mode != "ordinary":
            assert kwargs["control_latents"][0].eq(0.5).all()
        return torch.ones_like(kwargs["hidden_states"])

    pipeline.predict_noise = predict

    def step(noise, timestep, latents, **kwargs):
        scheduler_calls.append(timestep.item())
        return (latents - noise * 0.125,)

    pipeline.scheduler = SimpleNamespace(timesteps=torch.tensor([1000.0, 500.0]), step=step)
    sp = OmniDiffusionSamplingParams(num_frames=5, num_inference_steps=2, seed=42, extra_args=extra)
    expected_generator = torch.Generator().manual_seed(sp.seed)
    if noise_source != "seed":
        # An explicit generator, including its current position, takes
        # precedence over the request seed for every generated sensor.
        sp.generator = torch.Generator().manual_seed(123)
        if noise_source in {"advanced_generator", "injected"}:
            torch.randn(7, generator=sp.generator)
        expected_generator.set_state(sp.generator.get_state())
    camera_shape = (1, 1, 4, 2, 2)
    if noise_source == "injected":
        sp.latents = torch.full(camera_shape, 0.25)
        expected_camera = sp.latents.clone()
    else:
        expected_camera = torch.randn(camera_shape, generator=expected_generator)
    if mode == "joint":
        # LiDAR must use the next draw, without restarting the camera stream.
        expected_lidar = torch.randn((1, 2, 2, 1, 3), generator=expected_generator)
    if mode == "completion":
        expected_camera[:, :, :2] = 0.5
    else:
        expected_camera[:, :, ::2] = 0.5
    result = pipeline.forward(
        SimpleNamespace(prompts=[{"prompt": "ignored", "negative_prompt": "ignored too"}], sampling_params=sp)
    )
    assert scheduler_calls == [1000, 500]
    assert len(calls) == 2 and len(decoded) == 2
    initial_targets = unpack_state(calls[0]["hidden_states"], calls[0]["packed_shapes"])
    torch.testing.assert_close(initial_targets[0], expected_camera, rtol=0, atol=0)
    if mode == "joint":
        torch.testing.assert_close(initial_targets[1], expected_lidar, rtol=0, atol=0)
    if sp.generator is not None:
        assert torch.equal(sp.generator.get_state(), expected_generator.get_state())
    assert texts[1::2] == ["", ""]
    suffix = control_emphasis("wsm", joint=mode == "joint")
    for index, caption in enumerate(texts[::2]):
        assert f"Scene {index}." in caption
        assert caption.count(suffix) == int(emphasis and mode != "ordinary")
    if mode == "completion":
        assert decoded[0].eq(0.5).all()  # Entire known view stays fixed.
    else:
        assert all(view[:, :, 0].eq(0.5).all() for view in decoded)
    assert set(result.output) == {"payload", "metadata"}
    assert set(result.output["payload"]) == {"video"}
    assert result.output["metadata"]["multiview"]["cameras"] == list(cameras)


def test_view_completion_rejects_short_known_video(monkeypatch):
    import vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview as module

    pipeline = object.__new__(module.Cosmos3MultiviewPipeline)
    torch.nn.Module.__init__(pipeline)
    monkeypatch.setattr(
        module, "media_to_uint8_cthw", lambda *args, **kwargs: torch.zeros(3, 4, 32, 32, dtype=torch.uint8)
    )
    with pytest.raises(ValueError, match="complete RGB video of 5 frames"):
        pipeline._prepare_camera_major_pixels(
            [{"camera_key": "front", "vision_path": "short.mp4"}],
            field="vision",
            height=32,
            width=32,
            num_frames=5,
            keep_first=False,
            require_complete=True,
        )


def test_encoder_artifact_inventory_and_fp32_loading(tmp_path):
    import json

    config = lidar_config()
    with pytest.raises(ValueError, match="Incomplete joint artifact"):
        Cosmos3LidarEncoder.from_pretrained(str(tmp_path), config, torch.device("cpu"))
    folder = tmp_path / "lidar_encoder"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps(config))

    class TinyEncoder(Cosmos3LidarEncoder):
        def __init__(self, config):
            torch.nn.Module.__init__(self)
            self.weight = torch.nn.Parameter(torch.empty(1))
            self.register_buffer("latent_mean", torch.empty(1))
            self.register_buffer("latent_std", torch.empty(1))

    state = {"weight": torch.tensor([1.0001]), "latent_mean": torch.zeros(1), "latent_std": torch.ones(1)}
    save_file(state, folder / "model.safetensors")
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        loaded = TinyEncoder.from_pretrained(str(tmp_path), config, torch.device("cpu"))
    finally:
        torch.set_default_dtype(previous)
    assert loaded.weight.dtype == torch.float32
    torch.testing.assert_close(loaded.weight, state["weight"], rtol=0, atol=0)
    save_file({key: value for key, value in state.items() if key != "weight"}, folder / "model.safetensors")
    with pytest.raises(RuntimeError, match="Missing key"):
        TinyEncoder.from_pretrained(str(tmp_path), config, torch.device("cpu"))
    save_file({**state, "latent_std": torch.zeros(1)}, folder / "model.safetensors")
    with pytest.raises(ValueError, match="positive standard deviations"):
        TinyEncoder.from_pretrained(str(tmp_path), config, torch.device("cpu"))
    (folder / "config.json").write_text(json.dumps({**config, "fps": 11}))
    with pytest.raises(ValueError, match="metadata disagrees"):
        TinyEncoder.from_pretrained(str(tmp_path), config, torch.device("cpu"))


def test_unipc_updates_mixed_sensor_state_with_one_sigma_schedule():
    from vllm_omni.diffusion.models.schedulers.scheduling_flow_unipc_multistep import FlowUniPCMultistepScheduler

    camera, lidar = torch.randn(1, 1, 3, 2, 2), torch.randn(1, 2, 5, 1, 3)
    shapes = (tuple(camera.shape[1:]), tuple(lidar.shape[1:]))
    schedulers = [FlowUniPCMultistepScheduler() for _ in range(3)]
    for scheduler in schedulers:
        scheduler.set_timesteps(3, device="cpu", shift=10)
    state = pack_state([camera, lidar])
    for timestep in schedulers[0].timesteps:
        camera_v, lidar_v = camera * 0.1, lidar * 0.2
        state = schedulers[0].step(pack_state([camera_v, lidar_v]), timestep, state, return_dict=False)[0]
        camera = schedulers[1].step(camera_v, timestep, camera, return_dict=False)[0]
        lidar = schedulers[2].step(lidar_v, timestep, lidar, return_dict=False)[0]
        packed_camera, packed_lidar = unpack_state(state, shapes)
        torch.testing.assert_close(packed_camera, camera)
        torch.testing.assert_close(packed_lidar, lidar)
