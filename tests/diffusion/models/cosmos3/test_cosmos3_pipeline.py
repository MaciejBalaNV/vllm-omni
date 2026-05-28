# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

from tests.diffusion.models.cosmos3.conftest import make_sampling_params

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _ids(value: int) -> torch.Tensor:
    return torch.tensor([[value]], dtype=torch.long)


def _mask() -> torch.Tensor:
    return torch.ones(1, 1, dtype=torch.long)


def test_pipeline_registered_and_exported() -> None:
    from vllm_omni.diffusion.cache.cache_dit_backend import CUSTOM_DIT_ENABLERS
    from vllm_omni.diffusion.models import cosmos3
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import Cosmos3OmniDiffusersPipeline
    from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
    from vllm_omni.diffusion.registry import (
        _DIFFUSION_MODELS,
        _DIFFUSION_POST_PROCESS_FUNCS,
        _DIFFUSION_PRE_PROCESS_FUNCS,
    )

    assert issubclass(Cosmos3OmniDiffusersPipeline, nn.Module)
    assert issubclass(Cosmos3OmniDiffusersPipeline, ProgressBarMixin)
    assert Cosmos3OmniDiffusersPipeline.support_image_input is True
    assert Cosmos3OmniDiffusersPipeline.support_video_input is True
    assert _DIFFUSION_MODELS["Cosmos3OmniDiffusersPipeline"] == (
        "cosmos3",
        "pipeline_cosmos3",
        "Cosmos3OmniDiffusersPipeline",
    )
    assert _DIFFUSION_PRE_PROCESS_FUNCS["Cosmos3OmniDiffusersPipeline"] == "get_cosmos3_pre_process_func"
    assert _DIFFUSION_POST_PROCESS_FUNCS["Cosmos3OmniDiffusersPipeline"] == "get_cosmos3_post_process_func"
    assert "Cosmos3OmniDiffusersPipeline" in CUSTOM_DIT_ENABLERS
    assert "Cosmos3OmniDiffusersPipeline" in cosmos3.__all__


def test_preprocess_i2v_image_and_action_video_inputs() -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import get_cosmos3_pre_process_func

    preprocess = get_cosmos3_pre_process_func(SimpleNamespace())
    i2v = SimpleNamespace(
        prompts=[{"prompt": "A slow camera push.", "multi_modal_data": {"image": Image.new("RGB", (320, 160))}}],
        sampling_params=SimpleNamespace(height=None, width=None, extra_args={}),
    )

    result = preprocess(i2v)
    assert (result.sampling_params.height, result.sampling_params.width) == (672, 1344)
    assert tuple(result.prompts[0]["additional_information"]["preprocessed_image"].shape[-2:]) == (672, 1344)

    frames = [Image.new("RGB", (8, 4), color) for color in ("red", "green", "blue")]
    action = SimpleNamespace(
        prompts=[{"prompt": "Move.", "multi_modal_data": {"video": frames}}],
        sampling_params=SimpleNamespace(height=16, width=32, extra_args={"action_mode": "forward_dynamics"}),
    )

    additional = preprocess(action).prompts[0]["additional_information"]
    assert tuple(additional["preprocessed_image"].shape) == (1, 3, 16, 32)
    assert tuple(additional["preprocessed_video"].shape) == (1, 3, 3, 16, 32)

    frames = [Image.new("RGB", (8, 4), color) for color in ("red", "green", "blue", "yellow", "purple", "black")]
    v2v = SimpleNamespace(
        prompts=[{"prompt": "Continue.", "multi_modal_data": {"video": frames}}],
        sampling_params=SimpleNamespace(
            height=16,
            width=32,
            extra_args={"condition_frame_indexes_vision": [0, 1], "condition_video_keep": "last"},
        ),
    )
    additional = preprocess(v2v).prompts[0]["additional_information"]
    assert tuple(additional["preprocessed_video"].shape) == (1, 3, 5, 16, 32)
    assert additional["condition_frame_indexes_vision"] == [0, 1]


def test_postprocess_handles_image_video_audio_and_validation() -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import get_cosmos3_post_process_func

    func = get_cosmos3_post_process_func(SimpleNamespace())
    video = torch.zeros(1, 3, 1, 4, 4)

    assert func(video, output_type="latent") is video
    assert func({"image": video})[0].size == (4, 4)
    assert "video" in func({"video": video})
    assert (
        func(
            {"video": video, "audio": torch.ones(1, 2, 16), "audio_sample_rate": 48000},
            sampling_params=SimpleNamespace(extra_args={"resolved_frame_rate": 12}),
        )["audio_sample_rate"]
        == 48000
    )

    with pytest.raises(ValueError, match="text-to-image postprocess expects"):
        func({"image": torch.zeros(1, 3, 2, 4, 4)})
    with pytest.raises(ValueError, match="both image and video"):
        func({"image": video, "video": video})


def test_prompt_formatting_and_checkpoint_key_remap(make_cosmos3_pipeline) -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import Cosmos3OmniDiffusersPipeline

    pipeline = make_cosmos3_pipeline()
    captured: list[str] = []
    pipeline._tokenize_prompt = lambda text, *args, **kwargs: (captured.append(text) or _ids(len(captured)), _mask())

    pipeline._format_and_tokenize_prompts(
        "A robot",
        "bad",
        num_frames=48,
        frame_rate=24,
        height=720,
        width=1280,
        max_sequence_length=32,
        sp=SimpleNamespace(extra_args={"negative_metadata_mode": "inverse"}),
        use_system_prompt=True,
        is_t2i=False,
    )
    assert "The video is 2.0 seconds long" in captured[0]
    assert "The video is not 2.0 seconds long" in captured[1]

    remaps = {
        "embed_tokens.weight": "transformer.language_model.embed_tokens.weight",
        "model.embed_tokens.weight": "transformer.language_model.embed_tokens.weight",
        "norm.weight": "transformer.language_model.norm.weight",
        "norm_moe_gen.weight": "transformer.norm_moe_gen.weight",
        "proj_in.weight": "transformer.proj_in.weight",
        "proj_out.bias": "transformer.proj_out.bias",
        "layers.3.self_attn.to_q.weight": "transformer.language_model.layers.3.self_attn.to_q.weight",
        "layers.3.self_attn.to_out.weight": "transformer.language_model.layers.3.self_attn.to_out.weight",
        "layers.3.self_attn.norm_q.weight": "transformer.language_model.layers.3.self_attn.norm_q.weight",
        "layers.3.self_attn.add_q_proj.weight": "transformer.gen_layers.3.cross_attention.to_q.weight",
        "layers.3.self_attn.to_add_out.weight": "transformer.gen_layers.3.cross_attention.to_out.weight",
        "layers.3.self_attn.norm_added_q.weight": "transformer.gen_layers.3.cross_attention.norm_q.weight",
        "transformer.model.layers.3.self_attn.add_k_proj.weight": (
            "transformer.gen_layers.3.cross_attention.to_k.weight"
        ),
        "audio_proj_in.weight": "transformer.audio_proj_in.weight",
        "audio_modality_embed.weight": "transformer.audio_modality_embed",
        "action_proj_in.fc.weight": "transformer.action_proj_in.fc.weight",
        "action_modality_embed.weight": "transformer.action_modality_embed",
        "lm_head.weight": None,
    }
    assert {key: Cosmos3OmniDiffusersPipeline._remap_ckpt_key(key) for key in remaps} == remaps


def test_prepare_latents_for_video_image_sound_and_action(make_cosmos3_pipeline) -> None:
    pipeline = make_cosmos3_pipeline()
    latents = pipeline._prepare_latents(16, 24, 5, torch.Generator(device="cpu").manual_seed(0))
    assert latents.shape == (1, 2, 2, 2, 3)

    pipeline._encode_conditioning_video = lambda *args, **kwargs: torch.full((1, 2, 2, 2, 3), 5.0)
    i2v_latents, velocity_mask, image_latent = pipeline._prepare_latents_i2v(
        torch.zeros(1, 3, 16, 24), 16, 24, 5, torch.Generator(device="cpu").manual_seed(0)
    )
    torch.testing.assert_close(i2v_latents[:, :, 0], torch.full((1, 2, 2, 3), 5.0))
    assert velocity_mask.tolist() == [[[[[0.0]], [[1.0]]]]]
    assert image_latent.shape == (1, 2, 1, 2, 3)

    pipeline._encode_video_tensor = lambda *args, **kwargs: torch.full((1, 2, 3, 2, 3), 6.0)
    v2v_latents, v2v_velocity_mask, v2v_condition = pipeline._prepare_latents_v2v(
        torch.zeros(1, 3, 5, 16, 24),
        16,
        24,
        9,
        torch.Generator(device="cpu").manual_seed(0),
        [0, 1],
    )
    torch.testing.assert_close(v2v_latents[:, :, 0:2], torch.full((1, 2, 2, 2, 3), 6.0))
    assert v2v_velocity_mask.tolist() == [[[[[0.0]], [[0.0]], [[1.0]]]]]
    assert v2v_condition.shape == (1, 2, 3, 2, 3)

    pipeline.transformer = pipeline.transformer.__class__(latent_channel_size=2, sound_gen=True, sound_dim=3)
    pipeline._sound_tokenizer = SimpleNamespace(
        sample_rate=10,
        latent_ch=3,
        hop_size=4,
        decode=lambda x: torch.ones(x.shape[0], 2, 24),
    )
    assert pipeline._resolve_sound_target_samples(SimpleNamespace(extra_args={"sound_duration": 2.0}), 9, 3.0) == (
        20,
        2.0,
        10,
    )
    sound_latents, latent_frames = pipeline._prepare_sound_latents(21, torch.Generator(device="cpu").manual_seed(0))
    assert (sound_latents.shape, latent_frames) == (torch.Size([1, 3, 6]), 6)
    assert pipeline._decode_sound_latents(torch.zeros(1, 3, 6), target_audio_samples=21).shape == (1, 2, 21)

    pipeline.transformer = pipeline.transformer.__class__(action_gen=True, action_dim=4)
    action, action_mask, clean, raw_dim = pipeline._prepare_action_latents(
        mode="forward_dynamics",
        action_chunk_size=2,
        raw_action_dim=None,
        generator=torch.Generator(device="cpu").manual_seed(0),
        sp=SimpleNamespace(extra_args={"action": [[1.0, 2.0], [3.0, 4.0]]}),
    )
    assert raw_dim == 2
    assert action_mask.tolist() == [[[0.0], [0.0]]]
    torch.testing.assert_close(action, clean)


def test_diffuse_covers_cfg_i2v_and_multimodal_steps(make_cosmos3_pipeline) -> None:
    pipeline = make_cosmos3_pipeline()
    latents = torch.zeros(1, 2, 1, 1, 1)

    result = pipeline.diffuse(
        latents=latents,
        timesteps=torch.tensor([900, 100]),
        cond_ids=_ids(2),
        cond_mask=_mask(),
        uncond_ids=_ids(1),
        uncond_mask=_mask(),
        guidance_scale=3.0,
        shared_kwargs={"video_shape": (1, 1, 1), "fps": 24.0},
        guidance_interval=(500.0, 1000.0),
    )
    assert [call["token"] for call in pipeline.transformer.calls] == [2, 1, 2]
    torch.testing.assert_close(result, torch.full_like(latents, 6.0))

    i2v = pipeline.diffuse(
        latents=torch.zeros(1, 2, 2, 1, 1),
        timesteps=torch.tensor([7]),
        cond_ids=_ids(2),
        cond_mask=_mask(),
        uncond_ids=_ids(1),
        uncond_mask=_mask(),
        guidance_scale=1.0,
        shared_kwargs={"video_shape": (2, 1, 1), "fps": 24.0},
        velocity_mask=torch.tensor([[[[[0.0]], [[1.0]]]]]),
        image_latent=torch.full((1, 2, 1, 1, 1), 7.0),
    )
    torch.testing.assert_close(i2v[:, :, 0:1], torch.full((1, 2, 1, 1, 1), 7.0))

    pipeline.transformer = pipeline.transformer.__class__(latent_channel_size=2, action_gen=True, action_dim=4)
    video_result, action_result = pipeline.diffuse(
        latents=latents,
        action_latents=torch.zeros(1, 3, 4),
        action_velocity_mask=torch.ones(1, 3, 1),
        action_condition_latents=torch.zeros(1, 3, 4),
        timesteps=torch.tensor([7, 3]),
        cond_ids=_ids(2),
        cond_mask=_mask(),
        uncond_ids=_ids(1),
        uncond_mask=_mask(),
        guidance_scale=1.0,
        shared_kwargs={"video_shape": (1, 1, 1), "fps": 24.0, "action_domain_ids": torch.tensor([0])},
    )
    torch.testing.assert_close(video_result, torch.full_like(latents, 4.0))
    torch.testing.assert_close(action_result, torch.full((), 44.0).expand_as(action_result))


class TestForwardRouting:
    def _install_forward_stubs(self, pipeline):
        captured: dict[str, object] = {"diffuse_calls": [], "prepare_calls": []}

        def fake_format(prompt, negative_prompt, num_frames, frame_rate, height, width, *args, **kwargs):
            captured["format"] = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_frames": num_frames,
                "frame_rate": frame_rate,
                "height": height,
                "width": width,
                "is_t2i": kwargs["is_t2i"],
            }
            return _ids(2), _mask(), _ids(1), _mask()

        def fake_prepare(height, width, num_frames, generator):
            captured["prepare_calls"].append((height, width, num_frames, generator.initial_seed()))
            return torch.zeros(1, 2, 1, 1, 1)

        def fake_diffuse(**kwargs):
            captured["diffuse_calls"].append(kwargs)
            outputs = [kwargs["latents"] + len(captured["diffuse_calls"])]
            if kwargs.get("action_latents") is not None:
                outputs.append(kwargs["action_latents"] + 3.0)
            if kwargs.get("sound_latents") is not None:
                outputs.append(kwargs["sound_latents"] + 2.0)
            return outputs[0] if len(outputs) == 1 else tuple(outputs)

        pipeline._format_and_tokenize_prompts = fake_format
        pipeline._prepare_latents = fake_prepare
        pipeline._set_flow_shift = lambda target: captured.setdefault("flow_shifts", []).append(target)

        def fake_set_scheduler_timesteps(steps):
            captured.setdefault("scheduler_steps", []).append(steps)
            pipeline.scheduler.timesteps = torch.tensor([7])

        pipeline._set_scheduler_timesteps = fake_set_scheduler_timesteps
        pipeline.diffuse = fake_diffuse
        pipeline._decode_latents = lambda latents: latents
        return captured

    @pytest.mark.parametrize(
        ("prompt", "sampling_params", "expected"),
        [
            (
                {"prompt": "A painted robot", "modalities": ["image"]},
                make_sampling_params(num_outputs_per_prompt=2),
                {"key": "image", "is_t2i": True, "flow": [3.0], "steps": [50, 50], "frames": 1},
            ),
            (
                "A warehouse robot",
                make_sampling_params(),
                {"key": "video", "is_t2i": False, "flow": [1.0], "steps": [35], "frames": 189},
            ),
        ],
    )
    def test_forward_defaults_and_mode_selection(
        self,
        make_cosmos3_pipeline,
        prompt,
        sampling_params,
        expected,
    ) -> None:
        pipeline = make_cosmos3_pipeline()
        captured = self._install_forward_stubs(pipeline)

        output = pipeline.forward(SimpleNamespace(prompts=[prompt], sampling_params=sampling_params))

        assert expected["key"] in output.output
        assert captured["format"]["is_t2i"] is expected["is_t2i"]
        assert captured["format"]["num_frames"] == expected["frames"]
        assert captured["flow_shifts"] == expected["flow"]
        assert captured["scheduler_steps"] == expected["steps"]

    def test_forward_i2v_sound_and_action_routes(self, make_cosmos3_pipeline) -> None:
        pipeline = make_cosmos3_pipeline()
        captured = self._install_forward_stubs(pipeline)
        image_tensor = torch.zeros(1, 3, 16, 16)
        velocity_mask = torch.ones(1, 1, 1, 1, 1)

        pipeline._prepare_latents_i2v = lambda *args, **kwargs: (
            torch.zeros(1, 2, 1, 1, 1),
            velocity_mask,
            torch.zeros(1, 2, 1, 1, 1),
        )
        pipeline.forward(
            SimpleNamespace(
                prompts=[
                    {
                        "prompt": "move",
                        "modalities": ["video"],
                        "additional_information": {"preprocessed_image": image_tensor},
                    }
                ],
                sampling_params=make_sampling_params(height=16, width=16, num_frames=5),
            )
        )
        assert captured["diffuse_calls"][-1]["shared_kwargs"]["noisy_frame_mask"] is velocity_mask

        video_tensor = torch.zeros(1, 3, 5, 16, 16)
        v2v_condition = torch.full((1, 2, 2, 1, 1), 4.0)
        v2v_mask = torch.tensor([[[[[0.0]], [[1.0]]]]])
        pipeline._prepare_latents_v2v = lambda *args, **kwargs: (
            torch.zeros(1, 2, 2, 1, 1),
            v2v_mask,
            v2v_condition,
        )
        pipeline.forward(
            SimpleNamespace(
                prompts=[
                    {
                        "prompt": "continue",
                        "modalities": ["video"],
                        "additional_information": {
                            "preprocessed_video": video_tensor,
                            "condition_frame_indexes_vision": [0],
                        },
                    }
                ],
                sampling_params=make_sampling_params(height=16, width=16, num_frames=5),
            )
        )
        assert captured["flow_shifts"][-1] == 10.0
        assert captured["format"]["negative_prompt"] == ""
        assert captured["diffuse_calls"][-1]["shared_kwargs"]["noisy_frame_mask"] is v2v_mask
        assert captured["diffuse_calls"][-1]["condition_latents"] is v2v_condition

        pipeline.transformer = pipeline.transformer.__class__(latent_channel_size=2, sound_gen=True, sound_dim=3)
        sound_latents = torch.zeros(1, 3, 4)
        pipeline._resolve_sound_target_samples = lambda *args: (20, 2.0, 10)
        pipeline._prepare_sound_latents = lambda *args: (sound_latents, 4)
        pipeline._decode_sound_latents = lambda *args: torch.ones(1, 2, 20)
        output = pipeline.forward(
            SimpleNamespace(
                prompts=[{"prompt": "A robot", "modalities": ["video"], "generate_sound": True}],
                sampling_params=make_sampling_params(num_frames=9, frame_rate=3.0),
            )
        )
        assert captured["diffuse_calls"][-1]["sound_latents"] is sound_latents
        assert output.output["audio_sample_rate"] == 10

        pipeline.transformer = pipeline.transformer.__class__(latent_channel_size=2, action_gen=True, action_dim=4)
        output = pipeline.forward(
            SimpleNamespace(
                prompts=[
                    {
                        "prompt": "Pick the block.",
                        "modalities": ["video"],
                        "additional_information": {"preprocessed_image": image_tensor},
                    }
                ],
                sampling_params=make_sampling_params(
                    height=16,
                    width=16,
                    extra_args={
                        "action_mode": "policy",
                        "action_chunk_size": 2,
                        "raw_action_dim": 2,
                        "domain_name": "bridge_orig_lerobot",
                    },
                ),
            )
        )
        assert captured["diffuse_calls"][-1]["shared_kwargs"]["action_domain_ids"].tolist() == [7]
        assert output.custom_output["action"].shape == (1, 2, 2)

    @pytest.mark.parametrize(
        ("prompt", "sampling_params", "message"),
        [
            (["one", "two"], make_sampling_params(), "single prompt"),
            ([{"prompt": "one", "modalities": ["image", "video"]}], make_sampling_params(), "both image and video"),
            (
                [{"prompt": "x", "modalities": ["image"], "generate_sound": True}],
                make_sampling_params(),
                "only for video",
            ),
        ],
    )
    def test_forward_rejects_invalid_public_requests(
        self,
        make_cosmos3_pipeline,
        prompt,
        sampling_params,
        message,
    ) -> None:
        pipeline = make_cosmos3_pipeline()
        pipeline.transformer = pipeline.transformer.__class__(latent_channel_size=2, sound_gen=True, sound_dim=3)

        with pytest.raises(ValueError, match=message):
            pipeline.forward(SimpleNamespace(prompts=prompt, sampling_params=sampling_params))
