# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.cosmos3 import transformer_cosmos3 as cosmos3

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _output_model(dtype: torch.dtype = torch.float32) -> cosmos3.Cosmos3VFMTransformer:
    # Only the output head is needed; do not construct weights or TP layers.
    model = object.__new__(cosmos3.Cosmos3VFMTransformer)
    nn.Module.__init__(model)
    model.norm_moe_gen = cosmos3.RMSNorm(8, eps=1e-6, dtype=dtype)
    model.norm_moe_gen.weight.data.copy_(torch.linspace(0.5, 1.5, 8, dtype=dtype))
    model.proj_out = nn.Linear(8, 3, dtype=dtype)
    model._output_projection_chunk_size = 8
    return model


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("sequence_length", [3, 8, 19])
@torch.inference_mode()
def test_output_projection_matches_native_norm_with_bounded_chunks(dtype, batch, sequence_length) -> None:
    torch.manual_seed(42)
    model = _output_model(dtype)
    # A target slice can retain the strides of a larger control+target tensor.
    packed = torch.randn(batch, sequence_length + 3, 8, dtype=dtype) * 20
    hidden = packed[:, 3:]
    original = packed.clone()
    normalized = model.norm_moe_gen.forward_native(hidden)
    expected = model.proj_out(normalized)
    norm_chunks = []

    def record_norm(module, args, output):
        assert args[0].numel() // args[0].shape[-1] <= model._output_projection_chunk_size
        norm_chunks.append(output.clone())

    model.norm_moe_gen.register_forward_hook(record_norm)
    actual = model._project_video_tokens(hidden)

    torch.testing.assert_close(actual, expected)
    # Chunking must preserve the FP32 norm arithmetic even for BF16 input.
    torch.testing.assert_close(torch.cat(norm_chunks, dim=1), normalized, rtol=0, atol=0)
    torch.testing.assert_close(packed, original, rtol=0, atol=0)
    assert actual.dtype == expected.dtype


@pytest.mark.parametrize("sequence_length", [3, 19])
@torch.inference_mode()
def test_output_projection_preserves_autocast_dtype(sequence_length) -> None:
    torch.manual_seed(42)
    model = _output_model()
    hidden = torch.randn(1, sequence_length, 8)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        expected = model.proj_out(model.norm_moe_gen.forward_native(hidden))
        actual = model._project_video_tokens(hidden)
    assert actual.dtype == expected.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected)


class _RecordingGenLayer(nn.Module):
    def forward(self, hidden, **kwargs):
        self.output = hidden.flip(1) + 0.25
        return self.output


@pytest.mark.parametrize("controls,modality", [(0, None), (1, None), (2, None), (0, "action"), (0, "sound")])
@torch.inference_mode()
def test_forward_output_matches_full_norm_and_skips_controls(monkeypatch, controls, modality) -> None:
    torch.manual_seed(42)
    monkeypatch.setattr(cosmos3, "_get_ulysses_state", lambda: (1, 0, None))
    monkeypatch.setattr(cosmos3, "current_omni_platform", SimpleNamespace(device_type="cpu"))
    model = _output_model()
    model._output_projection_chunk_size = 5
    model.latent_channel_size = 2
    model.latent_patch_size = 2
    model.proj_in = nn.Linear(8, 8)
    model.proj_out = nn.Linear(8, 8)
    model.time_embedder = cosmos3.TimestepEmbedder(8, frequency_embedding_size=16)
    model.timestep_scale = 0.001
    layer = _RecordingGenLayer()
    model.gen_layers = nn.ModuleList([layer])
    model.gen_sp_prepare = cosmos3.Cosmos3GenSPPrepare()
    model.gen_sp_gather = nn.Identity()
    model.cached_kv = [(torch.zeros(1), torch.zeros(1))]
    model.cached_freqs_gen = (torch.zeros(1), torch.zeros(1))
    model.action_gen = modality == "action"
    model.sound_gen = modality == "sound"
    extra = {}
    if model.action_gen:
        model.action_dim = 3
        model.action_proj_in = cosmos3.DomainAwareLinear(3, 8, 2, dtype=torch.float32)
        model.action_proj_out = cosmos3.DomainAwareLinear(8, 3, 2, dtype=torch.float32)
        model.action_modality_embed = nn.Parameter(torch.zeros(8))
        extra = {"action_latents": torch.randn(1, 3, 3), "action_domain_ids": torch.tensor([1])}
    if model.sound_gen:
        model.sound_dim = 3
        model.audio_proj_in = nn.Linear(3, 8)
        model.audio_proj_out = nn.Linear(8, 3)
        model.audio_modality_embed = nn.Parameter(torch.zeros(8))
        extra = {"sound_latents": torch.randn(1, 3, 3)}

    hidden = torch.randn(1, 2, 2, 3, 5)
    embedding_refs = []
    original_cat = torch.cat

    def record_packed_embeddings(tensors, dim=0, *, out=None):
        if not embedding_refs and dim == 1 and all(tensor.ndim == 3 for tensor in tensors):
            embedding_refs.extend(weakref.ref(tensor) for tensor in tensors)
        return original_cat(tensors, dim=dim, out=out)

    def check_embeddings_released(module, args):
        # Keep only weak references so this check cannot extend tensor lifetimes.
        assert len(embedding_refs) == controls + 1 + (modality is not None)
        assert all(ref() is None for ref in embedding_refs)

    monkeypatch.setattr(torch, "cat", record_packed_embeddings)
    layer.register_forward_pre_hook(check_embeddings_released)
    norm_inputs = []
    model.norm_moe_gen.register_forward_pre_hook(lambda module, args: norm_inputs.append(args[0].clone()))
    actual = model(
        hidden_states=hidden,
        timestep=torch.tensor([1.0]),
        text_ids=torch.tensor([[1, 2]]),
        text_mask=torch.ones(1, 2, dtype=torch.long),
        video_shape=(2, 3, 5),
        control_latents=[torch.randn_like(hidden) for _ in range(controls)],
        **extra,
    )

    video_tokens = 12  # Two latent frames, each padded to 2 x 3 spatial patches.
    target_start = controls * video_tokens
    # Reference the previous output path: normalize the entire packed sequence,
    # discard controls, then project each output modality in one operation.
    full_norm = model.norm_moe_gen.forward_native(layer.output)
    expected_video = model.unpatchify(model.proj_out(full_norm[:, target_start : target_start + video_tokens]), 2, 3, 5)
    if modality == "action":
        expected = (
            expected_video,
            model.unpack_action(model.action_proj_out(full_norm[:, video_tokens:], extra["action_domain_ids"])),
        )
    elif modality == "sound":
        expected = (expected_video, model.unpack_sound(model.audio_proj_out(full_norm[:, video_tokens:])))
    else:
        expected = expected_video
    torch.testing.assert_close(actual, expected)
    # No control tokens reach the output norm, and camera/token order is retained.
    torch.testing.assert_close(torch.cat(norm_inputs, dim=1), layer.output[:, target_start:], rtol=0, atol=0)
    assert max(chunk.shape[1] for chunk in norm_inputs) <= 5
