# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref

import pytest
import torch
from torch import nn
from torch.multiprocessing.reductions import StorageWeakRef

from vllm_omni.diffusion.distributed.sp_plan import (
    SequenceParallelConfig,
    SequenceParallelInput,
    SequenceParallelPartialInput,
)
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.hooks.base import HookRegistry
from vllm_omni.diffusion.hooks.sequence_parallel import SequenceParallelSplitHook, apply_sequence_parallel

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture
def parallel_state(monkeypatch):
    from vllm_omni.diffusion.attention import selector
    from vllm_omni.diffusion.distributed import parallel_state, sp_sharding

    class MaskBackend:
        @staticmethod
        def supports_attention_mask(spec):
            return True

    def configure(size=4, rank=1):
        monkeypatch.setattr(parallel_state, "get_sequence_parallel_world_size", lambda: size)
        monkeypatch.setattr(parallel_state, "get_sequence_parallel_rank", lambda: rank)
        monkeypatch.setattr(parallel_state, "get_ring_parallel_world_size", lambda: 1)
        monkeypatch.setattr(sp_sharding, "get_sequence_parallel_world_size", lambda: size)
        monkeypatch.setattr(sp_sharding, "get_sequence_parallel_rank", lambda: rank)
        monkeypatch.setattr(selector, "get_attn_backend_for_capability", lambda **kwargs: MaskBackend())

    configure()
    return configure


@pytest.mark.parametrize("sequence_length", [16, 17])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("rank", [0, 3])
@torch.inference_mode()
def test_cosmos3_real_sp_hook_releases_full_embedding(parallel_state, sequence_length, batch, rank):
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import Cosmos3GenSPPrepare, Cosmos3VFMTransformer

    parallel_state(rank=rank)
    model = nn.Module()
    model.gen_sp_prepare = Cosmos3GenSPPrepare()
    metadata = Cosmos3VFMTransformer._sp_plan["gen_sp_prepare"]
    assert metadata[0].clone_shard
    assert not metadata[1].clone_shard and not metadata[2].clone_shard
    apply_sequence_parallel(model, SequenceParallelConfig(ulysses_degree=4), {"gen_sp_prepare": metadata})

    def prepare():
        hidden = torch.arange(batch * sequence_length * 8, dtype=torch.bfloat16).reshape(batch, sequence_length, 8)
        full_ref = weakref.ref(hidden)
        full_pointer = hidden.untyped_storage().data_ptr()
        cos = torch.ones(batch, sequence_length, 1, 2)
        sin = torch.zeros_like(cos)
        return model.gen_sp_prepare(hidden, cos, sin), full_ref, full_pointer

    for _ in range(2):
        with set_forward_context():
            (hidden, cos, sin), full_ref, full_pointer = prepare()
            # Check the real backing allocation, not only Tensor.data_ptr():
            # nonzero-offset and inference-mode views can hide the shared base.
            assert full_ref() is None
            assert hidden.untyped_storage().data_ptr() != full_pointer
            assert hidden.untyped_storage().nbytes() == hidden.numel() * hidden.element_size()
            assert hidden.is_contiguous()
            expected = torch.arange(batch * sequence_length * 8, dtype=torch.bfloat16).reshape(
                batch, sequence_length, 8
            )
            padding = -sequence_length % 4
            expected = torch.nn.functional.pad(expected, (0, 0, 0, padding)).chunk(4, dim=1)[rank]
            torch.testing.assert_close(hidden, expected, rtol=0, atol=0)
            assert cos.shape[1] == sin.shape[1] == hidden.shape[1]

            # Once the next layer replaces the shard, the hook must not keep
            # even the smaller initial activation alive through later layers.
            shard_ref = weakref.ref(hidden)
            hidden = hidden + 1
            assert shard_ref() is None


@pytest.mark.parametrize("sequence_length", [16, 17])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("world_size", [1, 4])
@pytest.mark.parametrize("grouped_layers", [False, True])
@torch.inference_mode()
def test_cosmos3_forward_releases_prepared_embedding(
    monkeypatch, parallel_state, sequence_length, batch, world_size, grouped_layers
):
    from vllm_omni.diffusion.models.cosmos3 import transformer_cosmos3

    parallel_state(size=world_size, rank=0)
    monkeypatch.setattr(transformer_cosmos3, "_get_ulysses_state", lambda: (world_size, 0, None))
    layer_calls = []

    class RecordingLayer(nn.Module):
        def __init__(self, index):
            super().__init__()
            self.index = index

        def forward(self, hidden, **kwargs):
            layer_calls.append(self.index)
            # SP releases the full allocation before the first layer. Without
            # SP, replacing that layer's input must release it by the next one.
            if world_size > 1 or self.index > 0:
                assert model.full_ref() is None
                assert model.full_storage.expired()
            if world_size > 1 and self.index == 0:
                assert hidden.untyped_storage().data_ptr() != model.full_pointer
                assert hidden.untyped_storage().nbytes() == hidden.numel() * hidden.element_size()
            hidden = hidden + self.index + 1
            return (hidden,) if grouped_layers else hidden

    class Gather(nn.Module):
        def forward(self, hidden):
            # Tokens are identical within each sample. Repeating rank 0's
            # shard and trimming padding simulates the collective on CPU.
            return hidden.repeat(1, world_size, 1)[:, :sequence_length]

    class TinyCosmos3(transformer_cosmos3.Cosmos3VFMTransformer):
        def __init__(self):
            # Exercise the real preprocess/stack/postprocess without loading
            # checkpoint weights, attention kernels, or distributed groups.
            nn.Module.__init__(self)
            self.latent_patch_size = 1
            self.latent_channel_size = 8
            self.timestep_scale = 1.0
            self.time_embedder = lambda t: t.unsqueeze(1).expand(-1, 8)
            self.proj_in = nn.Identity()
            self.proj_out = nn.Identity()
            self.norm_moe_gen = nn.Identity()
            self.gen_sp_prepare = transformer_cosmos3.Cosmos3GenSPPrepare()
            self.gen_sp_gather = Gather()
            self.gen_layers = nn.ModuleList([RecordingLayer(0), RecordingLayer(1)])
            # A different cache/block count selects the grouped cache-dit path.
            self.cached_kv = [(torch.empty(0), torch.empty(0))] * (1 if grouped_layers else 2)
            self.cached_freqs_gen = (
                torch.ones(batch, sequence_length, 1, 2),
                torch.zeros(batch, sequence_length, 1, 2),
            )

        def _gen_preprocess(self, *args, **kwargs):
            prep = super()._gen_preprocess(*args, **kwargs)
            self.full_ref = weakref.ref(prep.hidden_gen)
            self.full_storage = StorageWeakRef(prep.hidden_gen.untyped_storage())
            self.full_pointer = prep.hidden_gen.untyped_storage().data_ptr()
            return prep

    model = TinyCosmos3()
    if world_size > 1:
        apply_sequence_parallel(
            model,
            SequenceParallelConfig(ulysses_degree=world_size),
            {"gen_sp_prepare": model._sp_plan["gen_sp_prepare"]},
        )
    video = torch.arange(batch * 8, dtype=torch.bfloat16).reshape(batch, 8, 1, 1, 1)
    video = video.expand(-1, -1, 1, 1, sequence_length)
    for _ in range(2):
        with set_forward_context():
            prediction = model(
                hidden_states=video,
                timestep=torch.ones(batch),
                text_ids=torch.ones(batch, 2, dtype=torch.long),
                text_mask=torch.ones(batch, 2, dtype=torch.long),
                video_shape=(1, 1, sequence_length),
            )
        torch.testing.assert_close(prediction, video + 4, rtol=0, atol=0)
        assert model.full_storage.expired()
    assert layer_calls == [0, 1, 0, 1]


@pytest.mark.parametrize("clone_shard", [False, True])
@pytest.mark.parametrize("world_size", [1, 4])
def test_shard_storage_policy_is_opt_in_and_skips_single_rank(parallel_state, clone_shard, world_size):
    parallel_state(size=world_size, rank=0)
    hook = SequenceParallelSplitHook({}, SequenceParallelConfig(ulysses_degree=4))
    tensor = torch.arange(128).reshape(1, 16, 8)
    spec = SequenceParallelInput(split_dim=1, clone_shard=clone_shard)
    with set_forward_context():
        shard = hook._prepare_sp_input(tensor, spec)
    torch.testing.assert_close(shard, tensor.chunk(world_size, dim=1)[0])
    owns_storage = shard.untyped_storage().data_ptr() != tensor.untyped_storage().data_ptr()
    assert owns_storage == (clone_shard and world_size > 1)
    if world_size == 1:
        assert shard is tensor


def test_owned_shard_preserves_gradients(parallel_state):
    hook = SequenceParallelSplitHook({}, SequenceParallelConfig(ulysses_degree=4))
    tensor = torch.randn(2, 16, 8, requires_grad=True)
    with set_forward_context():
        shard = hook._prepare_sp_input(tensor, SequenceParallelInput(split_dim=1, clone_shard=True))
        shard.sum().backward()
    expected = torch.zeros_like(tensor)
    expected[:, 4:8] = 1
    torch.testing.assert_close(tensor.grad, expected, rtol=0, atol=0)


@pytest.mark.parametrize("phase", ["success", "non_tensor", "forward_error", "split_error"])
@pytest.mark.parametrize("keyword", [False, True])
@torch.inference_mode()
def test_split_hook_does_not_retain_arguments_on_any_exit(parallel_state, phase, keyword):
    class Module(nn.Module):
        def forward(self, x):
            if phase == "forward_error":
                raise RuntimeError("forward failed")
            return None if phase == "non_tensor" else x

    module = Module()
    index = 1 if phase == "split_error" else 0
    hook = SequenceParallelSplitHook(
        {index: SequenceParallelInput(split_dim=1, split_output=True, clone_shard=True)},
        SequenceParallelConfig(ulysses_degree=4),
    )
    HookRegistry.get_or_create(module).register_hook("sp", hook)
    refs = []

    def run():
        tensor = torch.randn(1, 16, 8)
        refs.append(weakref.ref(tensor))
        return module(x=tensor) if keyword else module(tensor)

    with set_forward_context():
        if phase.endswith("error"):
            with pytest.raises((RuntimeError, ValueError)):
                run()
        else:
            result = run()
            assert (result is None) == (phase == "non_tensor")
    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize("keyword", [False, True])
@pytest.mark.parametrize("tensor_source", [False, True])
@torch.inference_mode()
def test_partial_output_split_keeps_only_lengths_across_calls(parallel_state, keyword, tensor_source):
    class Module(nn.Module):
        def forward(self, x, text):
            return x

    module = Module()
    hook = SequenceParallelSplitHook(
        {0: SequenceParallelPartialInput(split_dim=0, text_len_source="text", split_output=True)},
        SequenceParallelConfig(ulysses_degree=4),
    )
    HookRegistry.get_or_create(module).register_hook("sp", hook)
    refs = []

    def run(text_length):
        tensor = torch.arange(text_length + 8).reshape(-1, 1)
        text = torch.zeros(text_length, 8) if tensor_source else text_length
        refs.append(weakref.ref(tensor))
        if tensor_source:
            refs.append(weakref.ref(text))
        return module(x=tensor, text=text) if keyword else module(tensor, text)

    for text_length in (2, 4):
        with set_forward_context():
            result = run(text_length)
        expected = torch.cat([torch.arange(text_length), torch.arange(text_length + 2, text_length + 4)]).reshape(-1, 1)
        torch.testing.assert_close(result, expected)
        assert all(ref() is None for ref in refs)
        assert hook._text_len_cache == {"text": text_length}
