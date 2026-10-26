# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contracts that also run without importing the CUDA/vLLM runtime.

Standalone: python -m pytest --noconftest -c /dev/null <this file>
The source modules are loaded directly so this suite exercises the real helpers
on development machines without a vLLM binary.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]
ROOT = Path(__file__).resolve().parents[4]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


conditioning = _load("shared_und_conditioning", "vllm_omni/diffusion/models/cosmos3/conditioning.py")
external = _load("shared_und_external", "vllm_omni/diffusion/models/cosmos3/external_und.py")
tracker_module = _load("shared_und_tracker", "vllm_omni/engine/cfg_companion_tracker.py")
capabilities = _load("shared_und_capabilities", "vllm_omni/outputs/stage_capabilities.py")


@dataclass
class Tokenizer:
    eos_token_id: int = 10001
    conversations: list = field(default_factory=list)

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize and add_generation_prompt
        self.conversations.append(messages)
        return [ord(c) for c in json.dumps(messages)]

    def convert_tokens_to_ids(self, token):
        assert token == "<|vision_start|>"
        return 10002


@dataclass
class Params:
    extra_args: dict = field(default_factory=dict)


class Formatter(conditioning.Cosmos3Conditioning):
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.device = torch.device("cpu")


@pytest.mark.parametrize("initialized,rank", [(False, None), (True, 0), (True, 1)])
def test_conditioning_warnings_only_on_rank_zero(monkeypatch, caplog, initialized, rank):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: initialized)

    def get_rank():
        assert initialized
        return rank

    monkeypatch.setattr(torch.distributed, "get_rank", get_rank)
    formatter = Formatter()
    formatter._format_and_tokenize_prompts(
        '{"caption":"A robot","aspect_ratio":"4,3"}', "", 48, 24, 720, 1280, 17, Params()
    )
    prompt = json.loads(formatter.tokenizer.conversations[0][0]["content"])
    assert prompt["aspect_ratio"] == "16,9"
    messages = [record.getMessage() for record in caplog.records if record.name == conditioning.logger.name]
    should_warn = not initialized or rank == 0
    assert any("conflicts with the generated 1280x720 canvas" in message for message in messages) == should_warn
    assert any("shortened to max_sequence_length" in message for message in messages) == should_warn


@pytest.mark.parametrize(
    "image,frames,expected", [(True, 1, {"H": 1024, "W": 1024}), (False, 189, {"H": 720, "W": 1280})]
)
def test_json_metadata_and_special_tokens(image, frames, expected):
    formatter = Formatter()
    ids, mask, neg, neg_mask = formatter._format_and_tokenize_prompts(
        '{"caption":"A crane", "fps": 1, "duration": "old"}',
        "",
        frames,
        24,
        expected["H"],
        expected["W"],
        4096,
        Params(),
        True,
        is_t2i=image,
    )
    messages = formatter.tokenizer.conversations[0]
    expected_system = conditioning.COSMOS3_T2I_SYSTEM_PROMPT if image else conditioning.COSMOS3_SYSTEM_PROMPT
    assert messages[0] == {"role": "system", "content": expected_system}
    obj = json.loads(messages[1]["content"])
    assert obj["resolution"] == expected
    if image:
        assert "duration" not in obj and "fps" not in obj
    else:
        assert obj["duration"] == "7s" and obj["fps"] == 24.0
    assert ids[0, -2:].tolist() == [10001, 10002]
    assert neg[0, -2:].tolist() == [10001, 10002]
    assert mask.tolist() == [[1] * ids.shape[1]]
    assert neg_mask.tolist() == [[1] * neg.shape[1]]
    assert ids.shape[1] != neg.shape[1]


@pytest.mark.parametrize("mode", ["none", "same", "inverse"])
def test_negative_metadata_and_truncation(mode):
    formatter = Formatter()
    sp = Params({"use_duration_template": True, "use_resolution_template": True, "negative_metadata_mode": mode})
    positive, _, negative, _ = formatter._format_and_tokenize_prompts(
        "A crane.",
        "",
        5,
        24,
        256,
        512,
        17,
        sp,
    )
    assert positive.shape == negative.shape == (1, 19)
    assert positive[0, -2:].tolist() == [10001, 10002]
    pos_text = formatter.tokenizer.conversations[0][0]["content"]
    neg_text = formatter.tokenizer.conversations[1][0]["content"]
    assert pos_text == "A crane. The video is 0.2 seconds long and is of 24 FPS. This video is of 256x512 resolution."
    if mode == "none":
        assert neg_text == ""
    elif mode == "same":
        assert neg_text == "The video is 0.2 seconds long and is of 24 FPS. This video is of 256x512 resolution."
    else:
        assert (
            neg_text
            == "The video is not 0.2 seconds long and is not of 24 FPS. This video is not of 256x512 resolution."
        )


@dataclass
class DenseKV:
    key_cache: list
    value_cache: list


def _export(length=17):
    cache = DenseKV([torch.randn(length, 2, 4) for _ in range(3)], [torch.randn(length, 2, 4) for _ in range(3)])
    meta = {
        "seq_len": length,
        "num_layers": 3,
        "layer_order": [0, 1, 2],
        "dtype": "float32",
        "cosmos3_conditioning_length": length,
        "confirmed_computed": length,
    }
    return cache, meta


def _validate(cache, meta, length=17):
    return external.validate_branch(cache, meta, length, num_layers=3, num_heads=2, head_dim=4, dtype=torch.float32)


@pytest.mark.parametrize("revision", [None, "other-revision", "pinned-revision"])
def test_export_checkpoint_revision_must_match_gen(revision):
    cache, meta = _export()
    checkpoint = {"model": "nano", "revision": "pinned-revision"}
    if revision is not None:
        meta["cosmos3_checkpoint"] = {"model": "nano", "revision": revision}
    kwargs = dict(num_layers=3, num_heads=2, head_dim=4, dtype=torch.float32, checkpoint=checkpoint)
    if revision == "pinned-revision":
        assert external.validate_branch(cache, meta, 17, **kwargs).expected_length == 17
    else:
        with pytest.raises(ValueError, match="checkpoint revision"):
            external.validate_branch(cache, meta, 17, **kwargs)


@pytest.mark.parametrize("length", [15, 16, 17, 31, 32, 33])
def test_dense_import_layout_is_exact_and_preserves_keys(length):
    cache, meta = _export(length)
    branch = _validate(cache, meta, length)
    assert branch.expected_length == length
    for layer, (key, value) in enumerate(branch.kv):
        assert key.shape == value.shape == (1, length, 2, 4)
        torch.testing.assert_close(key[0], cache.key_cache[layer], rtol=0, atol=0)
        torch.testing.assert_close(value[0], cache.value_cache[layer], rtol=0, atol=0)
    _, mask = branch.text_tensors(torch.device("cpu"))
    assert mask.sum().item() == length


@pytest.mark.parametrize(
    "fault",
    ["short", "long", "metadata", "layer", "order", "none", "heads", "dtype", "dtype_meta", "uncomputed", "canonical"],
)
def test_reject_invalid_exports(fault):
    cache, meta = _export()
    if fault in {"short", "long"}:
        cache.key_cache[1] = torch.zeros(16 if fault == "short" else 18, 2, 4)
    elif fault == "metadata":
        meta["seq_len"] = 18
    elif fault == "layer":
        cache.value_cache.pop()
    elif fault == "order":
        meta["layer_order"] = [1, 0, 2]
    elif fault == "none":
        cache.key_cache[1] = None
    elif fault == "heads":
        cache.key_cache[1] = torch.zeros(17, 1, 4)
    elif fault == "dtype":
        cache.key_cache[1] = cache.key_cache[1].bfloat16()
    elif fault == "dtype_meta":
        meta["dtype"] = "float16"
    elif fault == "uncomputed":
        meta["confirmed_computed"] = 16
    else:
        meta["cosmos3_conditioning_length"] = 18
    with pytest.raises(ValueError):
        _validate(cache, meta)


@dataclass
class Clock:
    now: float = 10.0

    def __call__(self):
        return self.now


def test_deadline_is_monotonic_nonrenewing_and_cleanup_ignores_late_outputs():
    clock = Clock()
    tracker = tracker_module.CfgCompanionTracker(300, clock)
    tracker.register_companion("parent", "negative", "negative")
    tracker.defer_parent("parent", "first", 0)
    clock.now = 309
    tracker.defer_parent("parent", "second", 0)
    assert not tracker.expired("parent")
    clock.now = 310
    assert tracker.expired("parent")
    assert tracker.cleanup_parent("parent") == ["negative"]
    tracker.set_companion_output("negative", "late")
    assert tracker.on_companion_completed("negative") is None
    assert tracker.pending_parents() == ()
    assert not tracker._companion_outputs
    assert not tracker._deadlines


def test_completion_requires_processed_output_and_deadline_does_not_gate_admission():
    tracker = tracker_module.CfgCompanionTracker()
    tracker.register_companion("parent", "negative", "negative")
    assert tracker.on_companion_completed("negative") is None
    assert not tracker.all_companions_done("parent")
    tracker.defer_parent("parent", "output", 0)
    assert not tracker.expired("parent")
    tracker.set_companion_output("negative", "processed")
    assert tracker.on_companion_completed("negative") == "parent"
    assert tracker.get_companion_outputs("parent") == ["processed"]


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_reject_invalid_timeout(timeout):
    with pytest.raises(ValueError):
        tracker_module.CfgCompanionTracker(timeout)


def test_capabilities_preserve_legacy_scalar_and_include_media_variants():
    assert capabilities.stage_output_types({"final_output_type": "text"}) == ("text",)
    assert capabilities.stage_output_types(
        {"final_output_type": "video", "final_output_types": ["image", "video"]}
    ) == ("image", "video")


def _source_definitions(path, names, namespace, *, class_name=None):
    """Exercise CPU definitions without loading their GPU-only import graph."""
    import __future__

    import ast

    tree = ast.parse((ROOT / path).read_text())
    if class_name is not None:
        tree = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    selected = [node for node in tree.body if isinstance(node, ast.ClassDef | ast.FunctionDef) and node.name in names]
    code = compile(
        ast.Module(body=selected, type_ignores=[]), str(ROOT / path), "exec", flags=__future__.annotations.compiler_flag
    )
    exec(code, namespace)
    return namespace


@pytest.mark.parametrize(
    "model_config,expected_frames", [(None, 1), ({}, 1), ({"external_und": False}, 1), ({"external_und": True}, 0)]
)
def test_startup_warmup_policy_requires_local_und(model_config, expected_frames):
    @dataclass
    class Config:
        model_config: dict | None

    policy = _source_definitions(
        "vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py",
        {"get_dummy_run_num_frames"},
        {},
        class_name="Cosmos3OmniDiffusersPipeline",
    )
    pipeline_cls = type("Cosmos3Policy", (), {"get_dummy_run_num_frames": policy["get_dummy_run_num_frames"]})

    class Registry:
        @staticmethod
        def _try_load_model_cls(name):
            assert name == "Cosmos3OmniDiffusersPipeline"
            return pipeline_cls

    resolver = _source_definitions(
        "vllm_omni/diffusion/io_support.py",
        {"get_dummy_run_num_frames", "supports_audio_output"},
        {"DiffusionModelRegistry": Registry},
    )["get_dummy_run_num_frames"]
    assert resolver("Cosmos3OmniDiffusersPipeline", False, Config(model_config)) == expected_frames


@dataclass
class GenerationParams(Params):
    height: int | None = None
    width: int | None = None
    num_frames: int | None = None
    num_outputs_per_prompt: int = 1
    guidance_scale_provided: bool = False
    guidance_scale: float = 6.0
    lora_request: object = None


@dataclass
class ARParams(Params):
    max_tokens: int = 2048
    min_tokens: int = 0

    def clone(self):
        import copy

        return copy.deepcopy(self)


def _admission():
    import copy
    import math

    from PIL import Image

    bagel = _load("shared_und_expanded_prompt", "vllm_omni/model_executor/stage_input_processors/bagel.py")
    scope = dict(vars(conditioning), copy=copy, math=math, Image=Image, ExpandedPrompt=bagel.ExpandedPrompt)
    _source_definitions(
        "vllm_omni/model_executor/stage_input_processors/cosmos3.py",
        {"Cosmos3PromptTransform", "expand_cfg_prompts"},
        scope,
    )
    transform = scope["Cosmos3PromptTransform"].__new__(scope["Cosmos3PromptTransform"])
    transform.device = torch.device("cpu")
    transform.tokenizer = Tokenizer()
    return transform, scope["expand_cfg_prompts"]


@pytest.mark.parametrize("mode,guidance", [("image", 7.0), ("video", 6.0), ("video", 1.0)])
def test_admission_empty_negative_is_independent_and_text_only(mode, guidance):
    transform, expand = _admission()
    original_ar = ARParams()
    original_gen = GenerationParams(guidance_scale_provided=True, guidance_scale=guidance)
    params = [original_ar, original_gen]
    prompt = {"prompt": "A crane", "negative_prompt": "", "modalities": [mode]}
    ar_prompt = transform(prompt, params)
    companions = expand(prompt, params[0])
    assert "prompt" not in ar_prompt and "multi_modal_data" not in ar_prompt
    assert ar_prompt["additional_information"]["omni_wait_for_processed_kv_output"]
    assert params[1].extra_args["cosmos3_expected_lengths"]["positive"] == len(ar_prompt["prompt_token_ids"])
    assert len(companions) == (1 if guidance > 1 else 0)
    if companions:
        assert companions[0].role == "cfg_text"
        assert (
            len(companions[0].prompt["prompt_token_ids"])
            == params[1].extra_args["cosmos3_expected_lengths"]["negative"]
        )
    assert original_ar.max_tokens == 2048 and original_ar.extra_args == {}
    assert original_gen.height is None and original_gen.extra_args == {}
    assert params[0].max_tokens == 1


def test_i2v_resolves_canvas_before_json_tokenization_and_retains_reference():
    from PIL import Image

    transform, _ = _admission()
    image = Image.new("RGB", (400, 200))
    prompt = {"prompt": '{"caption":"A crane"}', "modalities": ["video"], "multi_modal_data": {"image": image}}
    params = [ARParams(), GenerationParams(num_frames=6, extra_args={"frame_rate": 15})]
    ar_prompt = transform(prompt, params)
    assert prompt["multi_modal_data"]["image"] is image
    assert "multi_modal_data" not in ar_prompt
    gen = params[1]
    assert gen.num_frames == 9 and gen.height == 672 and gen.width == 1344
    text = json.loads(transform.tokenizer.conversations[0][0]["content"])
    assert text["resolution"] == {"H": 672, "W": 1344}
    assert text["fps"] == gen.extra_args["resolved_frame_rate"] == 15


def test_reasoning_preserves_native_multimodal_input():
    transform, _ = _admission()
    prompt = {"prompt": "Describe this", "modalities": ["text"], "multi_modal_data": {"image": object()}}
    params = [ARParams(), GenerationParams()]
    assert transform(prompt, params) is prompt
    assert params[0].max_tokens == 2048


@pytest.mark.parametrize("length,frames,hp,wp,fps", [(5, 1, 2, 2, 24), (19, 3, 3, 2, 15), (7, 1, 1, 3, 30)])
def test_branch_rotary_has_text_length_dependent_temporal_offset(length, frames, hp, wp, fps):
    import math

    from torch import nn

    scope = {"torch": torch, "nn": nn, "math": math}
    _source_definitions(
        "vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py",
        {
            "Qwen3VLTextRotaryEmbedding",
            "compute_mrope_position_ids_text",
            "compute_mrope_position_ids_vision",
        },
        scope,
    )
    rotary = scope["Qwen3VLTextRotaryEmbedding"](head_dim=8, rope_theta=10000, mrope_section=[2, 1, 1])
    text, offset = scope["compute_mrope_position_ids_text"](length, 0)
    assert offset == length
    assert torch.equal(text[0], torch.arange(length))
    positions, _ = scope["compute_mrope_position_ids_vision"](
        frames,
        hp,
        wp,
        offset + 15000,
        fps=fps if frames > 1 else None,
    )
    assert positions[0, 0] == length + 15000
    shifted, _ = scope["compute_mrope_position_ids_vision"](
        frames,
        hp,
        wp,
        offset + 15004,
        fps=fps if frames > 1 else None,
    )
    cos, sin = rotary(torch.empty(0), positions.unsqueeze(1))
    shifted_cos, shifted_sin = rotary(torch.empty(0), shifted.unsqueeze(1))
    assert not torch.equal(cos, shifted_cos)
    assert not torch.equal(sin, shifted_sin)


def test_external_frequencies_use_execution_dtype_and_independent_lengths():
    calls = []

    class Transformer:
        @staticmethod
        def _pad_to_patch_size(h, w):
            return (h + 1) // 2, (w + 1) // 2, h, w

        @staticmethod
        def _compute_rope_freqs(mask, frames, hp, wp, fps, device, dtype):
            calls.append((mask.sum().item(), frames, hp, wp, fps, dtype))
            freq = torch.zeros(1, frames * hp * wp, 1, 4, device=device, dtype=dtype)
            return (None, None), (freq, freq)

    branches = {False: external.ExternalBranch(17, []), True: external.ExternalBranch(5, [])}
    external.prepare_branch_frequencies(
        Transformer(), branches, torch.zeros(1, 2, 3, 6, 4, dtype=torch.float32), 15, dtype=torch.bfloat16
    )
    assert calls == [(17, 3, 3, 2, 15, torch.bfloat16), (5, 3, 3, 2, 15, torch.bfloat16)]
    assert all(branch.freqs[0].dtype == torch.bfloat16 for branch in branches.values())
