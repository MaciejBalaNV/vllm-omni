# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA encoder parity/layout tests; run on both H100 and GB200."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from diffusers.models.autoencoders import AutoencoderKLWan
from diffusers.models.autoencoders.autoencoder_kl_wan import AvgDown3D, WanResample

from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath import (
    encode_frames,
    install_wan_vae_encoder_fastpath,
)
from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath import encoder_forwards as ef
from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath import forwards as fp
from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath import triton_downsample as down

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]
DTYPES = [torch.bfloat16, torch.float16, torch.float32]
CONFIG = dict(
    base_dim=20,
    decoder_base_dim=32,
    z_dim=48,
    dim_mult=[1, 2, 4, 4],
    num_res_blocks=2,
    temperal_downsample=[False, True, True],
    is_residual=True,
    patch_size=2,
    in_channels=12,
    out_channels=12,
    scale_factor_temporal=4,
    scale_factor_spatial=16,
)


def bits_equal(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    integer = torch.int16 if a.element_size() == 2 else torch.int32
    assert torch.equal(a.contiguous().view(integer), b.contiguous().view(integer))


@torch.no_grad()
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", ["contiguous", "channels_last", "frame_major", "strided", "channels_last_strided"])
@pytest.mark.parametrize("batch,frames", [(1, 1), (1, 4), (2, 4)])
def test_spatial_pad_matches_reference(dtype, layout, batch, frames):
    x = torch.randn(batch, 160, frames, 12, 20, device="cuda", dtype=dtype)
    if layout in ("channels_last", "channels_last_strided"):
        x = x.contiguous(memory_format=torch.channels_last_3d)
        if layout == "channels_last_strided":
            x = x[:, :, :, 1::2, ::2]
    elif layout == "frame_major":
        x = x.permute(0, 2, 1, 3, 4).contiguous().permute(0, 2, 1, 3, 4)
    elif layout == "strided":
        x = x[:, :, :, 1::2, ::2]
    expected = F.pad(fp._merge_batch_and_frames(x), (0, 1, 0, 1))
    actual = down.spatial_downsample_input(x)
    assert actual is not None
    bits_equal(actual, expected)
    if layout.startswith("channels_last"):
        assert actual.is_contiguous(memory_format=torch.channels_last)
    else:
        assert actual.stride() == expected.stride()


@torch.no_grad()
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize("frames", [1, 4])
@pytest.mark.parametrize("cin,cout,ft,fs", [(160, 160, 1, 2), (160, 320, 2, 2), (320, 640, 2, 2), (640, 640, 1, 1)])
def test_average_shortcuts_match_cosmos3_grouping(dtype, channels_last, frames, cin, cout, ft, fs):
    shortcut = AvgDown3D(cin, cout, ft, fs)
    source = torch.randn(2, cin, frames, 8, 12, device="cuda", dtype=dtype)
    if channels_last:
        source = source.contiguous(memory_format=torch.channels_last_3d)
    shape = (2, cout, (frames + ft - 1) // ft, 8 // fs, 12 // fs)
    main = torch.randn(shape, device="cuda", dtype=dtype)
    if channels_last:
        main = main.contiguous(memory_format=torch.channels_last_3d)
    expected = main + shortcut(source)
    actual = down.avg_down3d_add(main, source, ft, fs, shortcut.group_size)
    assert actual is not None
    tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-3 if dtype == torch.float16 else 1e-6
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    assert actual.is_contiguous(memory_format=torch.channels_last_3d if channels_last else torch.contiguous_format)


@torch.no_grad()
@pytest.mark.parametrize("dtype", DTYPES)
def test_average_shortcut_edge_values_and_strides(dtype):
    # Zeros, tiny values, cancellation and temporal front padding. Values stay
    # finite so this also checks that masked temporal loads cannot leak NaNs.
    source = torch.zeros(1, 160, 1, 8, 12, device="cuda", dtype=dtype)
    source[:, ::4] = 1
    source[:, 1::4] = -1
    source[:, 2::4] = torch.finfo(dtype).tiny
    source = source[:, :, :, ::2, ::2]
    shortcut = AvgDown3D(160, 320, 2, 2)
    main = torch.zeros(1, 320, 1, 2, 3, device="cuda", dtype=dtype)
    actual = down.avg_down3d_add(main, source, 2, 2, 4)
    assert actual is not None and torch.isfinite(actual).all()
    torch.testing.assert_close(actual, main + shortcut(source))
    assert down.avg_down3d_add(main, source, 2, 2, 3) is None


@torch.no_grad()
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("channels_last", [False, True])
def test_temporal_downsample_cache_across_fused_and_fallback_calls(monkeypatch, dtype, channels_last):
    module = WanResample(160, "downsample3d").eval().to(device="cuda", dtype=dtype)
    setattr(module, fp.CFG_ATTR, fp.FastPathConfig(channels_last=channels_last))
    if channels_last:
        module.resample[1].to(memory_format=torch.channels_last)
        module.time_conv.to(memory_format=torch.channels_last_3d)
    chunks = [torch.randn(1, 160, t, 12, 20, device="cuda", dtype=dtype) for t in (1, 4, 4, 4)]
    if channels_last:
        chunks = [x.contiguous(memory_format=torch.channels_last_3d) for x in chunks]
    reference_cache, cache = [None], [None]
    original = ef.dm.cat_time_5d
    for i, chunk in enumerate(chunks):
        monkeypatch.setattr(ef.dm, "cat_time_5d", (lambda *a, **k: None) if i == 2 else original)
        expected = WanResample.forward(module, chunk, reference_cache, [0])
        actual = ef.downsample_forward(module, chunk, cache, [0])
        bits_equal(actual, expected)
        bits_equal(cache[0], reference_cache[0])


@torch.no_grad()
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("frames", [1, 5, 9])
def test_lossless_encoder_posterior_and_output_assembly(dtype, frames):
    torch.manual_seed(0)
    ref = AutoencoderKLWan(**CONFIG).eval().to(device="cuda", dtype=dtype)
    fast = AutoencoderKLWan(**CONFIG).eval().to(device="cuda", dtype=dtype)
    fast.load_state_dict(ref.state_dict())
    assert install_wan_vae_encoder_fastpath(fast).installed
    x = torch.rand(1, 3, frames, 64, 96, device="cuda", dtype=dtype) * 2 - 1
    with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
        expected = ref.encode(x).latent_dist
        actual = fast.encode(x).latent_dist
        bits_equal(actual.parameters, expected.parameters)
        bits_equal(actual.logvar, expected.logvar)
        bits_equal(actual.mode(), expected.mode())
        bits_equal(encode_frames(fast, x), expected.parameters)


@torch.no_grad()
@pytest.mark.parametrize(
    "dtype,tf32", [(torch.bfloat16, True), (torch.float16, True), (torch.float32, False), (torch.float32, True)]
)
def test_channels_last_encoder_posterior_and_layout(dtype, tf32):
    torch.manual_seed(1)
    ref = AutoencoderKLWan(**CONFIG).eval().to(device="cuda", dtype=dtype)
    fast = AutoencoderKLWan(**CONFIG).eval().to(device="cuda", dtype=dtype)
    fast.load_state_dict(ref.state_dict())
    assert install_wan_vae_encoder_fastpath(fast, level="channels_last").installed
    # Audit kernels directly; functional convolution calls bypass module hooks.
    source = torch.randn(1, 640, 4, 8, 12, device="cuda", dtype=dtype).contiguous(memory_format=torch.channels_last_3d)
    assert down.spatial_downsample_input(source).is_contiguous(memory_format=torch.channels_last)
    x = torch.rand(1, 3, 9, 64, 96, device="cuda", dtype=dtype) * 2 - 1
    with (
        torch.backends.cudnn.flags(allow_tf32=tf32),
        torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32),
    ):
        expected = ref.encode(x).latent_dist.parameters
        actual = encode_frames(fast, x)
    assert torch.isfinite(actual).all()
    error = (actual.float() - expected.float()).square().mean().sqrt()
    scale = expected.float().square().mean().sqrt().clamp_min(1e-8)
    assert error / scale <= 0.01


@torch.no_grad()
@pytest.mark.parametrize("channels", [160, 320, 640])
@pytest.mark.parametrize("dtype", DTYPES)
def test_production_width_normalization(channels, dtype):
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanRMS_norm

    norm = WanRMS_norm(channels, images=False).to(device="cuda", dtype=dtype)
    x = torch.randn(1, channels, 4, 8, 12, device="cuda", dtype=dtype)
    setattr(norm, fp.CFG_ATTR, fp.FastPathConfig())
    bits_equal(fp.rms_norm_fastpath(norm, x), norm(x))
    setattr(norm, fp.CFG_ATTR, fp.FastPathConfig(channels_last=True))
    cl = x.contiguous(memory_format=torch.channels_last_3d)
    actual = fp.rms_norm_fastpath(norm, cl, silu=True)
    expected = F.silu(norm(cl))
    tol = 0.04 if dtype == torch.bfloat16 else 0.004 if dtype == torch.float16 else 1e-5
    torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
