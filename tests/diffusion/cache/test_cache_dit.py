# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Model specific tests for CacheDiT enablement.
"""

from unittest.mock import Mock, patch

import pytest

import vllm_omni.diffusion.cache.cache_dit_backend as cd_backend
from vllm_omni.diffusion.data import DiffusionCacheConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SEPARATE_CFG_ENABLERS = [
    cd_backend.enable_cache_for_ltx2,
    cd_backend.enable_cache_for_helios,
    cd_backend.enable_cache_for_wan22,
    cd_backend.enable_cache_for_longcat_image,
    cd_backend.enable_cache_for_cosmos3,
]

SAMPLE_CACHE_CONFIG = DiffusionCacheConfig()


@pytest.mark.parametrize("enabler", SEPARATE_CFG_ENABLERS)
@patch("vllm_omni.diffusion.cache.cache_dit_backend.BlockAdapter")
@patch("vllm_omni.diffusion.cache.cache_dit_backend.cache_dit")
def test_separate_cfg(mock_cache_dit, mock_block_adapter, enabler):
    """Ensure that custom enablers for models with separate CFG pass
    the param through to cache_dit correctly.

    Regression test for: https://github.com/vllm-project/vllm-omni/pull/2860
    """
    mock_pipeline = Mock()
    enabler(mock_pipeline, SAMPLE_CACHE_CONFIG)

    mock_cache_dit.enable_cache.assert_called_once()
    adapter_kwargs = mock_block_adapter.call_args.kwargs
    assert adapter_kwargs["has_separate_cfg"] is True


@patch("vllm_omni.diffusion.cache.cache_dit_backend.BlockAdapter")
@patch("vllm_omni.diffusion.cache.cache_dit_backend.cache_dit")
def test_cosmos3_cache_dit_wraps_gen_layers(mock_cache_dit, mock_block_adapter):
    """Cosmos3 should cache only the repeated GEN pathway blocks."""
    mock_pipeline = Mock()
    gen_layers = object()
    mock_pipeline.transformer.gen_layers = gen_layers

    cd_backend.enable_cache_for_cosmos3(mock_pipeline, SAMPLE_CACHE_CONFIG)

    mock_cache_dit.enable_cache.assert_called_once()
    adapter_kwargs = mock_block_adapter.call_args.kwargs
    assert adapter_kwargs["transformer"] is mock_pipeline.transformer
    assert adapter_kwargs["blocks"] == [gen_layers]
    assert adapter_kwargs["has_separate_cfg"] is True
    assert adapter_kwargs["check_forward_pattern"] is False
