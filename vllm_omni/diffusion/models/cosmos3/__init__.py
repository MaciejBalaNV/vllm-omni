# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Lazy exports keep CPU conditioning preparation independent of GPU kernels."""

from importlib import import_module

_EXPORTS = {
    "Cosmos3OmniDiffusersPipeline": ".pipeline_cosmos3",
    "get_cosmos3_post_process_func": ".pipeline_cosmos3",
    "get_cosmos3_pre_process_func": ".pipeline_cosmos3",
    "Cosmos3VFMTransformer": ".transformer_cosmos3",
    "Cosmos3EdgeVFMTransformer": ".transformer_cosmos3_edge",
}
__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
