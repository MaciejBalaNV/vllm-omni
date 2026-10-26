# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Omni adapter preserving native Cosmos3 multimodal processing and AR execution."""

import torch
from vllm.config import VllmConfig
from vllm.model_executor.models.cosmos3 import Cosmos3ForConditionalGeneration, Cosmos3ProcessingInfo
from vllm.model_executor.models.qwen3_vl import Qwen3VLDummyInputsBuilder, Qwen3VLMultiModalProcessor
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm_omni.diffusion.models.cosmos3.conditioning import checkpoint_identity


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Cosmos3ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class OmniCosmos3ForConditionalGeneration(Cosmos3ForConditionalGeneration):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        parallel = vllm_config.parallel_config
        if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
            raise ValueError("Shared Cosmos3 UND supports TP=1 and PP=1")
        if vllm_config.model_config.dtype != torch.bfloat16 or vllm_config.model_config.quantization is not None:
            raise ValueError("Shared Cosmos3 UND requires an unquantized BF16 checkpoint")
        self._cosmos3_checkpoint = checkpoint_identity(
            vllm_config.model_config.model, vllm_config.model_config.revision
        )
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def get_kv_transfer_metadata(self, req_id: str, num_computed_tokens: int | None = None) -> dict:
        # Native v1 bind_kv_cache orders runner tensors by decoder layer index.
        # This TP=1 model has exactly one cache-bearing attention per layer.
        return {
            "layer_order": list(range(self.config.text_config.num_hidden_layers)),
            "cosmos3_checkpoint": self._cosmos3_checkpoint,
        }
