# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Serving layer for the legacy OpenPI and root RoboLab websocket routes.

Flow: raw obs → engine request → actions.
The loaded policy model owns dataset transforms inside its pipeline.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import count
from typing import Any

import numpy as np
from omegaconf import OmegaConf
from vllm.logger import init_logger

logger = init_logger(__name__)

ActionOutput = np.ndarray | dict[str, np.ndarray]


def _to_builtin_container(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, Mapping):
        return {key: _to_builtin_container(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_to_builtin_container(item) for item in value]
    return value


def _config_value(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


@dataclass(frozen=True)
class PolicyServerConfig:
    """OpenPI policy server handshake config.

    Values are model-specific and must be provided by the loaded policy model.
    """

    values: dict[str, Any]

    @classmethod
    def from_model_config(cls, model_config: Any) -> PolicyServerConfig:
        if isinstance(model_config, Mapping):
            raw_config = model_config.get("policy_server_config")
        else:
            raw_config = getattr(model_config, "policy_server_config", None)

        if raw_config is None:
            raise ValueError("Robot OpenPI serving requires policy_server_config.")
        if isinstance(raw_config, cls):
            return raw_config
        if not isinstance(raw_config, Mapping):
            raise ValueError("Robot OpenPI serving requires policy_server_config.")
        return cls(_to_builtin_container(raw_config))

    def to_dict(self) -> dict[str, Any]:
        return _to_builtin_container(self.values)


class ServingRealtimeRobotOpenPI:
    """Robot policy serving layer for OpenPI protocol.

    Model-specific transform/state lives in the diffusion pipeline.
    """

    def __init__(
        self,
        engine_client: Any,
        model_name: str | None = None,
        *,
        policy_server_config: PolicyServerConfig | None = None,
    ) -> None:
        self.engine_client = engine_client
        self.model_name = model_name
        model_config = self._get_model_config(engine_client)
        self.policy_server_config = policy_server_config or PolicyServerConfig.from_model_config(model_config)
        self._request_counter = count()

    @classmethod
    def create_policy_server(
        cls,
        engine_client: Any,
        model_name: str | None = None,
    ) -> ServingRealtimeRobotOpenPI | None:
        try:
            return cls(engine_client=engine_client, model_name=model_name)
        except ValueError as exc:
            if "policy_server_config" not in str(exc):
                raise
            logger.info("Robot OpenPI serving disabled for model %s", model_name)
            return None

    @classmethod
    def create_robolab_policy_server(
        cls,
        engine_client: Any,
        model_name: str | None = None,
    ) -> ServingRealtimeRobotOpenPI | None:
        """Create the root-route adapter without requiring handshake metadata."""
        if not cls._supports_cosmos3_action_policy(engine_client, model_name):
            logger.info("RoboLab compatibility serving disabled for model %s", model_name)
            return None

        # Validate source-only dependencies before accepting clients, but keep
        # the standard vLLM endpoints available when the optional integration is
        # not installed or is version-skewed.
        try:
            from vllm_omni.diffusion.models.cosmos3.utils import preflight_robolab_framework_imports

            preflight_robolab_framework_imports()
        except (ImportError, AttributeError) as exc:
            logger.warning(
                "RoboLab compatibility serving disabled for model %s: %s",
                model_name,
                exc,
            )
            return None
        return cls(
            engine_client=engine_client,
            model_name=model_name,
            policy_server_config=PolicyServerConfig({}),
        )

    @staticmethod
    def _get_model_config(engine_client: Any) -> Any:
        model_config = None
        get_od_config = getattr(engine_client, "get_diffusion_od_config", None)
        if callable(get_od_config):
            od_config = get_od_config()
            model_config = getattr(od_config, "model_config", None)

        if model_config is None:
            for stage_config in getattr(engine_client, "stage_configs", []) or []:
                if getattr(stage_config, "stage_type", None) != "diffusion":
                    continue
                engine_args = getattr(stage_config, "engine_args", None)
                model_config = getattr(engine_args, "model_config", None)
                if model_config is not None:
                    break

        if model_config is None:
            od_config = getattr(engine_client, "od_config", None)
            model_config = getattr(od_config, "model_config", None)

        if model_config is None:
            model_config = getattr(engine_client, "model_config", None)
        return model_config

    @classmethod
    def _supports_cosmos3_action_policy(cls, engine_client: Any, model_name: str | None) -> bool:
        from vllm_omni.model_extras.cosmos3 import (
            COSMOS3_PIPELINE_NAMES,
            resolve_cosmos3_action_gen,
        )

        od_config = None
        get_od_config = getattr(engine_client, "get_diffusion_od_config", None)
        if callable(get_od_config):
            od_config = get_od_config()
        pipeline_names = {_config_value(od_config, "model_class_name")}
        for stage_config in getattr(engine_client, "stage_configs", []) or []:
            if getattr(stage_config, "stage_type", None) != "diffusion":
                continue
            engine_args = getattr(stage_config, "engine_args", None)
            pipeline_names.add(_config_value(engine_args, "model_class_name"))
        if pipeline_names.isdisjoint(COSMOS3_PIPELINE_NAMES):
            return False

        config_candidates = [
            od_config,
            _config_value(od_config, "tf_model_config"),
            _config_value(getattr(engine_client, "od_config", None), "tf_model_config"),
        ]
        for stage_config in getattr(engine_client, "stage_configs", []) or []:
            if getattr(stage_config, "stage_type", None) == "diffusion":
                config_candidates.append(_config_value(getattr(stage_config, "engine_args", None), "tf_model_config"))

        for config in config_candidates:
            action_gen = _config_value(config, "action_gen")
            if action_gen is not None:
                return _as_bool(action_gen)

        canonical_model_name = str(getattr(engine_client, "model", "") or "")
        revision = None
        for stage_config in getattr(engine_client, "stage_configs", []) or []:
            if getattr(stage_config, "stage_type", None) != "diffusion":
                continue
            engine_args = getattr(stage_config, "engine_args", None)
            revision = _config_value(engine_args, "revision")
            if revision is not None:
                break

        resolved_action_gen = resolve_cosmos3_action_gen(canonical_model_name, revision=revision)
        if resolved_action_gen is not None:
            return resolved_action_gen

        # Keep released checkpoints usable when metadata is temporarily
        # unavailable (for example an offline API process with a warm worker).
        # Check the canonical model independently of --served-model-name aliases.
        from vllm_omni.model_extras.cosmos3 import is_cosmos3_droid_policy_checkpoint

        return is_cosmos3_droid_policy_checkpoint(canonical_model_name) or (
            not canonical_model_name and is_cosmos3_droid_policy_checkpoint(model_name)
        )

    @classmethod
    def _get_policy_server_config(cls, engine_client: Any) -> PolicyServerConfig:
        return PolicyServerConfig.from_model_config(cls._get_model_config(engine_client))

    def reset(self, obs: dict) -> None:
        """Compatibility hook; per-connection state lives in RobotRealtimeConnection."""

    async def infer(self, obs: dict, *, session_id: str, reset: bool) -> ActionOutput:
        """raw obs → engine → actions."""
        # Build request, run inference through AsyncOmni
        request = self._build_request(obs, session_id=session_id, reset=reset)
        result = None
        # OpenPI policy serving is one request -> one action reply. AsyncOmni
        # exposes an async iterator, so consume it to completion and use the
        # final output, matching other non-streaming OpenAI serving paths.
        async for output in self.engine_client.generate(
            prompt=request.prompt,
            request_id=request.request_id,
            sampling_params_list=[request.sampling_params],
        ):
            result = output
        if result is None:
            raise RuntimeError("Robot OpenPI request produced no output.")

        return self._extract_actions(result)

    def _next_request_id(self, session_id: str) -> str:
        return f"robot-{session_id}-{next(self._request_counter)}"

    def _build_request(self, obs: dict, *, session_id: str, reset: bool) -> Any:
        """Build engine request from raw robot obs.

        Returns an `OmniDiffusionRequest` payload consumed by
        `AsyncOmni.generate()` and routed to the diffusion stage.
        """
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.entrypoints.openai.stage_params import (
            clone_sampling_params,
            get_default_sampling_params_list,
        )
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams
        from vllm_omni.model_extras.cosmos3 import is_cosmos3_droid_policy_checkpoint

        sampling_params = OmniDiffusionSamplingParams()
        for default_params in get_default_sampling_params_list(self.engine_client):
            if isinstance(default_params, OmniDiffusionSamplingParams):
                sampling_params = clone_sampling_params(default_params)
                break

        # ``sampling_params`` is cloned, and copy the nested request namespace as
        # an additional guard against a custom clone implementation sharing it.
        extra_args = copy.deepcopy(sampling_params.extra_args or {})
        canonical_model_name = str(getattr(self.engine_client, "model", "") or "")
        droid_model_name = canonical_model_name or self.model_name
        if is_cosmos3_droid_policy_checkpoint(droid_model_name):
            # This is a property of the released DROID checkpoints, not a global
            # Cosmos3 default. An explicit stage default still takes precedence.
            extra_args.setdefault("format_prompt_as_json", True)
        extra_args.update(
            {
                "reset": reset,
                "session_id": session_id,
                "robot_obs": obs,
            }
        )

        prompt = obs.get("prompt", "")
        sampling_params.extra_args = extra_args
        return OmniDiffusionRequest(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=self._next_request_id(session_id),
        )

    def _extract_actions(self, result: Any) -> ActionOutput:
        """Extract actions from engine result."""
        multimodal_output = getattr(result, "multimodal_output", None)
        if not isinstance(multimodal_output, Mapping):
            raise RuntimeError("Missing multimodal_output in robot policy result")

        actions = multimodal_output.get("actions")
        if actions is None:
            raise RuntimeError("Missing multimodal_output['actions'] in robot policy result")
        if isinstance(actions, Mapping):
            return {str(key): np.asarray(value, dtype=np.float32) for key, value in actions.items()}
        return np.asarray(actions, dtype=np.float32)
