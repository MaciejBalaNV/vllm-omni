# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Read multi-output stage capabilities while preserving scalar declarations."""

from collections.abc import Mapping


def stage_output_types(stage: object) -> tuple[str, ...]:
    if isinstance(stage, Mapping):
        multiple = stage.get("final_output_types")
        scalar = stage.get("final_output_type")
    else:
        multiple = getattr(stage, "final_output_types", ())
        scalar = getattr(stage, "final_output_type", None)
    return tuple(dict.fromkeys([*(multiple or ()), *([scalar] if scalar else [])]))
