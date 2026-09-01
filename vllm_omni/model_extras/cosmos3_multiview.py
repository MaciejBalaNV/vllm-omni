# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

COSMOS3_MULTIVIEW_EXTRA_BODY_PARAMS = frozenset(
    {
        "multiview",
        "wsm",
        "attention_scope",
        "flow_shift",
        "max_sequence_length",
        "negative_prompt",
        "resolution",
        "fps",
        "frame_rate",
        "resolved_frame_rate",
    }
)
