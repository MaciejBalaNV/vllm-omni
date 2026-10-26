# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in shared-UND serving acceptance; requires a Nano BF16 checkpoint and H200."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.media import generate_synthetic_image
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

pytestmark = [pytest.mark.advanced_model, pytest.mark.diffusion, *hardware_marks(res={"cuda": "H200"})]
SERVER = OmniServerParams(
    model="nvidia/Cosmos3-Nano",
    stage_config_path=get_deploy_config_path("cosmos3_shared_und.yaml"),
    stage_init_timeout=600,
)


@pytest.mark.parametrize("omni_server", [SERVER], indirect=True)
def test_reasoning_stream_and_generated_image_roundtrip(omni_server, online_client):
    online_client.send_omni_request(
        {
            "model": omni_server.model,
            "modalities": ["text"],
            "stream": True,
            "messages": [{"role": "user", "content": "What color is a clear daytime sky?"}],
            "max_tokens": 128,
        }
    )
    responses = online_client.send_images_generations_http_request(
        {
            "json": {
                "model": omni_server.model,
                "prompt": "A blue cube on a white table.",
                "negative_prompt": "",
                "size": "256x256",
                "n": 1,
                "response_format": "b64_json",
                "num_inference_steps": 2,
                "guidance_scale": 7.0,
                "seed": 42,
            }
        }
    )
    image = responses[0].json_body["data"][0]["b64_json"]
    online_client.send_omni_request(
        {
            "model": omni_server.model,
            "modalities": ["text"],
            "stream": True,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe the image."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
                    ],
                }
            ],
            "max_tokens": 128,
        }
    )


@pytest.mark.parametrize("omni_server", [SERVER], indirect=True)
def test_video_and_reasoning_overlap(omni_server, online_client):
    with ThreadPoolExecutor(max_workers=2) as pool:
        video = pool.submit(
            online_client.send_video_diffusion_request,
            {
                "model": omni_server.model,
                "form_data": {
                    "prompt": "A crane flies over a lake.",
                    "negative_prompt": "blurry",
                    "height": 256,
                    "width": 256,
                    "num_frames": 5,
                    "num_inference_steps": 2,
                    "guidance_scale": 6.0,
                    "fps": 24,
                    "seed": 42,
                },
            },
        )
        reasoning = pool.submit(
            online_client.send_omni_request,
            {
                "model": omni_server.model,
                "modalities": ["text"],
                "stream": True,
                "messages": [{"role": "user", "content": "Name two birds."}],
                "max_tokens": 128,
            },
        )
        video.result()
        reasoning.result()


@pytest.mark.parametrize("omni_server", [SERVER], indirect=True)
def test_i2v_without_cfg(omni_server, online_client):
    reference = generate_synthetic_image(384, 256, seed=42)["base64"]
    online_client.send_video_diffusion_request(
        {
            "model": omni_server.model,
            "image_reference": f"data:image/jpeg;base64,{reference}",
            "form_data": {
                "prompt": "The scene moves gently.",
                "height": 256,
                "width": 384,
                "num_frames": 5,
                "num_inference_steps": 2,
                "guidance_scale": 1.0,
                "fps": 15,
                "seed": 43,
            },
        }
    )


@pytest.mark.parametrize("omni_server", [SERVER], indirect=True)
def test_synchronous_video(omni_server, online_client):
    responses = online_client.send_videos_sync_http_request(
        {
            "data": {
                "model": omni_server.model,
                "prompt": "Clouds drift over a mountain.",
                "height": 256,
                "width": 256,
                "num_frames": 5,
                "num_inference_steps": 2,
                "guidance_scale": 1.0,
                "fps": 24,
                "seed": 42,
            },
            "timeout": 300,
        }
    )
    assert responses[0].status_code == 200
