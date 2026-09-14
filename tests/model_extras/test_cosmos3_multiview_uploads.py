# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The manifest contract can be checked without importing the inference runtime."""

import ast
import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("cosmos3_contract", _ROOT / "vllm_omni/model_extras/cosmos3.py")
assert _spec is not None and _spec.loader is not None
contract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(contract)


def manifest(vision=False):
    return {
        "wsm": True,
        "multiview": {
            "views": [
                {
                    "camera_key": camera,
                    "control_reference_index": index,
                    **({"vision_reference_index": index + 11} if vision else {}),
                }
                for index, camera in enumerate(contract.COSMOS3_MADS_CAMERAS)
            ]
        },
    }


@pytest.mark.parametrize("vision", [False, True])
@pytest.mark.parametrize("suffix", [".mp4", ".png"])
def test_resolves_camera_roles_without_mutating_manifest(vision, suffix):
    extra = manifest(vision)
    before = copy.deepcopy(extra)
    paths = [f"upload-{index}{suffix}" for index in range(22 if vision else 11)]
    resolved = contract.resolve_multiview_uploads(extra, paths)
    for index, view in enumerate(resolved["multiview"]["views"]):
        assert view["control_path"] == paths[index]
        if vision:
            assert view["vision_path"] == paths[index + 11]
        assert not any(key.endswith("_reference_index") for key in view)
    assert extra == before


def test_upload_order_is_explicit_and_path_inputs_remain_compatible():
    extra = manifest()
    extra["multiview"]["views"][0] = {"camera_key": contract.COSMOS3_MADS_CAMERAS[0], "control_path": "/owned.mp4"}
    for index, view in enumerate(extra["multiview"]["views"][1:]):
        view["control_reference_index"] = 9 - index
    paths = [f"upload-{index}.mp4" for index in range(10)]
    resolved = contract.resolve_multiview_uploads(extra, paths)
    assert resolved["multiview"]["views"][0]["control_path"] == "/owned.mp4"
    assert [view["control_path"] for view in resolved["multiview"]["views"][1:]] == paths[::-1]
    contract.validate_multiview_request(resolved)


@pytest.mark.parametrize("index", [-1, 11, True, 0.0, "0", None])
def test_rejects_invalid_indexes(index):
    extra = manifest()
    extra["multiview"]["views"][0]["control_reference_index"] = index
    with pytest.raises(ValueError, match="integer index"):
        contract.resolve_multiview_uploads(extra, ["upload.mp4"] * 11)


@pytest.mark.parametrize(
    "case,match",
    [
        ("duplicate", "more than once"),
        ("unused", "exactly once"),
        ("too_many", "at most 22"),
        ("conflict", "cannot be combined"),
        ("missing_control", "control input for every camera"),
        ("partial_vision", "every camera or none"),
        ("camera_order", "camera order"),
        ("duplicate_camera", "unique"),
        ("missing_camera", "camera order"),
        ("mixed_media", "all images or all videos"),
        ("no_wsm", "exactly one"),
        ("top_level_control", "supplied per view"),
        ("unknown_field", "Unsupported"),
    ],
)
def test_rejects_invalid_manifests(case, match):
    extra = manifest()
    paths = ["upload.mp4"] * 11
    views = extra["multiview"]["views"]
    if case == "duplicate":
        views[1]["control_reference_index"] = 0
    elif case == "unused":
        paths.append("unused.mp4")
    elif case == "too_many":
        paths *= 3
    elif case == "conflict":
        views[0]["control"] = "existing.mp4"
    elif case == "missing_control":
        views[-1].pop("control_reference_index")
        paths.pop()
    elif case == "partial_vision":
        views[0]["vision_reference_index"] = len(paths)
        paths.append("vision.mp4")
    elif case == "camera_order":
        views.reverse()
    elif case == "duplicate_camera":
        views[1]["camera_key"] = views[0]["camera_key"]
    elif case == "missing_camera":
        views.pop()
        paths.pop()
    elif case == "mixed_media":
        paths[0] = "image.png"
    elif case == "no_wsm":
        extra.pop("wsm")
    elif case == "top_level_control":
        extra["wsm"] = {"control_path": "control.mp4"}
    elif case == "unknown_field":
        views[0]["typo"] = 1
    with pytest.raises(ValueError, match=match):
        contract.resolve_multiview_uploads(extra, paths)


def test_indexes_without_uploads_and_malformed_views():
    assert contract.has_multiview_upload_indexes(manifest())
    assert not contract.has_multiview_upload_indexes({"multiview": {"views": None}})
    with pytest.raises(ValueError, match="integer index"):
        contract.resolve_multiview_uploads(manifest(), [])
    with pytest.raises(ValueError, match="at least one"):
        contract.resolve_multiview_uploads({"multiview": {"views": {}}}, [])


def test_pipeline_validation_accepts_in_memory_media_with_its_classifier():
    extra = {"wsm": {}, "multiview": {"views": [{"camera_key": "front", "control": object()}]}}
    _, views = contract.validate_multiview_request(extra, ("front",), media_kind=lambda _: "video")
    assert len(views) == 1


@pytest.mark.parametrize("envelope", [False, True])
def test_client_converts_local_manifest_to_upload_indexes(tmp_path, envelope):
    pytest.importorskip("httpx")
    spec = importlib.util.spec_from_file_location(
        "multiview_client", _ROOT / "examples/online_serving/multiview_video/cosmos3_multiview_client.py"
    )
    assert spec is not None and spec.loader is not None
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    extra = manifest(True)
    for index, view in enumerate(extra["multiview"]["views"]):
        for role in ("control", "vision"):
            view.pop(f"{role}_reference_index")
            filename = f"{role}-{index}.mp4"
            (tmp_path / filename).write_bytes(b"media")
            view[f"{role}_path"] = filename
    request = {"prompt": "drive", "fps": 30, "num_steps": 4, **({"extra_params": extra} if envelope else extra)}
    before = copy.deepcopy(request)
    data, paths = client.prepare_request(request, tmp_path)
    resolved = contract.resolve_multiview_uploads(json.loads(data["extra_params"]), [str(path) for path in paths])
    assert len(paths) == 22
    assert resolved["multiview"]["views"][0]["vision_path"] == str(tmp_path / "vision-0.mp4")
    assert data["num_inference_steps"] == "4"
    assert data["fps"] == "30"
    assert request == before


@pytest.mark.parametrize("mode", ["async", "sync", "failed"])
@pytest.mark.parametrize("resolution", [None, "480", "720"])
@pytest.mark.parametrize("aspect_ratio", [None, "auto", "3:4", "9:16"])
def test_client_uploads_and_downloads_or_reports_failure(tmp_path, monkeypatch, mode, resolution, aspect_ratio):
    httpx = pytest.importorskip("httpx")
    spec = importlib.util.spec_from_file_location(
        "multiview_client", _ROOT / "examples/online_serving/multiview_video/cosmos3_multiview_client.py"
    )
    assert spec is not None and spec.loader is not None
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    extra = manifest()
    for index, view in enumerate(extra["multiview"]["views"]):
        view.pop("control_reference_index")
        filename = f"control-{index}.mp4"
        (tmp_path / filename).write_bytes(f"media-{index}".encode())
        view["control_path"] = filename
    manifest_path = tmp_path / "manifest.json"
    request_manifest = {"prompt": "drive", **extra}
    if aspect_ratio is not None:
        request_manifest["aspect_ratio"] = "1:1"
        request_manifest["multiview"]["aspect_ratio"] = "16:9"
    if resolution is not None:
        # The CLI must replace conflicting fields and stale dimensions together.
        request_manifest.update(resolution="480", width=832, height=480)
        request_manifest["multiview"]["resolution"] = "720"
    manifest_path.write_text(json.dumps(request_manifest))
    output = tmp_path / "output.mp4"
    requests = []

    def respond(request):
        requests.append(request.url.path)
        if request.method == "POST":
            body = request.read()
            assert body.count(b'name="input_references"') == 11
            assert b"control_reference_index" in body
            assert b"control_path" not in body
            expected_resolution = resolution or "480"
            assert f'"resolution": "{expected_resolution}"'.encode() in body
            expected_ratio = (aspect_ratio or "auto").replace(":", ",")
            assert f'"aspect_ratio": "{expected_ratio}"'.encode() in body
            if expected_ratio == "auto":
                assert b'name="width"' not in body
                assert b'name="height"' not in body
            else:
                width, height = client.SUPPORTED_RESOLUTIONS[expected_resolution][expected_ratio]
                assert f'name="width"\r\n\r\n{width}\r\n'.encode() in body
                assert f'name="height"\r\n\r\n{height}\r\n'.encode() in body
            for index in range(11):
                assert f"media-{index}".encode() in body
            if mode == "sync":
                return httpx.Response(200, content=b"generated-video")
            return httpx.Response(200, json={"id": "video-test", "status": "queued"})
        if request.url.path.endswith("/content"):
            return httpx.Response(200, content=b"generated-video")
        if mode == "failed":
            return httpx.Response(500, json={"status": "failed", "error": {"message": "corrupt input"}})
        return httpx.Response(200, json={"status": "completed"})

    real_client = httpx.Client
    monkeypatch.setattr(
        client.httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(respond), **kwargs)
    )
    monkeypatch.setattr(client.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["client", str(manifest_path), "--output", str(output)]
        + (["--sync"] if mode == "sync" else [])
        + (["--resolution", resolution] if resolution is not None else [])
        + (["--aspect-ratio", aspect_ratio] if aspect_ratio is not None else []),
    )
    if mode == "failed":
        with pytest.raises(RuntimeError, match="corrupt input"):
            client.main()
        assert not output.exists()
    else:
        client.main()
        assert output.read_bytes() == b"generated-video"
        assert requests[-1] == ("/v1/videos/sync" if mode == "sync" else "/v1/videos/video-test/content")


@pytest.fixture
def multiview_client():
    pytest.importorskip("httpx")
    spec = importlib.util.spec_from_file_location(
        "multiview_client", _ROOT / "examples/online_serving/multiview_video/cosmos3_multiview_client.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("envelope", [False, True])
@pytest.mark.parametrize("location", ["top", "extra", "nested", "all", "omitted"])
def test_client_preserves_resolution_and_manifest(tmp_path, multiview_client, envelope, location):
    (tmp_path / "control.mp4").write_bytes(b"media")
    extra = {"multiview": {"views": [{"camera_key": "front", "control_path": "control.mp4"}]}, "wsm": {}}
    request = {"prompt": "drive", **({"extra_params": extra} if envelope else extra)}
    if location in ("top", "all"):
        request["resolution"] = 720
    if location in ("extra", "all"):
        request.setdefault("extra_params", {})["resolution"] = "720"
    if location in ("nested", "all"):
        extra["multiview"]["resolution"] = 720
    before = copy.deepcopy(request)
    data, paths = multiview_client.prepare_request(request, tmp_path)
    resolved = json.loads(data["extra_params"])
    expected = "480" if location == "omitted" else "720"
    assert resolved["resolution"] == resolved["multiview"]["resolution"] == expected
    assert "width" not in data and "height" not in data
    assert resolved["multiview"]["aspect_ratio"] == "auto"
    assert paths == [tmp_path / "control.mp4"]
    assert request == before


@pytest.mark.parametrize("resolution", [256, 704, "1080", "720p", True])
def test_client_rejects_unsupported_resolution(tmp_path, multiview_client, resolution):
    request = {"multiview": {"views": [], "resolution": resolution}}
    with pytest.raises(ValueError, match="Unsupported Cosmos3 multiview resolution"):
        multiview_client.prepare_request(request, tmp_path)


@pytest.mark.parametrize("location", ["top", "extra"])
def test_client_rejects_resolution_conflicts(tmp_path, multiview_client, location):
    request = {"multiview": {"views": [], "resolution": "720"}}
    target = request if location == "top" else request.setdefault("extra_params", {})
    target["resolution"] = "480"
    with pytest.raises(ValueError, match="Conflicting Cosmos3 multiview resolutions"):
        multiview_client.prepare_request(request, tmp_path)


@pytest.mark.parametrize("dimensions", [{"width": 832}, {"height": 480}])
def test_client_rejects_dimensions_that_disagree_with_resolution(tmp_path, multiview_client, dimensions):
    request = {"multiview": {"views": [], "resolution": "720", "aspect_ratio": "16:9"}, **dimensions}
    with pytest.raises(ValueError, match="resolution='720' requires"):
        multiview_client.prepare_request(request, tmp_path)


def test_client_buckets_match_runtime_without_importing_inference(multiview_client):
    tree = ast.parse((_ROOT / "vllm_omni/diffusion/models/cosmos3/utils.py").read_text())
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "VIDEO_RES_SIZE_INFO"
    )
    canonical = ast.literal_eval(assignment.value)
    assert multiview_client.SUPPORTED_RESOLUTIONS == {key: canonical[key] for key in ("480", "720")}
    assert set(contract.COSMOS3_MULTIVIEW_ASPECT_RATIOS) == set(canonical["480"]) == set(canonical["720"])


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, "auto"),
        ("auto", "auto"),
        ("1:1", "1,1"),
        ("4,3", "4,3"),
        (" 6 : 8 ", "3,4"),
        ("32:18", "16,9"),
        ("9:16", "9,16"),
    ],
)
def test_ratio_normalization_matches_client(multiview_client, value, expected):
    assert contract.normalize_multiview_aspect_ratio(value) == expected
    assert multiview_client.normalize_aspect_ratio(value) == expected


@pytest.mark.parametrize("value", ["21:9", "0:1", "-1:1", "4:0", "", "square", True, 1.5, "1.5:1", [4, 3]])
def test_invalid_ratios_rejected_by_contract_and_client(multiview_client, value):
    for normalize in (contract.normalize_multiview_aspect_ratio, multiview_client.normalize_aspect_ratio):
        with pytest.raises(ValueError, match="aspect_ratio"):
            normalize(value)
    extra = manifest()
    extra["multiview"]["aspect_ratio"] = value
    with pytest.raises(ValueError, match="aspect_ratio"):
        contract.resolve_multiview_uploads(extra, [f"{index}.mp4" for index in range(11)])


@pytest.mark.parametrize("resolution", ["480", "720"])
@pytest.mark.parametrize("ratio", ["1:1", "4:3", "3:4", "16:9", "9:16"])
@pytest.mark.parametrize("location", ["top", "extra", "nested"])
def test_client_explicit_aspect_ratios(tmp_path, multiview_client, resolution, ratio, location):
    extra = {"resolution": resolution, "multiview": {"views": []}}
    request = {"extra_params": extra}
    target = {"top": request, "extra": extra, "nested": extra["multiview"]}[location]
    target["aspect_ratio"] = ratio
    before = copy.deepcopy(request)
    data, _ = multiview_client.prepare_request(request, tmp_path)
    normalized = ratio.replace(":", ",")
    assert (int(data["width"]), int(data["height"])) == multiview_client.SUPPORTED_RESOLUTIONS[resolution][normalized]
    assert json.loads(data["extra_params"])["multiview"]["aspect_ratio"] == normalized
    assert request == before


def test_client_ratio_conflicts_and_independent_overrides(tmp_path, multiview_client):
    request = {
        "aspect_ratio": "1:1",
        "multiview": {"views": [], "aspect_ratio": "9:16", "resolution": "720"},
        "width": 832,
        "height": 480,
    }
    with pytest.raises(ValueError, match="Conflicting.*aspect ratios"):
        multiview_client.prepare_request(request, tmp_path)
    # A resolution override cannot hide an unrelated aspect-ratio conflict.
    with pytest.raises(ValueError, match="Conflicting.*aspect ratios"):
        multiview_client.prepare_request(request, tmp_path, resolution_override="480")
    explicit, _ = multiview_client.prepare_request(request, tmp_path, aspect_ratio_override="3:4")
    assert (explicit["width"], explicit["height"]) == ("832", "1104")
    automatic, _ = multiview_client.prepare_request(request, tmp_path, aspect_ratio_override="auto")
    assert "width" not in automatic and "height" not in automatic
    assert json.loads(automatic["extra_params"])["resolution"] == "720"
    request["aspect_ratio"] = "18:32"
    resolved, _ = multiview_client.prepare_request(request, tmp_path, resolution_override="480")
    assert (resolved["width"], resolved["height"]) == ("480", "832")


@pytest.mark.parametrize("dimensions", [{"width": 640}, {"height": 640}, {"width": 640, "height": 640}])
def test_client_auto_preserves_only_explicit_dimension_constraints(tmp_path, multiview_client, dimensions):
    request = {"multiview": {"views": []}, **dimensions}
    data, _ = multiview_client.prepare_request(request, tmp_path)
    assert {key: int(data[key]) for key in ("width", "height") if key in data} == dimensions
    overridden, _ = multiview_client.prepare_request(request, tmp_path, resolution_override="720")
    assert "width" not in overridden and "height" not in overridden
