# Cosmos3 Phase 2.2 maskless attention

The experiment
`cosmos3_nano_mv_transfer_maskless_decomp_attn_cam_lidar_480p720p_perviewcap_phase2p2_192n`
requires schema version 2 with `multiview.backend="maskless"`. Re-export older
artifacts: changing the JSON alone does not establish that the weights were
trained with these semantics. HF → Diffusers conversion checks the attention
metadata against the embedded source model configuration.

Maskless runs noncausal same-sensor/view, target-only same-instant, and caption
passes. Overlapping sensor keys intentionally contribute twice. Controls only
participate in the view and caption passes. Cameras read their own caption (or
the shared caption); LiDAR reads all captions unless `lidar_attends_captions=false`.
Missing caption-access metadata defaults to true for older sparse artifacts.
`same_view` and single sensor/view layouts omit the instant pass. One camera plus
LiDAR has two groups. A null temporal window and `control_attends_sensor=true`
are required. Triton ↔ FA4 overrides remain supported for sparse checkpoints;
`VLLM_OMNI_COSMOS3_MULTIVIEW_BACKEND` cannot cross the sparse/maskless boundary.

Caption K/V remain compact, with an admission ceiling of 4,098 tokens per
caption. Every CFG branch retains batch one. Plans and merge workspace are
model-local and reset with request caches. TP, Ulysses, CFG parallelism and HSDP
retain the existing topology constraints (including no simultaneous TP/HSDP).
FA is selected once per worker according to hardware and retained during serving.
Worker logs identify GPU, FA, NATTEN, PyTorch and CUDA versions. Cross-GPU bitwise
reproducibility is not guaranteed.

## Install

Use the existing supported vLLM PyTorch/CUDA environment. The optional group
`cosmos3-maskless` pins `natten==0.21.6`, matching the reference training image.
Install a NATTEN 0.21.6 wheel built for that exact PyTorch/CUDA environment, or
build from source with its CUDA toolkit and build dependencies installed:

```bash
# Matching wheel downloaded from the NATTEN distribution for this environment:
python -m pip install --no-deps /wheels/natten-0.21.6-*.whl
# Alternatively, compile against the currently installed PyTorch/CUDA:
python -m pip install --no-deps --no-build-isolation --no-binary=natten natten==0.21.6
python -m pip check
python -c 'import torch, natten; from natten.functional import merge_attentions; print(torch.__version__, torch.version.cuda, natten.__version__)'
```

Do not downgrade PyTorch to fit a wheel. Installing `.[cosmos3-maskless]` also
expresses the dependency, but check the resolver's proposed environment changes
before applying them. Loading a maskless transformer fails immediately if
NATTEN is missing, has the wrong version, or its merge API/extension cannot load.
Sparse loading does not import NATTEN.

## Convert and reproduce

Select a **completed Phase 2.2 checkpoint** and its own resolved config. The
checkpoint and reference media are external inputs; none are bundled with this
update. These placeholders must point to that completed run, not an earlier
sparse checkpoint or the Phase 2.2 resume source.

```bash
export COSMOS3_MASKLESS_DCP=/models/phase2p2/completed_iteration
export COSMOS3_MASKLESS_CONFIG=/models/phase2p2/config.yaml
export COSMOS3_MASKLESS_HF=/models/phase2p2-hf
export COSMOS3_MASKLESS_DIFFUSERS=/models/phase2p2-diffusers

# Installed imaginaire4/Cosmos3 environment, with access to V1.2 tokenizer artifacts:
python -m cosmos3.scripts.export_model \
  --checkpoint-path "$COSMOS3_MASKLESS_DCP" --config-file "$COSMOS3_MASKLESS_CONFIG" \
  --backbone-type cosmos3_multiview -o "$COSMOS3_MASKLESS_HF"
python -m cosmos3.scripts.convert_model_to_diffusers \
  --checkpoint-path "$COSMOS3_MASKLESS_HF" -o "$COSMOS3_MASKLESS_DIFFUSERS"
```

Inspect both exported configs for `backend=maskless`, `attention_scope=decomposed`,
null temporal window, control access, and the source's LiDAR caption flag.
Weight mappings, projections, V1.2 tokenizer artifacts, preprocessing and request
formats are unchanged. Use the existing [preparation workflow](cosmos3_multiview_lidar_validation.md#prepare-and-run-the-supplied-five-records)
with the Phase 2.2 media and the same rig/order on both runtimes. Produce separate
camera-only/joint T2V/I2V manifests for seven/eleven cameras at 480p and six at
720p. Include `lidar.return_output=true` in one joint manifest to check numeric
LiDAR output.

```bash
# Reference, in imaginaire4:
python -m cosmos3.scripts.inference \
  --checkpoint-path "$COSMOS3_MASKLESS_DCP" --config-file "$COSMOS3_MASKLESS_CONFIG" \
  -i /data/phase2p2/reference.jsonl -o /data/results/reference --load-caption-from-data

# In vllm-omni: run once eager, then without --enforce-eager for regional compilation.
python examples/offline_inference/multiview_video/cosmos3_multiview.py \
  --model "$COSMOS3_MASKLESS_DIFFUSERS" --input /data/phase2p2/requests.jsonl \
  --output-dir /data/results/omni-eager --enforce-eager
python examples/offline_inference/multiview_video/cosmos3_multiview.py \
  --model "$COSMOS3_MASKLESS_DIFFUSERS" --input /data/phase2p2/requests.jsonl \
  --output-dir /data/results/omni-compiled

vllm serve "$COSMOS3_MASKLESS_DIFFUSERS" --omni \
  --model-class-name Cosmos3MultiviewPipeline --enforce-eager --port 8091
python examples/online_serving/multiview_video/cosmos3_multiview_client.py \
  /data/phase2p2/request_0000.json --server http://localhost:8091 --output /data/results/http-async.mp4
python examples/online_serving/multiview_video/cosmos3_multiview_client.py \
  /data/phase2p2/request_0000.json --server http://localhost:8091 --sync --output /data/results/http-sync.mp4
```

Keep seeds, captions, frame count, FPS, controls, conditioning, scheduler, steps
and guidance identical. Record cold/warm latency and peak allocated/reserved GPU
memory. Review conditioning, control adherence, temporal stability, camera
ordering and cross-camera consistency. Attention numerical correctness and
functional/visual generation are acceptance gates; full-generation latent parity
and performance thresholds are not.

## Tests and qualification status (2026-09-18)

Implementation base revisions (plus this working-tree update):

| Repository | Base SHA |
|---|---|
| imaginaire4 | `7252403e648c0c838edeb97182e74941de4fa8d1` |
| vllm-omni | `4ddeb4dc22cded115527a529f89229d646868874` |

Before production qualification, record the final committed SHAs, completed
checkpoint URI/iteration/checksum, resolved config, reference media checksums,
GPU/driver, selected FA, NATTEN, PyTorch, CUDA and vLLM versions with the results.
A completed Phase 2.2 checkpoint has **not** been pinned or validated here.

Local checks use macOS CPU. Runtime tests use a temporary bootstrap for unavailable
vLLM package initialization; PyTorch tensor operations and the repository's
attention/planning code run unchanged. CPU FlashAttention/NATTEN stand-ins are
independent mathematical oracles, not validation of CUDA kernels. The compact
attention boundary is tested under both Dynamo's eager backend and CPU Inductor.

| Check | Status |
|---|---|
| Export metadata, stale metadata, invalid semantics, sparse compatibility | 49 tests passed |
| Maskless float64 oracle, duplicates, mixed rates/geometries, controls, GQA, captions, indexing, batch admission, chunk tails, dynamic prompt compilation | 97 passed; 7 GPU/reference tests skipped |
| Packed transformer, cache reuse/reset, control-CFG removal, backend overrides | 48 selected CPU adapter tests passed |
| Actual Phase 2.2 experiment composition | Test added; local collection blocked by missing Hydra/reference environment |
| Real FA/BF16 attention, standalone NATTEN merge, internal merge compilation, reference wrapper comparison | Tests added; pending CUDA/reference environment |
| Eager/regional GEN with TP/Ulysses/CFG/HSDP, repeated requests and >8 prompt lengths | Distributed tests extended; pending GPUs |
| Completed checkpoint conversion, offline/HTTP generation, optional LiDAR outputs | Pending checkpoint, media and GPUs |
| 7/11-camera 480p and 6-camera 720p camera-only/joint T2V/I2V visual review, latency/memory | Pending |

Run in the installed environments:

```bash
# imaginaire4
pytest packages/cosmos3/cosmos3/scripts/multiview_export_test.py
pytest projects/cosmos3/cosmos3/configs/base/experiment/multiview/av/av_configs_test.py \
  -k phase2p2_deployment_attention_metadata

# vllm-omni
pytest tests/diffusion/models/cosmos3/test_multiview_maskless_attention.py \
  tests/diffusion/attention/test_fa_varlen.py
pytest tests/diffusion/models/cosmos3/test_multiview_flex_attention.py \
  tests/diffusion/models/cosmos3/test_multiview_parallel.py \
  tests/diffusion/models/cosmos3/test_cosmos3_lidar.py \
  tests/diffusion/models/cosmos3/test_cosmos3_multiview_pipeline.py
pytest tests/diffusion/distributed/test_cosmos3_multiview_parallel.py -k maskless
```

The reference-wrapper test additionally needs `imaginaire.attention` importable in
the NATTEN/CUDA test environment. GPU qualification remains pending until the
completed Phase 2.2 checkpoint and reference media have been exercised.

Local execution details: Python 3.12.11; attention bootstrap PyTorch 2.12.0,
transformer/pipeline adapter PyTorch 2.14.0; CUDA unavailable. Alongside the 97
maskless checks, 106 existing sparse/FA4-metadata/topology checks passed, and two
real Gloo subgroup collective checks passed outside the sandbox. The wider
attention run skipped 41 additional CUDA-only sparse tests. Ruff lint/format
and `git diff --check` passed for both repositories. These results do not
establish NATTEN kernel, deployed serving, or production visual qualification.

Review fixes: merge scratch initialization now touches only partial-chunk tails;
resolved FA versions must be 2, 3 or 4; the custom op allocates its result in the
caller's tensor mode before entering inference mode for its kernels. The initial review-fix
suite passed 96 CPU checks (88 maskless plus eight FA-version cases), with seven
GPU/reference checks skipped. Regressions cover poisoned scratch reused across
full/partial chunks and calls, and output tensor mode under no-grad/inference
contexts in both eager and Inductor execution. HSDP GPU qualification is still
pending; these changes do not claim to establish it.

Further review fixes keep the two-element maxima tensors static in shape, reject
GEN/plan token-count mismatches before attention, gather directly into merge
scratch with in-place zeroing of excluded rows, and derive valid backend names
from sparse registrations plus maskless. The maskless and sparse-attention suites
passed 164 CPU checks (97 maskless, 67 sparse), with seven GPU/reference checks
skipped. New cases include singleton broadcasting mismatches in eager/compiled
execution and nonfinite placeholders in excluded instant/caption contributions.
