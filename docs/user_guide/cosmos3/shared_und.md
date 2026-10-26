# Cosmos3 with one UND owner

`cosmos3_shared_und` is an opt-in two-stage topology for Nano BF16. Stage 0
runs the native vLLM Cosmos3 understanding model through Omni's AR runner and
`OmniARAsyncScheduler`. Stage 1 loads GEN and the VAE, consumes exported UND
K/V, and computes GEN rotary frequencies locally. It allocates no UND
embeddings, decoder layers, or final normalization weights.

This topology requires the local vLLM Cosmos3 implementation. Existing Cosmos3
auto-discovery and single-stage deployments are unchanged. GPU numerical,
resource, and performance acceptance remains required; CPU contract tests do
not establish model quality or measured memory savings.

## Deploy

```bash
vllm serve nvidia/Cosmos3-Nano --omni \
  --revision <checkpoint-commit> \
  --deploy-config vllm_omni/deploy/cosmos3_shared_und.yaml
```

Use the same checkpoint revision for both stages; do not override one stage's
model or revision. Initial full-resolution validation targets an H200 141 GB.
The supplied deployment puts both processes on GPU 0 with TP=1. It configures
four AR sequences, an 8,192-token model limit, a 4,096-token batched-token
budget, an 8 GiB AR KV cache, and one active GEN request. Chunked prefill and
asynchronous AR scheduling are enabled. Inter-stage asynchronous chunking is
disabled and diffusion runs eagerly. Shared-memory transfers copy K/V from GPU
to host and back to GPU.

Stage 1 skips the generic standalone diffusion warmup because it requires real
stage-0 conditioning exports. The first media request warms GEN and the VAE
through the full two-stage path; include it before collecting steady-state timings.

Chat defaults to text output and terminates at stage 0, including streaming
chat with image inputs. Use `/v1/images/generations` for T2I and the existing
synchronous or asynchronous video routes for T2V/I2V. Media requests must select
an explicit output modality. Chat requests selecting `video` return HTTP 400
with a pointer to `/v1/videos` and `/v1/videos/sync`; chat cannot render video
responses. Offline `Omni.generate` uses the same admission path: pass a prompt
dictionary with `modalities: ["image"]` or `["video"]`,
and a sampling-parameter entry for each stage. An I2V reference is a decoded
PIL image under `multi_modal_data.image`; the frontend retains it for GEN's VAE
and sends only canonical text tokens to UND.

The image API forwards `negative_prompt`, including an empty string, and
Cosmos3 system-prompt controls use Cosmos3 formatting. This first version does
not implement prompt expansion, critique loops, image editing, V2V/transfer,
audio/action generation, Edge/distilled/quantized variants, LoRA, offloading,
distributed execution, diffusion caching, or multiple outputs per request.
AR can schedule while GEN is active; there is no GPU execution arbiter or
reasoning-latency guarantee.

## Conditioning contract

The shared CPU formatter owns JSON metadata, prompt templates, system-prompt
text, truncation, and appended EOS/vision-start tokens. Admission resolves
image dimensions (including I2V aspect-derived defaults), rounded frame count,
and FPS before tokenization. GEN receives those same settings. Native chat and
multimodal processing remain in use for reasoning.

Conditioning submits explicit token IDs without another chat template. Expected
positive and negative lengths are recorded independently before execution.
CFG creates a negative request even for an empty negative prompt; branch
lengths may differ. All companions are constructed before the parent is
submitted, then admitted independently without capacity reservation. The
parent declares its companions before AR admission so an early parent output
cannot bypass the completion gate. Only processed outputs satisfy that gate;
a raw `kv_ready` notification cannot dispatch conditioning to GEN.

The scheduler keeps blocks alive through extraction. For conditioning only,
asynchronous decode lookahead selects the canonical prefill prefix. Other
pipelines retain their existing snapshot semantics. GEN rejects short/long
exports, partial extraction, metadata disagreement, missing or misordered
layers, incorrect heads/dimensions, dtype disagreement, and differing model or
checkpoint revision identities. It never silently
truncates imported tensors or falls back to local UND.

Each validated branch owns `[1, S, Hkv, D]` K/V, a mask of exactly `S` tokens,
and GEN rotary frequencies computed for that branch's text length, final
latent geometry, and FPS. Imported UND keys are already rotated. Transformer
caches reset before frequencies are prepared; `_kv_load_und` installs both K/V
and frequencies before each branch forward and reports success, including
without the experimental session manager. Completion, failure, and cancellation
release request state.

`cfg_companion_wait_timeout_s` is optional and defaults to absent for existing
deployments. This deployment sets it to 300 seconds. The first parent deferral
starts a monotonic deadline; repeated deferral cannot extend it. An orchestrator
background task checks every second independently of both output-loop variants.
It reconciles bindings and live replicas and claims failure before awaiting
cleanup. The separate K/V receive timeout starts at retrieval, so it cannot
replace the companion deadline.

Completed companions retain their transfer resources until parent cleanup.
Serialized worker RPCs release unconsumed dense exports on both endpoints;
a surviving receiver can reclaim shared memory after a sender dies.

## Validation status and acceptance

The isolated CPU contract suite can run without a vLLM binary:

```bash
python -m pytest --noconftest -c /dev/null \
  -o cache_dir=/tmp/cosmos3-pytest-cache \
  tests/diffusion/models/cosmos3/test_shared_und_contract.py
```

With the repository vLLM environment, also run:

```bash
python -m pytest \
  tests/config/test_cosmos3_shared_und.py \
  tests/diffusion/test_diffusion_engine_dummy_run.py \
  tests/engine/test_cfg_companion_tracker.py \
  tests/engine/test_cfg_companion_lifecycle.py \
  tests/engine/test_stage_pool_release_batching.py \
  tests/core/sched/test_omni_ar_scheduler_kv_transfer.py \
  tests/distributed/omni_connectors/test_kv_flow.py \
  tests/diffusion/models/cosmos3/test_cosmos3_pipeline.py \
  tests/diffusion/models/cosmos3/test_cosmos3_transformer.py

python -m pytest tests/e2e/online_serving/test_cosmos3_shared_und.py \
  -m 'advanced_model and cuda' --run-level=advanced_model
```

Before production acceptance, record exact Omni/vLLM/checkpoint revisions and
hardware, and complete the following gates:

- Compare real AR exports against internal UND outputs layer by layer with
  dtype-appropriate tolerances under sync/async scheduling, chunked prefill,
  and KV block boundaries. Compare canonical tokens, masks, metadata, positions,
  dimensions, and denoising using identical initial latents.
- Exercise unequal branch lengths, repeated CFG switches, CFG intervals,
  no-CFG, and consecutive requests changing lengths, geometry, and FPS.
- Run streaming reasoning, T2I, synchronous/asynchronous T2V and I2V, alternating
  and concurrent requests, and generated-image reasoning on one server. Compare
  reasoning with standalone vLLM and generation with the original pipeline;
  review the actual generated artifacts.
- Verify exactly one UND weight allocation. Against a duplicated-UND baseline,
  record TTFT, time per output token, throughput, conditioning-prefill time,
  transfer bytes/latency, generation latency, and peak device memory. Report
  weight savings separately from AR KV and peak execution memory. No benchmark
  or numerical-parity claim is established by this change's CPU tests.

Prefix caching stays **off by default**. A separate explicitly enabled run must
prove real cache hits, partial-prefix reuse, repeated conditioning and CFG,
cache pressure, shared-block cancellation, exact exports, and numerical parity
before that default changes. Merely enabling caching or repeating a prompt is
not evidence of a cache hit.

## Rollback

Switch the deploy config back to `vllm_omni/deploy/cosmos3_omni.yaml`, or omit the
shared-UND deploy config to use existing Cosmos3 discovery. That restores the
single-stage diffusion pipeline with its internal UND owner.
