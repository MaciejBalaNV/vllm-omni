---
title: Engine Orchestration
kind: module
status: draft
architecture_state: current-plus-in-flight
owners:
  - "@tzhouam"
  - "@fake0fan"
document_stewards:
  - "@hsliuustc0106"
  - "@Gaohan123"
  - "@david6666666"
required_reviewers:
  - "@yinpeiqi"
  - "@Sy0307"
  - "@yenuo26"
  - "@NickCao"
primary_code_paths:
  - vllm_omni/engine/omni_engine_base.py
  - vllm_omni/engine/async_omni_engine.py
  - vllm_omni/engine/async_engine_utils.py
  - vllm_omni/engine/orchestrator.py
  - vllm_omni/engine/cfg_companion_tracker.py
  - vllm_omni/engine/rpc_result_router.py
related_code_paths:
  - vllm_omni/engine/messages.py
  - vllm_omni/engine/orchestrator_monitor.py
  - vllm_omni/engine/stage_client.py
  - vllm_omni/engine/stage_pool.py
  - vllm_omni/outputs/output_processor.py
  - vllm_omni/distributed/omni_coordinator/**
  # The duplex siblings (DuplexOmniEngine / DuplexOrchestrator) and the
  # engine-resident session framework are owned by ../fullduplex.md; listed
  # here because they fill the template seams of OmniEngineBase and
  # OrchestratorBase.
  - vllm_omni/engine/duplex_omni_engine.py
  - vllm_omni/engine/duplex_orchestrator.py
  - vllm_omni/engine/duplex/**
depends_on:
  - input_output_modality_contracts.md
  - error_contracts.md
  - stage_runtime.md
  - omni_connector.md
validation_paths:
  - tests/engine/test_async_omni_engine_input.py
  - tests/engine/test_async_omni_engine_outputs.py
  - tests/engine/test_async_omni_engine_abort.py
  - tests/e2e/offline_inference/test_qwen3_omni.py
  - tests/engine/test_async_omni_engine_stage_init.py
  - tests/engine/test_orchestrator.py
  - tests/engine/test_orchestrator_error_handling.py
  - tests/engine/test_orchestrator_stage_input_bridge.py
  - tests/engine/test_cfg_companion_lifecycle.py
  - tests/engine/test_rpc_result_router.py
  - tests/engine/duplex/**
  - tests/engine/test_duplex_orchestrator.py
  - tests/engine/test_duplex_omni_engine.py
  - tests/engine/test_duplex_import_boundary.py
upstream_refs:
  - vllm.v1.engine.EngineCoreRequest
  - vllm.v1.engine.EngineCoreOutputs
  - vllm.v1.engine.exceptions.EngineDeadError
  - vllm.outputs.RequestOutput
invariant_namespace: ORCH-INV
last_reviewed: 2026-08-07
last_verified_commit: 3d7fc3b9ba3cac88d579d4dc35b78b0b641675fc
---

# Engine orchestration

Engine orchestration coordinates configured AR and diffusion stages while
keeping public clients independent from stage processes and transports.

## Contract status

This document is a draft description of current behavior. It also identifies
the boundary affected by the in-flight stage client/process refactor in
[#5441](https://github.com/vllm-project/vllm-omni/pull/5441). Names and
responsibilities proposed only by that PR are not current contracts.

The orchestration loop is event-driven by default, as proposed in
[#5221](https://github.com/vllm-project/vllm-omni/pull/5221);
`VLLM_OMNI_EVENT_DRIVEN_ORCH=0` selects the legacy 1 ms poll loop. The mode changes poll
cadence only: the routing, ordering, and terminal-state contracts below hold
identically on both loops.

## Ownership boundary

This document owns `AsyncOmniEngine`, `Orchestrator`, request-state creation,
cross-stage routing, output ordering, companion tracking, control/RPC
correlation, cancellation propagation, and terminal-state convergence.

It does not own stage placement, replica selection, stage-process startup,
payload schemas, public protocol rendering, connector transport, or semantic
error classification. Those responsibilities belong to the stage runtime,
I/O, entrypoint, connector, and error contracts.

## Candidate invariants

These identifiers are proposals while the document is `draft`.

### ORCH-INV-001: The orchestrator owns cross-stage routing

**Rule:** Entrypoints and stage clients MUST NOT independently forward a
request to a downstream logical stage.

### ORCH-INV-002: Stage clients do not own routing policy

**Rule:** Stage clients MUST implement communication and lifecycle operations
without selecting the next logical stage.

### ORCH-INV-100: Terminal state is monotonic

**Rule:** Once a request reaches a terminal state, orchestration MUST NOT
forward new work for that request.

## Invariant namespace

`ORCH-INV` reserves `001-099` for dependency direction, `100-199` for
request state and ordering, `200-299` for failure/cancellation/cleanup, and
`300-399` for extension points, shutdown ordering, and upstream alignment.
Numbers become append-only after normative promotion.

## Safe-change guide

Test routing, output ordering, queue correlation, cancellation, failure
propagation, shutdown ordering, and representative multi-stage execution.
AR abort of a final-stage LLM request must deliver a terminal output with
`finish_reason="abort"` and the generated prefix so collocated training can
resume; diffusion abort remains whole-sample retry. Aborting a parallel-sampling
child must drop `parent_requests` once no children remain. EngineCore control
RPCs (`sleep` / `wake_up` / `pause_scheduler` / `resume_scheduler`) must
propagate worker exceptions rather than returning `{"supported": False}`.
When a request has multiple final output stages, only the last abort message
is request-terminal. Output-processor abort state is committed only after the
physical EngineCore abort succeeds.
Cleanup claims abort-output ownership before fencing late output and dispatch.
That cleanup may deliver the generated prefix while the request is terminating;
internal companions and requests with an already-claimed terminal outcome do
not emit an additional abort output.
Changes that cross into stage runtime or public error behavior require review
from that contract's owners.

Multimodal cache-miss feedback invalidates only the reported hashes in the
owning sender cache before request error delivery. Downstream caches are
updated under the preprocessing lock; stage 0 feedback crosses to the frontend
and is applied while draining outputs, including cancelled requests. Ordinary
request failures retain retryable classification; duplex session termination
keeps its existing semantics.

## Promotion gate

- Reconcile current names and boundaries after #5441 merges, closes, or is
  replaced.
- Demonstrate one terminal outcome per request across success, abort,
  request-scoped failure, and fatal engine failure.
- Verify queue correlation and orchestrator-before-runtime shutdown ordering
  in the cited validation paths.
- Obtain approval from a technical owner and an independent validation
  reviewer.

## CFG companion completion and liveness

Companions are built before the parent is queued. The parent submission carries
its expected companion IDs, which the orchestrator registers before submitting
the parent to stage 0. The AR scheduler still admits each request independently.
Processed companion outputs must be stored before completion is recorded.
Requests declaring `omni_wait_for_processed_kv_output` also require the parent's
processed output before forwarding; raw K/V readiness does not satisfy this gate.
This flag is internal: frontend admission removes caller-supplied values before
running the trusted prompt transform, which may set it for conditioning requests.
Preprocessed caller requests are sanitized at the same boundary.

When `cfg_companion_wait_timeout_s` is configured, the first parent deferral
starts a monotonic, non-renewing deadline. A one-second background task owned by
the orchestrator runs independently of polling and event-driven output loops.
It checks pending companions against bindings and replica membership, and fails
silent stalls at the deadline. Failure claims terminal state before awaiting
cleanup, aborts remaining companions, and releases tracker, binding, transfer,
and request state. Late outputs cannot forward work after this claim. This
wait is distinct from the receive timeout inside the K/V transfer manager.
