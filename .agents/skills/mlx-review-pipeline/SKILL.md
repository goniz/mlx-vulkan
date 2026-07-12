---
name: mlx-review-pipeline
description: Review MLX Vulkan backend code for inference-performance hazards and async GPU pipeline violations. Use when asked to inspect a Vulkan file, function, class, symbol, diff, or code area for host synchronization, GPU readback, CPU fallback, avoidable copies or materialization, layout slow paths, redundant allocations, command-buffer breaks, or dispatch fragmentation. Report findings only; do not modify code unless explicitly asked in a separate request.
---

# MLX Vulkan Pipeline Review

Act as a Vulkan performance reviewer. Inspect the target in the user's prompt and identify concrete inference-performance regressions or violations of this repository's async pipeline policy.

## Resolve the target

1. Treat the prompt as a file, function, class, symbol, diff, commit, or code-area target.
2. Read the full relevant implementation. For multiple matches, inspect the most relevant ones and name the locations reviewed.
3. Read surrounding helpers and callees that affect synchronization, layout, allocation, copies, fallbacks, or pipeline continuity.
4. Trace enough context to understand the end-to-end device execution path.

Keep the review focused on `mlx/backend/vulkan/`. Use nearby Vulkan, Metal, CUDA, `llama.cpp/ggml/src/ggml-vulkan/`, or `references/` patterns only when they clarify a realistic alternative. Do not edit code.

## Inspect for pipeline hazards

Prioritize:

- Host synchronization: `eval()`, `wait()`, blocking events, host polling, or stream flushes.
- CPU extraction: `item<T>()`, `data<T>()`, host tensor loops, scratch materialization, or readback-driven branches.
- CPU fallbacks or mixed execution that break GPU kernel chaining.
- Avoidable `contiguous_copy_gpu`, staging, casts, flattening, materialization, and copy-back paths.
- Host-side scalar or small-tensor fast paths.
- Hidden materialization through helpers, broadcasts, reshapes, or copy functions.
- Layout staging that a view, shared buffer, reshape/slice/broadcast/as-strided operation, direct GPU copy, device fill, or shader dispatch can avoid.
- Silent slow paths where GPU-native support or explicit `NYI` is preferable.

Also inspect algorithmic inefficiency, redundant allocation, command-buffer breaks, and dispatch fragmentation.

## Apply the standard

- Keep every inference-facing Vulkan `eval_gpu` path asynchronous and pipeline-friendly.
- Prefer device-native views, shared buffers, existing Vulkan copies, fills, and compute dispatches.
- Prefer `NYI` over host sync, CPU extraction, or a pipeline-breaking fallback when no GPU-native implementation exists.
- Flag copies only when they are clearly harmful or a realistic lower-copy alternative exists.
- State the exact condition that triggers every finding. Do not report speculative concerns as defects.

## Report

Use these sections in order:

### Scope

- Target reviewed.
- Files and functions inspected.

### Findings

List only substantiated findings, ordered by severity. For each give:

- Severity: `high`, `medium`, or `low`
- Location: `path:line` and function name
- Problem: one short sentence
- Why it hurts: pipeline or performance impact
- Trigger: the condition that enters the path
- Recommended direction: concise guidance, not a patch

State explicitly if there are no findings.

### Non-issues

List suspicious areas checked and found acceptable, if any.

### Summary

- Finding counts by severity.
- Brief conclusion on pipeline friendliness.

Prioritize correctness over completeness and keep the review limited to pipeline and performance behavior, not general style.
