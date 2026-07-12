---
name: mlx-pipeline-review
description: Review MLX Vulkan backend code for inference-performance hazards and async GPU pipeline violations. Use when asked to inspect a Vulkan file, function, class, symbol, diff, or code area for host synchronization, GPU readback, CPU fallback, avoidable copies or materialization, layout slow paths, redundant allocations, command-buffer breaks, or dispatch fragmentation. Report findings only; do not modify code unless explicitly asked in a separate request.
---

# MLX Vulkan Pipeline Review

Act as a Vulkan performance reviewer. Inspect the target named in the user's prompt and identify concrete inference-performance regressions or violations of this repository's async pipeline policy.

## Resolve the target

1. Treat the user's prompt as the review target. It may name a file, function, class, symbol, diff, commit, or short description of a code area.
2. If it names a path, read the full relevant file.
3. If it names a symbol, search for it and read its complete implementation.
4. If it has multiple matches, inspect the most relevant matches and state which locations were reviewed.
5. Read surrounding helpers and callees when they affect synchronization, layout, allocation, copies, fallback behavior, or pipeline continuity.
6. Do not stop at matching lines; trace enough context to understand the device-to-device execution path.

Keep the review focused on `mlx/backend/vulkan/`. Read Metal, CUDA, nearby Vulkan code, `llama.cpp/ggml/src/ggml-vulkan/`, or `references/` only when useful for comparison. Do not edit source while performing a review.

## Review priorities

Treat these as primary concerns:

- Host synchronization in hot paths: `eval()`, `wait()`, blocking event waits, host polling, stream flushes, or equivalent barriers.
- CPU extraction from GPU-backed arrays: `item<T>()`, `data<T>()`, host loops over tensor contents, host scratch materialization, or readback-driven branching.
- CPU fallbacks or mixed execution that break GPU kernel chaining.
- Avoidable copies: unnecessary `contiguous_copy_gpu`, staging or cast copies, flatten/materialize paths, or copy-back steps.
- Scalar or small-tensor special cases that leave the GPU pipeline.
- Hidden materialization caused by helpers, broadcasts, reshapes, or copy utilities.
- Layout handling that stages data when a view, shared buffer, reshape, slice, broadcast, as-strided representation, or direct shader dispatch is realistic.
- Silent slow paths that should instead be GPU-native or explicitly return `NYI`.

Treat these as secondary concerns:

- Obvious algorithmic inefficiency.
- Redundant allocation.
- Unnecessary command-buffer breaks or dispatch fragmentation.
- Behavior changes caused by fallback or materialization decisions.

## Apply the review standard

- Require every Vulkan `eval_gpu` path used by inference to remain asynchronous and pipeline-friendly.
- Prefer views, shared buffers, reshape/slice/broadcast/as-strided operations, existing Vulkan copy helpers, device fills, and compute dispatches.
- Prefer explicit `NYI` to host synchronization, CPU extraction, or pipeline-breaking fallback when no GPU-native implementation exists.
- Flag a copy only when it is clearly harmful or a realistic lower-copy design exists.
- Name the exact condition that triggers each regression.
- Separate definite findings from weaker observations. Do not inflate speculative concerns into findings.
- Do not propose a host-sync workaround.

## Analyze the execution path

1. Trace inputs through helpers that allocate, copy, cast, materialize, read back, synchronize, or fall back.
2. Verify that work stays on-device from inputs to outputs.
3. Check whether layouts and scalar cases use views or direct GPU operations before materialization.
4. Compare nearby Vulkan patterns and Metal or CUDA implementations when they clarify whether a faster design is feasible.
5. Confirm line numbers against the current working tree before reporting.

## Report the review

Return these sections in order.

### Scope

- State the target reviewed.
- List the files and functions inspected.

### Findings

List only substantiated findings, ordered by severity. For each finding include:

- Severity: `high`, `medium`, or `low`
- Location: `path:line` and function name
- Problem: one short sentence
- Why it hurts: the pipeline or performance impact
- Trigger: the exact condition that enters the path
- Recommended direction: concise guidance, not a patch

If there are no findings, state that explicitly.

### Non-issues

Briefly list suspicious areas that were checked and found acceptable. Omit this section only when none were examined.

### Summary

- Count findings by severity.
- Conclude briefly whether the target is pipeline-friendly.

Prioritize correctness over completeness. Keep the review limited to async pipeline and performance behavior rather than general style.
