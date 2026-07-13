---
name: mlx-vulkan-review-fix
description: Review the MLX Vulkan backend for one major async-pipeline or inference-performance issue, obtain user approval, implement the scoped fix, then run full MLX validation, benchmarks, and model generation reporting. Use when the user wants an end-to-end Vulkan improvement workflow with an explicit review-before-fix approval boundary.
---

# MLX Vulkan Review, Fix, and Validate

Run this workflow in two phases. Do not begin phase 2 until the user explicitly approves the proposed fix.

## Phase 1: Review and approval

1. Read and follow `$mlx-pipeline-review` from `.agents/skills/mlx-pipeline-review/SKILL.md`.
2. Review the full `mlx/backend/vulkan/` backend and identify exactly one substantiated high-severity issue. Prioritize host synchronization, host readback, CPU fallback, unnecessary materialization/copies, or severe dispatch fragmentation.
3. Return the pipeline-review report format: Scope, Findings, Non-issues, and Summary. Report only the single strongest finding.
4. Ask whether to fix the finding. Wait for explicit approval. Do not edit source, build, test, benchmark, or run model-report before approval.

## Phase 2: Implement after approval

1. Restate the approved scope and any user constraints. Inspect `git status` in the parent repository and `mlx/`; preserve unrelated changes.
2. Trace the full affected Vulkan execution path and consult nearby Metal/CUDA code or `llama.cpp/ggml/src/ggml-vulkan/` only when it clarifies a GPU-native design.
3. Implement the smallest correct Vulkan-only fix. Keep inference work asynchronous and on-device. Prefer an explicit `NYI` over host synchronization, readback, CPU fallback, or a pipeline-breaking workaround.
4. Do not modify source outside `mlx/backend/vulkan/`; test files are allowed. Do not add a new test unless the user explicitly requests one.
5. Build with `./dev.sh build`. Never run builds, tests, benchmarks, or model-report concurrently.

## Repair and validation loop

After a successful build, run these sequentially:

1. `./dev.sh test-cpp`
2. `./dev.sh test-py`

For a failure, reproduce it with the narrowest relevant `./dev.sh test-cpp` or `./dev.sh test-py` command, determine whether it is caused by the change, and make the smallest permitted fix. Rebuild and rerun the failed test, then restart the full validation sequence. Continue until both full suites pass, or stop and report a concrete blocker that cannot be resolved within the allowed source scope.

Run `./dev.sh generate` after the final test pass and confirm that output is coherent.

## Performance and generation checks

Read `benchmarks/results.csv` before benchmarking and use the latest comparable row for each model/precision as the baseline. If no comparable row exists, say so rather than comparing incompatible hardware or settings.

Run these benchmarks strictly one at a time, in this order:

1. `./dev.sh benchmark bf16`
2. `./dev.sh benchmark 8bit`
3. `./dev.sh benchmark 8bit --model mlx-community/Qwen3.6-35B-A3B-8bit`

For each benchmark, report prompt throughput, generation throughput, peak memory, and absolute/percentage differences from the selected `benchmarks/results.csv` baseline. Do not compare a different model, quantization, hardware, or benchmark configuration as though it were equivalent.

Then run `./dev.sh model-report` once, serially, and report every failed model or state that all default models completed successfully.

## Final handoff

Report:

- The approved finding and the implemented fix, with file links.
- Full C++ and Python test results, including any repairs made during the loop.
- Generation result.
- A benchmark table with current values, baseline values, and absolute/percentage changes.
- Full model-report outcome.
- Remaining limitations, uncommitted files, and any blocker.

Do not commit, push, open a PR, or modify benchmark baseline files unless the user explicitly asks.
