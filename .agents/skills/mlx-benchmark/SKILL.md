---
name: mlx-benchmark
description: Run and compare MLX Vulkan inference benchmarks. Use when asked to benchmark this repository, measure a change, run bf16 or 8-bit MLX performance tests, or compare current results with the README baseline.
---

# MLX Benchmark

Run benchmarks through `./dev.sh` only. Treat any user-supplied model or benchmark option as an override; otherwise run both standard precisions.

1. Read `README.md` and identify the relevant current baseline numbers.
2. Run `./dev.sh benchmark bf16`.
3. After it completes, run `./dev.sh benchmark 8bit`.
4. Never run benchmark commands concurrently or in parallel tool calls.
5. Compare each result with the comparable README baseline. State the absolute and percentage difference when the metrics permit it; otherwise explain the direct comparison.

Report the commands run, results for each precision, the baseline used, and the performance difference. Do not change code unless explicitly asked.
