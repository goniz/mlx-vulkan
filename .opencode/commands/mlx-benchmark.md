---
description: Run the MLX benchmarks
model: opencode-go/deepseek-v4-flash
subtask: true
---
Run the benchmarks using `@dev.sh benchmark` (both bf16 and 8bit).
**IMPORTANT: Never run benchmarks in parallel tool calls** — benchmarks must run sequentially, one at a time.
Read the file @README.md to see the latest baseline numbers.
After the benchmarks finished running, indicate the performance difference against baseline

Optional User input: $ARGUMENTS
