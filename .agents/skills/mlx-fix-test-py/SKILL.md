---
name: mlx-fix-test-py
description: Build MLX, reproduce one failing Python test, and implement the smallest Vulkan-only fix. Use when asked to repair a Python test failure in this repository while preserving test code and avoiding unrelated fixes.
---

# Fix One Vulkan Python Test

Fix exactly one Python failure per invocation.

1. Run `./dev.sh build`.
2. After the build completes, run `./dev.sh test-py -x`.
3. Identify the first failing test and reproduce only that failure as needed.
4. Do not modify test code. Modify implementation only under `mlx/mlx/backend/vulkan/`.
5. Before editing, inspect corresponding CUDA and Metal implementations under `mlx/mlx/backend/cuda/` and `mlx/mlx/backend/metal/`.
6. Make the smallest correct fix consistent with existing backend patterns and the async Vulkan pipeline policy.
7. Run `./dev.sh build`, then `./dev.sh test-py -x` sequentially to verify the addressed failure. Continue only on the same failure until it passes or a clear blocker remains.
8. After implementation changes, run `./dev.sh generate` to validate inference output coherence.

Report the failure addressed, Vulkan files changed, reference backend files consulted, and final test and generation results. Do not fix later or unrelated failures.
