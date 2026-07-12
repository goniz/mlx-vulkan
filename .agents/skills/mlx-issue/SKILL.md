---
name: mlx-issue
description: Investigate and address a `goniz/mlx-vulkan` GitHub issue in the MLX submodule. Use when given an issue number and asked to read the issue, plan the work, reproduce a reported bug, implement the scoped fix, and work from an MLX feature branch based on `goniz/mlx:feat/vulkan`.
---

# Address an MLX Vulkan Issue

Use the supplied issue number to inspect `goniz/mlx-vulkan`. Work on a feature branch inside `mlx/` that tracks `goniz/mlx:feat/vulkan`; do not create implementation changes outside the allowed Vulkan backend scope unless the user explicitly expands it.

1. Read the issue with `gh issue view <number> --repo goniz/mlx-vulkan`.
2. Create and maintain a concise plan with `update_plan`; update it as work completes.
3. Inspect the submodule status and create a feature branch based on the `goniz/mlx:feat/vulkan` branch.
4. For a bug report, reproduce it before modifying implementation. Record the reproduction command and result.
5. Execute the planned investigation and implementation. Keep Vulkan work async and GPU-native; prefer `NYI` to pipeline-breaking CPU fallback.
6. Verify proportionately with `./dev.sh` commands. Run `./dev.sh generate` after Vulkan implementation changes.

Report the issue, plan outcome, branch, reproduction result, changes, and verification. Do not create a PR unless the user asks.
