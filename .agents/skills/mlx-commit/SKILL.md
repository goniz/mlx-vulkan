---
name: mlx-commit
description: Commit changes solely within the MLX submodule in this Vulkan repository. Use when asked to inspect, stage, and commit work in `mlx/` without modifying or committing the `mlx-vulkan` parent repository.
---

# MLX Submodule Commit

Commit only the `mlx/` submodule (`goniz/mlx`). Do not stage, commit, reset, or otherwise change the parent `goniz/mlx-vulkan` repository.

1. Inspect `git -C mlx status --short` and the staged and unstaged diffs.
2. Confirm the intended files and preserve unrelated user changes.
3. Use a user-supplied commit message when present; otherwise derive a concise imperative message from the intended diff.
4. Stage only the intended `mlx/` files and create the commit in the submodule.
5. Report the commit hash, message, and files committed. Also state whether the parent repository was left untouched.
