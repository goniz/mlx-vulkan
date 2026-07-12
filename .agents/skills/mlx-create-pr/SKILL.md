---
name: mlx-create-pr
description: Commit MLX submodule changes, run required Vulkan benchmarks, push a branch, and open a PR against `goniz/mlx`. Use when asked to create a pull request for MLX Vulkan work, especially when an issue number must be closed and bf16 plus 8-bit benchmark results are required.
---

# Create an MLX PR

Work in `mlx/` (`goniz/mlx`) and leave the `goniz/mlx-vulkan` parent repository uncommitted and otherwise untouched.

1. Inspect `git -C mlx status --short` and `git -C mlx log --oneline -3`.
2. Confirm the intended diff, then commit it with a descriptive message. Do not include unrelated changes.
3. Push the submodule commit to a new branch.
4. Run `./dev.sh benchmark bf16`, wait for completion, then run `./dev.sh benchmark 8bit`. Never run benchmarks in parallel.
5. Create a PR in `goniz/mlx`. Include the relevant `goniz/mlx-vulkan` issue-closing reference supplied by the user and both benchmark outputs in the PR description.
6. Report the PR URL, branch, commit, and benchmark results.

Do not open the PR until both benchmarks have completed successfully.
