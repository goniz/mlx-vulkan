---
description: Commit mlx submodule changes and create PR with benchmarks
---

1. First, check the git status of the mlx/ submodule: `cd mlx && git status && git log --oneline -3`
2. Commit the changes in mlx/ with a descriptive commit message
3. Push the mlx/ submodule to a new branch
4. Create a PR for the goniz/mlx repository that:
   - Indicates it closes the issue in goniz/mlx-vulkan repo (mention the issue number)
   - Includes benchmark results (run `./dev.sh benchmark bf16` and `./dev.sh benchmark 8bit` and include the output)
5. After creating the PR, provide the PR URL

Make sure benchmarks run successfully before attaching results.
