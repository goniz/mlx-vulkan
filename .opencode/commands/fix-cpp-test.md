---
description: Run C++ tests and fix one Vulkan failure
---

Build before running the C++ tests:

```bash
./dev.sh build
```

Then run the C++ tests with:

```bash
./dev.sh test-cpp --fail-after=1
```

Fix exactly one failing test.

Rules:

- Do not modify test code.
- Only modify implementation code under `mlx/mlx/backend/vulkan/`.
- Before changing Vulkan code, look for reference implementations in `mlx/mlx/backend/cuda/` and `mlx/mlx/backend/metal/`.
- Prefer the smallest correct fix that matches existing backend patterns.
- After editing, run `./dev.sh build` before `./dev.sh test-cpp --fail-after=1` again to verify the addressed failure.
- If the same test still fails, continue fixing only that failure until it passes or the blocker is clear.
- Do not fix unrelated failures in the same run.

Report:

- The failing test that was addressed.
- The Vulkan files changed.
- The reference backend files consulted.
- The final test command result.
