---
name: mlx-review
description: Review changes in the MLX submodule for concrete bugs, behavioral regressions, structural mismatches, and obvious performance problems. Use when asked to review uncommitted `mlx/` changes, a supplied diff, commit, file, or change set and provide actionable evidence-based feedback without modifying code.
---

# Review MLX Changes

Review code changes, not pre-existing code. Do not modify files.

1. Resolve the requested review target. With no target, inspect `git -C mlx diff`, `git -C mlx diff --cached`, and `git -C mlx status --short`.
2. Read the full contents of every modified and untracked file, not only the diff. Read `AGENTS.md`, `.editorconfig`, and other applicable conventions.
3. Investigate uncertain behavior through surrounding code and established patterns before reporting an issue.
4. Prioritize concrete bugs: logic and boundary errors, incorrect guards, error handling, race conditions, security risks, and unintended behavior changes.
5. Flag structural mismatches only when they conflict with established project patterns. Flag performance only when clearly problematic, such as unbounded quadratic work or blocking I/O on a hot path.
6. Do not invent hypothetical concerns. State the realistic input, environment, or control-flow condition required for every finding.

Report findings first, ordered by severity, with `path:line`, a concise explanation, the triggering scenario, and why it is a defect. State explicitly when no actionable findings remain. Keep the tone factual and avoid non-actionable praise or style preferences.
