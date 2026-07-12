---
name: mlx-pr-comments
description: Fetch and address unresolved pull-request review comments for the MLX submodule. Use when asked to review outstanding feedback with `./dev.sh pr-comments --submodule mlx`, fix actionable Vulkan comments, or reply with an evidence-based explanation for comments that need no code change.
---

# Address MLX PR Comments

1. Run `./dev.sh pr-comments --submodule mlx` to fetch unresolved review feedback.
2. Inspect the full code context and comment thread for every unresolved comment.
3. For each comment, either make the necessary fix or reply with a concise, evidence-based reason that no change is needed.
4. Preserve unrelated user changes and follow the Vulkan async pipeline policy for implementation work.
5. Verify code changes with appropriate sequential `./dev.sh` build and test commands, then run `./dev.sh generate` after Vulkan implementation changes.
6. Continue until every unresolved comment has been fixed or explicitly addressed with a reply.

Report each comment's disposition, files changed, verification performed, and any remaining external blocker.
