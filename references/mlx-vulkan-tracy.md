## Project-specific guidance for MLX Vulkan
This part is where the rubber meets the weird GPU road.

Instrument these subsystems first:

### Pipeline creation
Your Vulkan backend prebuilds many compute pipelines. Add zones around pipeline creation and statistics retrieval so you can separate startup compile cost from steady-state execution. The reference implementation also queries pipeline executable statistics when supported, which pairs nicely with Tracy CPU zones around pipeline creation. fileciteturn1file13

### Submission pipeline
Add zones around command pool acquisition, command buffer recording, semaphore wiring, queue submission, and fence waits. These areas are classic latency gremlins in deferred submission designs.

### Buffer synchronization
Wrap explicit sync points such as buffer flushes, host-device staging, split-k reduction syncs, and waits. The ggml Vulkan reference has explicit `ggml_vk_sync_buffers(...)` points that are natural anchors for Tracy zones. fileciteturn1file10

### Kernel-level dispatches
Place zones around high-level dispatch wrapper functions rather than every microscopic helper. For example, one zone at `ggml_vk_dispatch_pipeline(...)` call sites often gives more signal than confetti-bombing the whole file.

### Memory pressure
If you have pinned host memory, staging buffers, sub-buffers, or allocator churn, add Tracy memory instrumentation to allocator entry points and key buffer lifecycle sites.

## Recommended instrumentation order
1. one zone around whole graph execution
2. zones around queue submit and waits
3. zones around command recording
4. zones around pipeline creation / cache misses
5. plots for queue depth, uploaded bytes, compiled pipelines, active command buffers
6. memory instrumentation
7. Vulkan GPU zones
8. optional call stacks for stubborn hotspots

