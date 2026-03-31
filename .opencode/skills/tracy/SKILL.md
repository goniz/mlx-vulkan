---
name: tracy
description: Set up and use Tracy profiler for MLX Vulkan backend profiling
---

# Tracy Setup and Usage for MLX Vulkan

## Purpose
Use this skill when you need to add, enable, or analyze Tracy profiling in a native C++ project, especially an MLX backend or Vulkan-heavy codebase. The goal is to get from “no profiling” to actionable traces with minimal nonsense and minimal hot-path damage.

This skill is grounded in the Tracy user manual quick-start and feature documentation. Tracy integration requires adding `TracyClient.cpp`, including `tracy/Tracy.hpp`, defining `TRACY_ENABLE` project-wide, and instrumenting code with macros such as `ZoneScoped` and `FrameMark`. Tracy also supports GPU profiling, memory profiling, plots, message logs, and call stacks. fileciteturn1file0

## When to use
Use this skill when the user asks to:
- set up Tracy in a C or C++ project
- instrument CPU hot paths
- profile Vulkan submission, queueing, synchronization, or kernels
- capture traces from a desktop app, CLI tool, benchmark, or test harness
- add memory, plot, lock, or message instrumentation
- analyze performance regressions with a real-time profiler

## When not to use
Do not use this skill when:
- the user only wants generic metrics export like Prometheus counters
- the project cannot add third-party source right now
- the user needs production telemetry only, not dev-time profiling
- the user wants a sampling-only external profiler with zero source integration

## Core facts to rely on
- Tracy is a real-time profiler with nanosecond-resolution timing, supporting CPU, GPU, memory allocations, locks, context switches, screenshots, and more. fileciteturn1file8
- The minimal integration is: add the Tracy repo, compile `TracyClient.cpp`, include `tracy/Tracy.hpp`, define `TRACY_ENABLE`, place `ZoneScoped` in functions of interest, and place `FrameMark` at frame boundaries. fileciteturn1file0
- The profiler GUI/server is built separately from the client. The manual recommends building the analyzer from the `profiler` directory with CMake. fileciteturn1file7
- Tracy can capture plots via `TracyPlot`, messages via `TracyMessage`, memory events via `TracyAlloc` / `TracyFree`, and GPU profiling including Vulkan. fileciteturn1file5

## Workflow

### 1. Add Tracy to the project
Preferred options:
- git submodule under `third_party/tracy`
- CMake `FetchContent`

Minimal source integration:
- compile `TracyClient.cpp`
- add include path for `tracy/public`
- include `tracy/Tracy.hpp` where instrumentation is used
- define `TRACY_ENABLE` for the whole target or build configuration intended for profiling fileciteturn1file0

### 2. Wire build system
For CMake-based native projects, do one of these:

#### Option A: submodule
```cmake
option(TRACY_ENABLE "Enable Tracy profiler" ON)
add_subdirectory(third_party/tracy)

target_link_libraries(my_target PRIVATE Tracy::TracyClient)
target_compile_definitions(my_target PRIVATE TRACY_ENABLE)
```

#### Option B: FetchContent
```cmake
include(FetchContent)
option(TRACY_ENABLE "Enable Tracy profiler" ON)

FetchContent_Declare(
  tracy
  GIT_REPOSITORY https://github.com/wolfpld/tracy.git
  GIT_TAG master
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(tracy)

target_link_libraries(my_target PRIVATE Tracy::TracyClient)
target_compile_definitions(my_target PRIVATE TRACY_ENABLE)
```

### 3. Add CPU instrumentation first
Start embarrassingly small:

```cpp
#include <tracy/Tracy.hpp>

void run_graph(Graph& g) {
    ZoneScoped;
    // work
}
```

For thread entry points:
```cpp
void worker_thread_main() {
    tracy::SetThreadName("mlx-vk-worker");
    ZoneScoped;
    // loop
}
```

For frame-like iterations or benchmark iterations:
```cpp
void render_or_step() {
    ZoneScoped;
    // work
    FrameMark;
}
```

`ZoneScoped` records scope timing plus source location metadata, while `FrameMark` establishes frame boundaries for the frame profiler. fileciteturn1file0 fileciteturn1file9

### 4. Add richer CPU annotations where useful
Use these when the trace starts looking like spaghetti thrown into a server rack:

```cpp
ZoneScopedN("encode_command_buffer");
ZoneText(cmd_name.c_str(), cmd_name.size());
ZoneValue(batch_size);
```

Useful macros include:
- `ZoneScopedN(name)` for stable custom labels
- `ZoneText(text, size)` for dynamic context
- `ZoneValue(uint64_t)` for numeric context
- `ZoneScopedC(color)` / `ZoneColor(color)` for visual grouping fileciteturn1file9

### 5. Add message log and plots for runtime state
For queue depth, bytes uploaded, command buffers submitted, descriptor set pressure, split-k count, and similar bread-and-butter signals:

```cpp
TracyPlot("vk.pending_submissions", pending_submissions);
TracyPlot("vk.bytes_uploaded", static_cast<int64_t>(bytes_uploaded));
TracyMessageL("transfer queue stalled");
```

You can configure plot presentation with `TracyPlotConfig(name, format, step, fill, color)`. Tracy supports number, memory, and percentage plot formats. fileciteturn1file5

### 6. Add memory instrumentation if allocator behavior matters
Wrap custom allocators or strategic allocation sites:

```cpp
void* p = allocate(sz);
TracyAlloc(p, sz);

TracyFree(p);
free(p);
```

This enables memory graphs, leak visibility, active-allocation rewind, and memory hot-spot analysis. fileciteturn1file5

### 7. Add Vulkan GPU profiling carefully
Tracy supports Vulkan GPU profiling. Use it for command-buffer spans, dispatch batches, copy/upload passes, and queue submission analysis. The exact Vulkan API wrappers/macros are documented in the Tracy manual’s GPU profiling section. fileciteturn1file1

Recommended pattern for an MLX Vulkan backend:
- keep CPU zones around graph compilation, descriptor updates, command recording, submission, fence wait, buffer sync, and pipeline creation
- add GPU zones around dispatch groups and transfer passes
- correlate CPU submission zones with GPU execution zones using stable names
- do not over-instrument every tiny helper first; begin with one zone per meaningful backend phase

Suggested first-pass zone map for MLX Vulkan:
- `compile_pipeline`
- `allocate_descriptor_sets`
- `record_compute_dispatch`
- `record_buffer_upload`
- `queue_submit_compute`
- `queue_submit_transfer`
- `wait_for_fence`
- `sync_buffers`
- `graph_execute`
- `kernel::<op_name>`

### 8. Build and run the Tracy profiler
The manual states the profiler/analyzer is built separately with CMake, for example:

```bash
cmake -B profiler/build -S profiler -DCMAKE_BUILD_TYPE=Release
cmake --build profiler/build --config Release --parallel
```

The analyzer can connect to localhost or remote clients. Tracy recommends using matching client/server versions because protocol changes between releases can break connections. fileciteturn1file7

### 9. Capture and analyze
Basic loop:
- run instrumented application
- launch Tracy profiler GUI
- connect to target
- reproduce workload
- inspect timeline, frame view, zone stats, plots, locks, memory, and call stacks

Tracy can also be used in interactive or command-line capture flows, and the manual has dedicated sections for capture and later analysis. fileciteturn1file1

## Practical rules
- Enable Tracy only in profiling builds.
- Keep zone names stable so stats group cleanly.
- Use dynamic text sparingly in hot paths.
- Prefer one meaningful zone over twenty decorative ones.
- Add plots for state you otherwise print in logs like a medieval peasant ringing a bell.
- Match Tracy client and server versions when possible. fileciteturn1file7

## Troubleshooting checklist
- No connection: verify `TRACY_ENABLE` is defined and the analyzer version matches the client version. fileciteturn1file7
- No data in a function: ensure `#include <tracy/Tracy.hpp>` is present and `ZoneScoped` is actually executed. fileciteturn1file0
- Build issues: ensure `TracyClient.cpp` is compiled exactly once in the target graph. The manual covers initial setup and CMake integration. fileciteturn1file0
- Too much overhead: reduce instrumentation density in ultra-hot inner loops and use plots or sampling strategically. Tracy’s overhead is low, but physics remains stubborn. fileciteturn1file3
- Confusing frame view in non-graphical apps: use `FrameMark` for logical iterations such as benchmark steps or inference requests. fileciteturn1file0

## Output expectations
When using this skill, produce:
- exact build-system changes
- exact source snippets
- a suggested first instrumentation pass
- a capture workflow
- interpretation guidance tied to the user’s architecture

## Example response skeleton
Use this structure when helping the user:
1. explain the smallest viable Tracy integration
2. show the exact CMake changes
3. show CPU instrumentation examples
4. show MLX Vulkan-specific placement points
5. optionally add plots, memory, and GPU zones
6. explain how to capture and read the first trace

## Reference
Primary source: Tracy Profiler user manual, including quick-start, build, instrumentation, plots, memory profiling, and GPU profiling sections. fileciteturn1file0
