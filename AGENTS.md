# Repo Purpose: Vulkan Backend

This repo is the parent repo if my MLX fork (goniz/mlx) at feat/vulkan branch.
This branch adds Vulkan GPU support to MLX as a new backend.

**Vulkan Backend Location**: `mlx/mlx/backend/vulkan/`


## Build Commands

**Important**: Always use `./dev.sh` for code-build-test cycles. This ensures consistent builds with the virtual environment at `./virtual-env` and proper CMake configuration.

### Quick Development Build (Recommended for daily development)

```bash
# Fast incremental build for development
# Assumes: ./dev.sh init-venv has been run once
./dev.sh build
```

**What it does:**
- Uses the virtual environment at `./virtual-env` (uv-based)
- Builds only the `core` Python extension target (skips tests/examples)
- Automatically copies the resulting `.so` to `python/mlx/` for editable installs
- Configures CMake with RelWithDebInfo, Vulkan support, and ccache

### Full Build with Tests

```bash
# Complete build including C++ tests
./dev.sh build

# Build wheel for distribution
./dev.sh build-wheel

# Run command inside the virtual environment
./dev.sh run python3 --version
./dev.sh run pytest python/tests/

# Fetch PR comments (use --submodule mlx for PRs in the mlx submodule)
./dev.sh pr-comments --submodule mlx
```

**What it does:**
- Runs CMake build with Vulkan backend enabled
- Builds all targets including C++ tests (`./build/tests/test_mlx`)
- Uses virtual environment at `./virtual-env`

### Build Scripts Summary

| Script | Use Case | Time | Output |
|--------|----------|------|--------|
| `./dev.sh init-venv` | First time setup, creates venv | ~1-2 min | `./virtual-env/` with dependencies |
| `./dev.sh build` | Daily development, quick iterations | ~2-5 min | Updates `python/mlx/core*.so` |
| `./dev.sh build-wheel` | Full build for distribution | ~10-15 min | Wheel in `wheelhouse/` |
| `./dev.sh run <cmd>` | Run command inside venv | varies | Executes command in virtual-env |
| `./dev.sh benchmark [quant]` | Run Qwen3 performance benchmark | ~1-2 min | Performance metrics (bf16 or 8bit) |
| `./dev.sh profile [model]` | Profile Qwen3 inference (0.6b or 2b) | ~1-2 min | Detailed per-layer timing and fallback analysis |
| `./dev.sh pr-comments [args]` | Fetch unresolved PR review comments | ~1s | Active comments from current PR (use `--submodule mlx` for submodule PRs) |

## Test Commands

```bash
# All Python tests
python -m unittest discover python/tests -v

# Run on specific device
DEVICE=gpu python -m unittest discover python/tests -v
DEVICE=cpu python -m unittest discover python/tests -v

# C++ tests
./build/tests/test_mlx

# Distributed tests
mpirun --bind-to none -np 8 python python/tests/mpi_test_distributed.py
mlx.launch --verbose -n 8 python/tests/ring_test_distributed.py
```

## Lint/Format Commands

```bash
pre-commit run --all-files    # Run all checks
clang-format -i file.cpp      # Format C++
black file.py                 # Format Python
isort --profile=black file.py # Sort Python imports
cmake-format -i CMakeLists.txt
```

## Code Style

### C++ (C++20, clang-format)
- Indentation: 2 spaces
- Namespaces: `mlx::core`
- Naming: `PascalCase` for classes, `snake_case` for functions/variables
- Private members: `trailing_underscore_`
- Headers: Use `#pragma once`
- Public API: Add `MLX_API` macro (see `mlx/api.h`)

### C++ Import Order
1. System headers (`<vector>`, `<memory>`)
2. Third-party (`<nanobind/...>`, `<doctest/...>`, `<vulkan/vulkan.hpp>`)
3. MLX headers (`"mlx/..."`)
4. Local headers (`"..."`)

### Python (black, isort)
- Naming: `PascalCase` for classes, `snake_case` for functions/variables
- All files start with: `# Copyright © 20XX Apple Inc.`

### Python Import Order
1. Standard library
2. Third-party (numpy, etc.)
3. `mlx.core` and mlx imports
4. Test utilities (`mlx_tests`)

## Error Handling

### C++
- Use exceptions: `std::invalid_argument`, `std::runtime_error`, `std::out_of_range`
- Throw with descriptive messages

### Python
- Raise: `ValueError`, `TypeError`, `RuntimeError`
- Use numpy-compatible error behavior

## Testing

### C++ Tests (doctest)
- Location: `tests/` directory
- Naming: `*_tests.cpp`
- Macros: `TEST_CASE("name")`, `CHECK()`, `CHECK_EQ()`, `CHECK_THROWS_AS()`
- Running with fail-fast: `./build/tests/tests --abort-after=1`

### Python Tests (unittest)
- Location: `python/tests/`
- Naming: `test_*.py`
- Base class: `mlx_tests.MLXTestCase`
- Array comparison: `self.assertEqualArray(mx_res, expected, atol=, rtol=)`
- Set `MLX_ENABLE_TF32=0` for deterministic results

## Project Structure

```
mlx/               # C++ core library
  backend/         # cpu, metal, cuda, vulkan backends
    vulkan/        # Vulkan backend (this branch)
      *.cpp/*.hpp  # C++ implementation
      kernels/     # Additional shader resources
        *.comp     # Vulkan compute shaders (GLSL)
  io/              # safetensors, gguf I/O
python/src/        # nanobind bindings
python/tests/      # Python unit tests
tests/             # C++ unit tests
```

## Vulkan-Specific Details

### Compute Shaders (`.comp` files)
- Shaders are compiled to SPIR-V at build time using `glslc`
- Compiled headers are generated in `${CMAKE_CURRENT_BINARY_DIR}/${shader_name}.spv.h`
- Shaders follow GLSL compute shader syntax
- Reference existing shaders in `mlx/backend/metal/kernels/` for algorithm patterns

### Build Dependencies
- `Vulkan::Vulkan` CMake package
- `glslc` (SPIR-V compiler) from Vulkan SDK

### Key Components
- **allocator.cpp/h**: Memory allocation for Vulkan buffers
- **device.cpp**: Vulkan device initialization and management
- **device_info.cpp/h**: Device capability queries
- **primitives.cpp**: Operation implementations using Vulkan compute shaders
- **eval.cpp**: Evaluation logic for Vulkan backend
- **event.cpp/fence.cpp**: Synchronization primitives
- **vulkan.cpp/h**: Core Vulkan context and utilities

## Common Patterns

### Adding a New Operation
1. Declare in `mlx/ops.h` with `MLX_API` macro
2. Implement in `mlx/ops.cpp`
3. Add primitive in `mlx/primitives.h/cpp` if needed
4. Add Vulkan kernel in `mlx/backend/vulkan/` (`.comp` shader + C++ wrapper)
5. Add Python binding in `python/src/ops.cpp`
6. Add tests in both C++ and Python

### Working with Arrays
- Use `mlx::core::array` for all array operations
- Arrays are immutable - operations return new arrays
- Use `StreamOrDevice s = {}` for device placement

## Notes for Agents

- **Always use `./dev.sh` for code-build-test cycles** - don't run cmake or pip install directly
- **NEVER run build and test commands in parallel** - tests depend on builds completing first
- **Use the `question` tool** to ask the user questions during execution (for preferences, requirements, clarifications, or implementation decisions)
- Run tests after making changes
- Format code before committing
- Follow existing patterns in the codebase
- Check both CPU and GPU backends when applicable
- For Vulkan work: reference Metal backend (`mlx/backend/metal/`) for compute patterns and CUDA backend (`mlx/backend/cuda/`) for structure
- For Vulkan Reference implementation read ./llama.cpp/ggml/src/ggml-vulkan/
- **Use `references/` directory** to learn about dependencies and technologies used in this project
- Prefer throwing `NYI`/not-yet-implemented errors instead of falling back to CPU implementations when Vulkan support is missing
- ALWAYS check for existing shaders in ./mlx/backend/vulkan/kernels/ before introducing new onces
- Shaders should be compiled automatically by CMake; check build output if shaders fail
- NEVER edit source files outside of mlx/backends/vulkan !! (test files are allowed)

## Github instructions
- Every PR that you create, should contain the results of qwen3 benchmark by running it `./dev.sh benchmark [bf16|8bit]` against bf16 and 8bit quants, either as pr desc or as comment 
