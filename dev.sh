#!/bin/bash

set -euo pipefail

# Development entrypoint script
# Combines build-venv.sh, build-editable.sh, and build-vulkan.sh functionality
# Usage: ./dev.sh <command>
#
# Commands:
#   init-venv     Create and setup virtual environment (uv-based)
#   build         Fast editable build for development
#   build-wheel   Build wheel for distribution
#   benchmark     Run Qwen3 benchmark (bf16 or 8bit)
#   profile       Profile Qwen3 model inference with detailed tracing

show_help() {
    cat << EOF
Development entrypoint for MLX Vulkan

Usage: ./dev.sh <command> [options]

Commands:
  init-venv         Create and setup virtual environment
  build             Fast editable build for development
  build-wheel       Build wheel for distribution
  benchmark [quant] Run Qwen3 benchmark (default: bf16, use "8bit" for quantized)
  profile [model]   Profile Qwen3 model inference (0.6b or 2b, default: 0.6b)

Examples:
  ./dev.sh init-venv
  ./dev.sh build
  ./dev.sh build-wheel
  ./dev.sh benchmark        # Run with bf16
  ./dev.sh benchmark 8bit   # Run with 8-bit quantization
  ./dev.sh profile          # Profile 0.6B model
  ./dev.sh profile 2b       # Profile 2B model
EOF
}

cmd_init_venv() {
    echo "Initializing virtual environment..."
    uv venv --clear --seed --python 3.13 virtual-env
    source virtual-env/bin/activate

    SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
    PTH_FILE="${SITE_PACKAGES}/mlx-dev.pth"
    echo "$(pwd)/mlx/python" > "$PTH_FILE"
    echo "Created pth file: $PTH_FILE"

    uv pip install mlx-lm mlx-vlm
    # Uninstall the PyPI mlx package to avoid shadowing our local build
    uv pip uninstall mlx -y
    uv pip install pytest
    # Uncomment if you need PyTorch with ROCm support:
    # uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm7.2
    echo "Virtual environment initialized successfully!"
}

cmd_build() {
    echo "Running editable build..."

    BUILD_DIR="build"

    source virtual-env/bin/activate

    PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")
    PLATFORM=$(python -c "import platform; print(platform.machine().lower())")

    if [[ "$OSTYPE" == "darwin"* ]]; then
        SO_FILENAME="core.cpython-${PYTHON_VERSION}-darwin.so"
    else
        SO_FILENAME="core.cpython-${PYTHON_VERSION}-${PLATFORM}-linux-gnu.so"
    fi

    echo "Building for Python ${PYTHON_VERSION} on ${PLATFORM}..."
    echo "Expected output: ${SO_FILENAME}"

    CMAKE_ARGS="-DCMAKE_BUILD_TYPE=RelWithDebInfo -DMLX_BUILD_VULKAN=ON -DMLX_USE_CCACHE=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
    CMAKE_ARGS="-DMLX_BUILD_TESTS=ON $CMAKE_ARGS"

    if command -v ccache >/dev/null 2>&1; then
        export CCACHE_BASEDIR="$(pwd)"
        export CCACHE_DIR="${CCACHE_DIR:-$HOME/.cache/ccache}"
        export CCACHE_MAXSIZE="${CCACHE_MAXSIZE:-20G}"
        export CCACHE_COMPILERCHECK=content
        CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
    fi

    if [ ! -d "$BUILD_DIR" ] || [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
        echo "Configuring CMake..."
        mkdir -p "$BUILD_DIR"
        cmake -S mlx -B "$BUILD_DIR" $CMAKE_ARGS \
            -DMLX_BUILD_PYTHON_BINDINGS=ON \
            -DMLX_PYTHON_BINDINGS_OUTPUT_DIRECTORY="$(pwd)/mlx/python/mlx"
    fi

    echo "Building mlx Python extension..."
    cmake --build "$BUILD_DIR" --target core -j$(nproc)

    BUILT_SO="${BUILD_DIR}/${SO_FILENAME}"

    if [ ! -f "$BUILT_SO" ]; then
        BUILT_SO=$(find "$BUILD_DIR" -maxdepth 1 -name "core.cpython*.so" -type f | head -1)
        if [ -z "$BUILT_SO" ]; then
            echo "Error: Could not find built .so file in $BUILD_DIR"
            echo "Looking for: ${SO_FILENAME}"
            ls -la "$BUILD_DIR/"
            exit 1
        fi
    fi

    echo "Found built extension: $BUILT_SO"

    mkdir -p "mlx/python/mlx"

    cp -v "$BUILT_SO" "mlx/python/mlx/${SO_FILENAME}"

    if command -v ccache >/dev/null 2>&1; then
        echo ""
        ccache -s
    fi

    echo ""
    echo "Build complete! The extension is available at: mlx/python/mlx/${SO_FILENAME}"
}

cmd_build_wheel() {
    echo "Building wheel..."

    source virtual-env/bin/activate

    if [ "${OPENCODE:-0}" = "1" ]; then
        CMAKE_VERBOSE_MAKEFILE=OFF
        PIP_VERBOSE_FLAG=""
    else
        CMAKE_VERBOSE_MAKEFILE=ON
        PIP_VERBOSE_FLAG="-v"
    fi

    CMAKE_ARGS="-DMLX_BUILD_VULKAN=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_VERBOSE_MAKEFILE=${CMAKE_VERBOSE_MAKEFILE} -DMLX_USE_CCACHE=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON"

    if command -v ccache >/dev/null 2>&1; then
        export CCACHE_BASEDIR="$(pwd)"
        export CCACHE_DIR="${CCACHE_DIR:-$HOME/.cache/ccache}"
        export CCACHE_MAXSIZE="${CCACHE_MAXSIZE:-20G}"
        export CCACHE_COMPILERCHECK=content
        CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
    fi

    rm -rf wheelhouse
    mkdir -p wheelhouse
    CMAKE_ARGS="$CMAKE_ARGS" uv build --wheel --python 3.13 --out-dir wheelhouse .
    ln -sf wheelhouse/*.whl mlx-0.31.1-cp313-cp313-linux_x86_64.whl

    if command -v ccache >/dev/null 2>&1; then
        ccache -s
    fi

    echo ""
    echo "Wheel built successfully! Check wheelhouse/ directory"
}

cmd_benchmark() {
    local quant="${1:-bf16}"
    
    echo "Running Qwen3 benchmark with quantization: $quant"
    
    # Disable OpenMPI ROCm accelerator to prevent segfault on exit
    export OMPI_MCA_accelerator=^rocm
    
    source virtual-env/bin/activate
    mlx_lm.benchmark --model mlx-community/Qwen3-0.6B-$quant -p 4096 -g 128
}

cmd_profile() {
    local model_size="${1:-0.6b}"
    
    case "$model_size" in
        0.6b|0.6B)
            MODEL="mlx-community/qwen3-0.6b-bf16"
            ;;
        2b|2B)
            MODEL="mlx-community/Qwen3.5-2B-bf16"
            ;;
        *)
            echo "Unknown model size: $model_size" >&2
            echo "Supported sizes: 0.6b, 2b" >&2
            exit 1
            ;;
    esac
    
    echo "Profiling Qwen3 model: $MODEL"
    echo ""
    echo "Environment variables you can set:"
    echo "  MLX_VULKAN_PROFILE_MAX_TOKENS=N     Number of tokens to generate (default: 5)"
    echo "  MLX_VULKAN_PROFILE_SYNC_CHECKPOINTS=1  Synchronize after checkpoints"
    echo "  MLX_VULKAN_PROFILE_CAPTURE_SYNC_TRACE=0  Disable sync trace capture"
    echo "  MLX_VULKAN_PROFILE_ECHO_SYNC_TRACE=1  Echo sync trace to stderr"
    echo "  MLX_VULKAN_DEFERRED_SUBMISSION=0   Disable deferred submission"
    echo ""
    
    source virtual-env/bin/activate
    python scripts/profile_qwen3_vulkan.py --model "$MODEL"
}

# Main dispatch
if [ $# -eq 0 ]; then
    show_help
    exit 1
fi

COMMAND="$1"
shift

case "$COMMAND" in
    init-venv)
        cmd_init_venv
        ;;
    build)
        cmd_build
        ;;
    build-wheel)
        cmd_build_wheel
        ;;
    benchmark)
        cmd_benchmark "${1:-}"
        ;;
    profile)
        cmd_profile "${1:-}"
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "Unknown command: $COMMAND" >&2
        show_help
        exit 1
        ;;
esac
