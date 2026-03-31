#!/usr/bin/env python3
# Copyright © 2024 Apple Inc.

"""
Example demonstrating Vulkan decode batching API.

This shows how to wrap model forward passes with begin_decode_region/end_decode_region
to reduce GPU submissions during inference.

Usage:
    MLX_VULKAN_DECODE_BATCH=1 python example_decode_batching.py

For tracing/decoding metrics:
    MLX_VULKAN_DECODE_BATCH=1 MLX_VULKAN_TRACE_BATCH=1 python example_decode_batching.py
"""

import os
import sys

# Check if Vulkan decode batching is enabled
decode_batch_enabled = os.environ.get("MLX_VULKAN_DECODE_BATCH", "0") in (
    "1",
    "true",
    "True",
)

if decode_batch_enabled:
    print("Decode batching enabled via MLX_VULKAN_DECODE_BATCH")

    try:
        import mlx.core.vulkan as vulkan

        if hasattr(vulkan, "begin_decode_region"):
            print("Vulkan decode region API is available")
            print(f"  decode_batch_enabled() = {vulkan.decode_batch_enabled()}")
        else:
            print("Warning: Vulkan decode region API not found")
            decode_batch_enabled = False
    except ImportError:
        print("Warning: mlx.core.vulkan not available")
        decode_batch_enabled = False
else:
    print("Decode batching disabled (set MLX_VULKAN_DECODE_BATCH=1 to enable)")


def example_model_forward_with_batching(model, input_tokens, cache=None, stream=None):
    """
    Example showing how to wrap a model forward pass with decode batching.

    In a real inference loop, you would call:
        vulkan.begin_decode_region(stream)
        logits = model(input_tokens, cache=cache)
        vulkan.end_decode_region(stream, "decode-token")

    Args:
        model: The model to run
        input_tokens: Input token tensor
        cache: KV cache (optional)
        stream: MLX stream (optional, uses default if not provided)

    Returns:
        Model output logits
    """
    import mlx.core as mx

    if not decode_batch_enabled:
        return model(input_tokens, cache=cache)

    if stream is None:
        stream = mx.new_stream(mx.default_device())

    vulkan.begin_decode_region(stream)
    logits = model(input_tokens, cache=cache)
    vulkan.end_decode_region(stream, "single-token")

    return logits


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test-api":
        # Simple API test without model
        import mlx.core as mx

        if decode_batch_enabled:
            print("\nTesting decode region API...")
            stream = mx.new_stream(mx.default_device())

            # Create test arrays
            a = mx.ones((1024, 1024))
            b = mx.ones((1024, 1024))

            # Wrap computation with decode region
            vulkan.begin_decode_region(stream)
            c = mx.add(a, b)  # Simple operation
            vulkan.end_decode_region(stream, "test-operation")

            mx.synchronize()
            print("Decode region API test passed!")
        else:
            print("Skipping API test (decode batching not enabled)")
    else:
        print("Usage: python example_decode_batching.py --test-api")
        print("\nTo use with mlx_lm, apply the patch in mlx_lm_decode_batching.patch")
        print("\nExample patch application:")
        print("  cd $(pip show mlx-lm | grep Location | cut -d' ' -f2)")
        print("  patch -p1 < /path/to/mlx_lm_decode_batching.patch")
