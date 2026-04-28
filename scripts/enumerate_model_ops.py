#!/usr/bin/env python3
"""
Enumerate model layers and operations, showing operation names and shapes.

Usage:
    python scripts/enumerate_model_ops.py
    python scripts/enumerate_model_ops.py --model mlx-community/qwen3-0.6b-bf16
    python scripts/enumerate_model_ops.py --model mlx-community/Qwen3.5-2B-bf16 --prompt "Hello"

The script loads a model, runs a forward pass, and captures:
- Model layers and their structure
- Operations within each layer
- Input/output shapes for each operation
"""

import sys
import argparse
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

try:
    import mlx.core as mx
except ImportError:
    print("Error: mlx not installed.")
    sys.exit(1)

try:
    from mlx_lm import load
    from mlx_lm.models import cache as cache_utils
except ImportError:
    print("Error: mlx_lm not installed. Install with: pip install mlx-lm")
    sys.exit(1)


class OpInfo:
    """Stores information about an operation."""

    def __init__(self, name: str, op_type: str):
        self.name = name
        self.op_type = op_type
        self.inputs: List[mx.array] = []
        self.output: Optional[Any] = None
        self.shape_history: List[Tuple] = []

    def capture_input_shapes(self, args: Tuple, kwargs: Dict):
        """Capture shapes from input arguments."""
        shapes = []
        for arg in args:
            if isinstance(arg, mx.array):
                shapes.append(arg.shape)
            elif isinstance(arg, (tuple, list)):
                sub_shapes = []
                for item in arg:
                    if isinstance(item, mx.array):
                        sub_shapes.append(item.shape)
                    else:
                        sub_shapes.append(type(item).__name__)
                shapes.append(sub_shapes)
            else:
                shapes.append(type(arg).__name__)
        self.inputs = list(args)
        self.shape_history.append(("input", shapes))

    def capture_output_shapes(self, output: Any):
        """Capture shapes from output."""
        self.output = output
        if isinstance(output, mx.array):
            self.shape_history.append(("output", output.shape))
        elif isinstance(output, (tuple, list)):
            shapes = []
            for item in output:
                if isinstance(item, mx.array):
                    shapes.append(item.shape)
                else:
                    shapes.append(type(item).__name__)
            self.shape_history.append(("output", shapes))
        else:
            self.shape_history.append(("output", type(output).__name__))

    def format_shapes(self, shapes) -> str:
        """Format shapes for display."""
        if isinstance(shapes, str):
            return shapes
        if isinstance(shapes, tuple):
            return str(shapes)
        if isinstance(shapes, list):
            parts = []
            for s in shapes:
                if isinstance(s, tuple):
                    parts.append(str(s))
                elif isinstance(s, list):
                    parts.append(f"[{', '.join(str(x) for x in s)}]")
                else:
                    parts.append(str(s))
            return f"[{', '.join(parts)}]"
        return str(shapes)


class OpCapture:
    """Wraps a callable to capture its operation information."""

    def __init__(self, wrapped, name: str, layer_name: str = ""):
        self.wrapped = wrapped
        self.name = name
        self.layer_name = layer_name
        self.op_info = OpInfo(name, type(wrapped).__name__)
        self.call_count = 0

    def __call__(self, *args, **kwargs):
        self.call_count += 1
        self.op_info.capture_input_shapes(args, kwargs)
        result = self.wrapped(*args, **kwargs)
        self.op_info.capture_output_shapes(result)
        return result

    def __getattr__(self, name):
        return getattr(self.wrapped, name)


class LayerCapture:
    """Installs capture wrappers around each layer and its operations."""

    # Standard transformer layer operations to capture
    DEFAULT_LAYER_OPS = (
        "input_layernorm",
        "self_attn",
        "post_attention_layernorm",
        "mlp",
    )

    # Model-level operations to capture
    DEFAULT_MODEL_OPS = (
        "embed_tokens",
        "norm",
        "lm_head",
    )

    def __init__(
        self,
        model,
        capture_layer_ops: Tuple[str, ...] = DEFAULT_LAYER_OPS,
        capture_model_ops: Tuple[str, ...] = DEFAULT_MODEL_OPS,
    ):
        self.model = model
        self.captures: Dict[str, OpCapture] = {}
        self.layer_captures: Dict[int, Dict[str, OpCapture]] = defaultdict(dict)
        self.capture_layer_ops = capture_layer_ops
        self.capture_model_ops = capture_model_ops
        self._original_layers = None
        self._patched_attrs: List[Tuple[Any, str, Any]] = []
        self._install()

    def _install(self):
        """Install capture wrappers on model layers and operations."""
        layers = getattr(self.model, "layers", None)
        if layers is not None:
            self._original_layers = list(layers)
            for idx, layer in enumerate(self._original_layers):
                layer_name = f"layer[{idx}]"
                for attr_name in self.capture_layer_ops:
                    submodule = getattr(layer, attr_name, None)
                    if submodule is not None:
                        op_name = f"{layer_name}.{attr_name}"
                        capture = OpCapture(submodule, op_name, layer_name)
                        self.captures[op_name] = capture
                        self.layer_captures[idx][attr_name] = capture
                        setattr(layer, attr_name, capture)

        # Capture model-level operations
        for attr_name in self.capture_model_ops:
            submodule = getattr(self.model, attr_name, None)
            if submodule is not None:
                op_name = f"model.{attr_name}"
                capture = OpCapture(submodule, op_name, "model")
                self.captures[op_name] = capture
                setattr(self.model, attr_name, capture)
                self._patched_attrs.append((self.model, attr_name, submodule))

    def uninstall(self):
        """Remove capture wrappers and restore original operations."""
        # Restore layer operations
        layers = getattr(self.model, "layers", None)
        if layers is not None and self._original_layers is not None:
            for idx, layer in enumerate(self._original_layers):
                for attr_name in self.capture_layer_ops:
                    capture = self.layer_captures[idx].get(attr_name)
                    if capture is not None:
                        setattr(layer, attr_name, capture.wrapped)

        # Restore model-level operations
        for owner, attr_name, original in self._patched_attrs:
            setattr(owner, attr_name, original)

    def get_layer_names(self) -> List[str]:
        """Get list of layer names in order."""
        layers = getattr(self.model, "layers", None)
        if layers is None:
            return []
        return [f"layer[{i}]" for i in range(len(layers))]

    def get_captures_for_layer(self, layer_idx: int) -> Dict[str, OpCapture]:
        """Get all operation captures for a specific layer."""
        return self.layer_captures.get(layer_idx, {})

    def get_model_captures(self) -> Dict[str, OpCapture]:
        """Get all model-level operation captures."""
        return {name: cap for name, cap in self.captures.items() if name.startswith("model.")}


def format_shape(shape) -> str:
    """Format a shape tuple for display."""
    if isinstance(shape, tuple):
        return str(shape)
    if isinstance(shape, list):
        parts = []
        for s in shape:
            if isinstance(s, tuple):
                parts.append(str(s))
            else:
                parts.append(str(s))
        return f"[{', '.join(parts)}]"
    return str(shape)


def print_model_structure(model):
    """Print the overall model structure."""
    print(f"Model: {type(model).__name__}")

    # Print layer count
    layers = getattr(model, "layers", None)
    if layers is not None:
        print(f"Layers: {len(layers)}")
    print()


def print_compact_ops(capture: LayerCapture):
    """Print all operations in compact format (one op per line)."""
    all_ops: List[Tuple[str, OpCapture]] = []

    # Model-level ops
    for name, op_capture in sorted(capture.get_model_captures().items()):
        all_ops.append((name, op_capture))

    # Layer ops
    layers = getattr(capture.model, "layers", None)
    if layers is not None:
        for idx in range(len(layers)):
            layer_captures = capture.get_captures_for_layer(idx)
            for op_name, op_capture in sorted(layer_captures.items()):
                full_name = f"layer[{idx}].{op_name}"
                all_ops.append((full_name, op_capture))

    for name, op_capture in all_ops:
        info = op_capture.op_info
        # Get first input and output shapes
        in_shape = "-"
        out_shape = "-"
        if info.shape_history:
            for direction, shapes in info.shape_history:
                if direction == "input":
                    in_shape = info.format_shapes(shapes)
                elif direction == "output":
                    out_shape = info.format_shapes(shapes)
        print(f"{name:<45} {info.op_type:<25} in={in_shape:<25} out={out_shape}")


def print_compact_header():
    """Print header for compact output."""
    print(f"{'Operation':<45} {'Type':<25} {'Input':<25} {'Output'}")
    print("-" * 120)


def enumerate_model_ops(
    model_name: str,
    prompt: str = "Hello, world!",
) -> LayerCapture:
    """
    Load a model and enumerate all layers and operations.

    Args:
        model_name: HuggingFace model repository name
        prompt: Input prompt for forward pass

    Returns:
        LayerCapture object with all captured operation information
    """
    print(f"Model: {model_name}")
    print(f"Device: {mx.default_device()}")

    # Load model
    model, tokenizer = load(model_name)

    # Print model structure
    print_model_structure(model)

    # Install capture
    capture = LayerCapture(model)

    try:
        # Tokenize
        if hasattr(tokenizer, "apply_chat_template"):
            input_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="mlx",
            )
        else:
            input_ids = tokenizer.encode(prompt, return_tensors="mlx")

        input_len = input_ids.shape[1] if len(input_ids.shape) > 1 else input_ids.shape[0]
        print(f"Input: {input_ids.shape} ({input_len} tokens)")
        print()

        # Run forward pass
        cache = cache_utils.make_prompt_cache(model)
        output = model(input_ids, cache=cache)

        # Ensure computation completes
        if isinstance(output, tuple):
            mx.eval(output[0])
        else:
            mx.eval(output)

    finally:
        capture.uninstall()

    # Print compact results
    print_compact_header()
    print_compact_ops(capture)
    print()

    return capture


def main():
    parser = argparse.ArgumentParser(
        description="Enumerate model layers and operations (compact format)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/qwen3-0.6b-bf16",
        help="HuggingFace model repository name (default: mlx-community/qwen3-0.6b-bf16)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, world!",
        help="Input prompt for forward pass (default: 'Hello, world!')",
    )
    args = parser.parse_args()

    try:
        enumerate_model_ops(
            model_name=args.model,
            prompt=args.prompt,
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
