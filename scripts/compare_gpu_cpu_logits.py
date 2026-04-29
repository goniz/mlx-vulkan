#!/usr/bin/env python3
"""
Compare CPU and GPU execution op by op for MLX transformer models.

Purpose: find precision bugs in the Vulkan backend by replaying each
captured model op on GPU with the exact CPU input, so divergence can be
isolated to a specific layer and submodule without running full generation.

Usage:
    python scripts/compare_gpu_cpu_logits.py
    python scripts/compare_gpu_cpu_logits.py --model mlx-community/qwen3-0.6b-bf16
    python scripts/compare_gpu_cpu_logits.py --model mlx-community/Qwen3.5-2B-bf16 --prompt "Hello"
    python scripts/compare_gpu_cpu_logits.py --start-layer 0 --end-layer 0
    python scripts/compare_gpu_cpu_logits.py --ops self_attn
    python scripts/compare_gpu_cpu_logits.py --include-layer-outputs --include-final-logits

Environment variables:
    MLX_COMPARE_DEVICE=gpu     # Device to test against CPU (default: gpu)
"""

import os
import sys
import argparse
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

# Set default device from env before importing mlx
os.environ.setdefault("MLX_COMPARE_DEVICE", "gpu")

import mlx.core as mx
import numpy as np

try:
    from mlx_lm import load
    from mlx_lm.models import cache as cache_utils
except ImportError:
    print("Error: mlx_lm not installed. Install with: pip install mlx-lm")
    sys.exit(1)


DEFAULT_LAYER_OPS = (
    "input_layernorm",
    "self_attn",
    "post_attention_layernorm",
    "mlp",
)

DEFAULT_MODEL_OPS = (
    "embed_tokens",
    "norm",
    "lm_head",
)


def tree_map_arrays(obj: Any, fn) -> Any:
    """Apply fn to all mx.arrays in a nested structure."""
    if isinstance(obj, mx.array):
        return fn(obj)
    if isinstance(obj, tuple):
        return tuple(tree_map_arrays(item, fn) for item in obj)
    if isinstance(obj, list):
        return [tree_map_arrays(item, fn) for item in obj]
    if isinstance(obj, dict):
        return {k: tree_map_arrays(v, fn) for k, v in obj.items()}
    return obj


def move_to_device(obj: Any, device) -> Any:
    """Move all mx.arrays in obj to the given device.

    MLX handles cross-device ops automatically, but we force a copy by
    multiplying by 1 (preserving dtype) so the replay runs entirely on
    the target device.
    """
    old = mx.default_device()
    mx.set_default_device(device)
    try:
        return tree_map_arrays(
            obj,
            lambda x: x * mx.array(1.0, dtype=x.dtype),
        )
    finally:
        mx.set_default_device(old)


def allclose(a: mx.array, b: mx.array, rtol: float = 1e-5, atol: float = 1e-6) -> bool:
    """Check if two arrays are close."""
    if a.shape != b.shape:
        return False
    a_np = np.array(a.astype(mx.float32))
    b_np = np.array(b.astype(mx.float32))
    return np.allclose(a_np, b_np, rtol=rtol, atol=atol)


def max_diff(a: mx.array, b: mx.array) -> float:
    """Compute max absolute difference between two arrays."""
    if a.shape != b.shape:
        return float("inf")
    a_np = np.array(a.astype(mx.float32))
    b_np = np.array(b.astype(mx.float32))
    return float(np.max(np.abs(a_np - b_np)))


def mean_diff(a: mx.array, b: mx.array) -> float:
    """Compute mean absolute difference between two arrays."""
    if a.shape != b.shape:
        return float("inf")
    a_np = np.array(a.astype(mx.float32))
    b_np = np.array(b.astype(mx.float32))
    return float(np.mean(np.abs(a_np - b_np)))


def relative_diff(a: mx.array, b: mx.array) -> float:
    """Compute max relative difference."""
    if a.shape != b.shape:
        return float("inf")
    a_np = np.array(a.astype(mx.float32))
    b_np = np.array(b.astype(mx.float32))
    abs_diff = np.abs(a_np - b_np)
    denom = np.maximum(np.abs(a_np), np.abs(b_np))
    denom = np.where(denom == 0, 1.0, denom)
    return float(np.max(abs_diff / denom))


def tensor_stats(arr: mx.array) -> Dict[str, float]:
    """Get statistics for a tensor."""
    arr_np = np.array(arr.astype(mx.float32))
    return {
        "min": float(np.min(arr_np)),
        "max": float(np.max(arr_np)),
        "mean": float(np.mean(arr_np)),
        "std": float(np.std(arr_np)),
        "median": float(np.median(arr_np)),
    }


class LayerIOCapture:
    """Wraps a callable to capture its inputs and outputs."""

    def __init__(self, wrapped, name: str):
        self.wrapped = wrapped
        self.name = name
        self.inputs: Optional[Tuple] = None   # (args, kwargs)
        self.output: Any = None

    def __call__(self, *args, **kwargs):
        self.inputs = (args, kwargs)
        result = self.wrapped(*args, **kwargs)
        self.output = result
        return result

    def __getattr__(self, name):
        return getattr(self.wrapped, name)


class ModelLayerCapture:
    """Installs capture wrappers around each layer/submodule."""

    def __init__(
        self,
        model,
        selected_layer_ops: Tuple[str, ...],
        capture_layer_outputs: bool,
        capture_model_ops: Tuple[str, ...],
        start_layer: int,
        end_layer: Optional[int],
    ):
        self.model = model
        self.captures: Dict[str, LayerIOCapture] = {}
        self.selected_layer_ops = selected_layer_ops
        self.capture_layer_outputs = capture_layer_outputs
        self.capture_model_ops = capture_model_ops
        self.start_layer = start_layer
        self.end_layer = end_layer
        self._install()

    def _install(self):
        layers = getattr(self.model, "layers", None)
        if layers is None:
            print("Warning: model has no 'layers' attribute.")
            return

        for idx, layer in enumerate(layers):
            if idx < self.start_layer:
                continue
            if self.end_layer is not None and idx > self.end_layer:
                continue

            if self.capture_layer_outputs:
                cap = LayerIOCapture(layer, f"layer[{idx}]")
                self.captures[f"layer[{idx}]"] = cap
                layers[idx] = cap

            for attr_name in self.selected_layer_ops:
                submodule = getattr(layer, attr_name, None)
                if submodule is not None:
                    sub_name = f"layer[{idx}].{attr_name}"
                    sub_cap = LayerIOCapture(submodule, sub_name)
                    self.captures[sub_name] = sub_cap
                    setattr(layer, attr_name, sub_cap)

        for attr_name in self.capture_model_ops:
            submodule = getattr(self.model, attr_name, None)
            if submodule is not None:
                sub_name = f"model.{attr_name}"
                sub_cap = LayerIOCapture(submodule, sub_name)
                self.captures[sub_name] = sub_cap
                setattr(self.model, attr_name, sub_cap)

    def uninstall(self):
        layers = getattr(self.model, "layers", None)
        if layers is None:
            return

        for idx, _layer in enumerate(layers):
            if idx < self.start_layer:
                continue
            if self.end_layer is not None and idx > self.end_layer:
                continue

            cap = self.captures.get(f"layer[{idx}]")
            real_layer = cap.wrapped if cap is not None else layers[idx]
            if cap is not None:
                layers[idx] = real_layer

            for attr_name in self.selected_layer_ops:
                sub_name = f"layer[{idx}].{attr_name}"
                sub_cap = self.captures.get(sub_name)
                if sub_cap is not None:
                    setattr(real_layer, attr_name, sub_cap.wrapped)

        for attr_name in self.capture_model_ops:
            sub_name = f"model.{attr_name}"
            sub_cap = self.captures.get(sub_name)
            if sub_cap is not None:
                setattr(self.model, attr_name, sub_cap.wrapped)

    def get_capture(self, name: str) -> Optional[LayerIOCapture]:
        return self.captures.get(name)


def compare_tensors(name: str, gpu_val: Any, cpu_val: Any) -> Optional[Dict[str, Any]]:
    """Compare two tensors and return diff stats."""
    if not isinstance(gpu_val, mx.array) or not isinstance(cpu_val, mx.array):
        return None

    if gpu_val.shape != cpu_val.shape:
        return {
            "name": name,
            "shape_gpu": gpu_val.shape,
            "shape_cpu": cpu_val.shape,
            "match": False,
            "error": "Shape mismatch",
        }

    stats = {
        "name": name,
        "shape": gpu_val.shape,
        "dtype": str(gpu_val.dtype),
        "max_diff": max_diff(gpu_val, cpu_val),
        "mean_diff": mean_diff(gpu_val, cpu_val),
        "relative_diff": relative_diff(gpu_val, cpu_val),
        "match_strict": allclose(gpu_val, cpu_val, rtol=1e-5, atol=1e-6),
        "match_loose": allclose(gpu_val, cpu_val, rtol=1e-3, atol=1e-4),
    }

    if not stats["match_loose"]:
        stats["gpu_stats"] = tensor_stats(gpu_val)
        stats["cpu_stats"] = tensor_stats(cpu_val)

    return stats


def compare_outputs(name: str, gpu_out: Any, cpu_out: Any) -> List[Dict[str, Any]]:
    """Recursively compare outputs, which may be tuples/lists of tensors."""
    results = []

    if isinstance(gpu_out, mx.array) and isinstance(cpu_out, mx.array):
        r = compare_tensors(name, gpu_out, cpu_out)
        if r is not None:
            results.append(r)
        return results

    if isinstance(gpu_out, (tuple, list)) and isinstance(cpu_out, (tuple, list)):
        if len(gpu_out) != len(cpu_out):
            results.append({
                "name": name,
                "match": False,
                "error": f"Length mismatch: gpu={len(gpu_out)} cpu={len(cpu_out)}",
            })
            return results
        for i, (g, c) in enumerate(zip(gpu_out, cpu_out)):
            results.extend(compare_outputs(f"{name}[{i}]", g, c))
        return results

    if isinstance(gpu_out, dict) and isinstance(cpu_out, dict):
        for k in gpu_out:
            if k not in cpu_out:
                results.append({"name": f"{name}.{k}", "match": False, "error": "Missing in CPU"})
                continue
            results.extend(compare_outputs(f"{name}.{k}", gpu_out[k], cpu_out[k]))
        return results

    return results


def replay_layer(gpu_layer, cpu_inputs: Tuple, device) -> Any:
    """Replay a layer on GPU with CPU-captured inputs."""
    args, kwargs = cpu_inputs
    gpu_args = move_to_device(args, device)
    gpu_kwargs = move_to_device(kwargs, device)
    mx.set_default_device(device)
    return gpu_layer(*gpu_args, **gpu_kwargs)


def capture_sort_key(name: str) -> Tuple[int, int, str]:
    """Sort captures in model execution order."""
    if name == "model.embed_tokens":
        return (0, -1, name)
    if name.startswith("layer["):
        layer_idx = int(name.split("[")[1].split("]")[0])
        if name == f"layer[{layer_idx}]":
            return (2, layer_idx, name)

        submodule = name.split(".", 1)[1]
        submodule_order = {
            "input_layernorm": 0,
            "self_attn": 1,
            "post_attention_layernorm": 2,
            "mlp": 3,
        }
        return (1, layer_idx * 10 + submodule_order.get(submodule, 9), name)
    if name == "model.norm":
        return (3, 0, name)
    if name == "model.lm_head":
        return (4, 0, name)
    return (5, 0, name)


def layer_index_from_name(name: str) -> Optional[int]:
    if not name.startswith("layer["):
        return None
    return int(name.split("[")[1].split("]")[0])


def op_name_from_capture(name: str) -> str:
    if name.startswith("model."):
        return name.split(".", 1)[1]
    if "." in name:
        return name.split(".", 1)[1]
    return "layer_output"


def should_include_capture(name: str, selected_ops: set[str], include_layer_outputs: bool) -> bool:
    op_name = op_name_from_capture(name)
    if op_name == "layer_output":
        return include_layer_outputs
    return op_name in selected_ops


def parse_ops_arg(ops_arg: Optional[str]) -> Tuple[str, ...]:
    if not ops_arg:
        return DEFAULT_LAYER_OPS + DEFAULT_MODEL_OPS
    ops = tuple(part.strip() for part in ops_arg.split(",") if part.strip())
    valid = set(DEFAULT_LAYER_OPS + DEFAULT_MODEL_OPS + ("layer_output",))
    invalid = [op for op in ops if op not in valid]
    if invalid:
        raise ValueError(f"Unknown ops: {', '.join(invalid)}")
    return ops


def run_layer_by_layer_comparison(
    model_name: str,
    prompt: str,
    test_device_name: str = "gpu",
    selected_ops: Tuple[str, ...] = DEFAULT_LAYER_OPS + DEFAULT_MODEL_OPS,
    start_layer: int = 0,
    end_layer: Optional[int] = None,
    include_layer_outputs: bool = False,
    include_final_logits: bool = False,
    stop_after_fail: bool = False,
    verbose: bool = True,
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Run inference layer by layer:
      1. Run one CPU prefill pass while capturing selected op inputs/outputs.
      2. Replay each captured op on GPU with the exact CPU input.
      3. Optionally compare whole-layer outputs and final logits.

    Returns (all_match, diff_records).
    """
    if verbose:
        print("=" * 100)
        print(f"OP-BY-OP COMPARISON: {model_name}")
        print("=" * 100)
        print(f"Test device: {test_device_name}")
        print(f"Reference device: cpu")
        print(f"Selected ops: {', '.join(selected_ops)}")
        if end_layer is None:
            print(f"Layer range: {start_layer}-end")
        else:
            print(f"Layer range: {start_layer}-{end_layer}")
        print()

    test_device = getattr(mx, test_device_name) if hasattr(mx, test_device_name) else mx.gpu
    cpu_device = mx.cpu

    # ------------------------------------------------------------------
    # 1. Load model on CPU and run forward, capturing layer I/O
    # ------------------------------------------------------------------
    if verbose:
        print("Loading model on CPU (reference)...")
    mx.set_default_device(cpu_device)
    model_cpu, tokenizer = load(model_name)
    if verbose:
        num_layers = len(getattr(model_cpu, "layers", []))
        print(f"  Model loaded: {num_layers} layers")

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
    if verbose:
        print(f"  Input length: {input_len} tokens")
        print()

    if verbose:
        print("Running CPU forward pass with layer capture...")
    cache_cpu = cache_utils.make_prompt_cache(model_cpu)
    layer_ops = tuple(op for op in selected_ops if op in DEFAULT_LAYER_OPS)
    model_ops = tuple(op for op in selected_ops if op in DEFAULT_MODEL_OPS)
    capture_cpu = ModelLayerCapture(
        model_cpu,
        selected_layer_ops=layer_ops,
        capture_layer_outputs=include_layer_outputs,
        capture_model_ops=model_ops,
        start_layer=start_layer,
        end_layer=end_layer,
    )
    try:
        output_cpu = model_cpu(input_ids, cache=cache_cpu)
        if isinstance(output_cpu, tuple):
            logits_cpu = output_cpu[0]
        else:
            logits_cpu = output_cpu
        if logits_cpu.ndim > 1:
            logits_cpu = logits_cpu[:, -1, :]
        mx.eval(logits_cpu)
    finally:
        capture_cpu.uninstall()

    if verbose:
        print("  CPU forward pass complete.")
        print()

    # ------------------------------------------------------------------
    # 2. Free CPU model, load same model on GPU
    # ------------------------------------------------------------------
    if verbose:
        print("Freeing CPU model and loading on GPU...")
    del model_cpu
    del cache_cpu
    mx.eval(mx.array(0))

    mx.set_default_device(test_device)
    model_gpu, _ = load(model_name)
    if verbose:
        print("  GPU model loaded.")
        print()

    # ------------------------------------------------------------------
    # 3. Op-by-op replay on GPU
    # ------------------------------------------------------------------
    diff_records: List[Dict[str, Any]] = []
    selected_op_names = set(selected_ops)
    all_names = [
        name
        for name in sorted(capture_cpu.captures.keys(), key=capture_sort_key)
        if should_include_capture(name, selected_op_names, include_layer_outputs)
    ]
    first_fail_name = None

    if verbose:
        print("Replaying captured ops on GPU (inference order):")
        print(f"{'Op':<50} {'Max Diff':<14} {'Mean Diff':<14} {'Rel Diff':<14} {'Status':<8}")
        print("-" * 100)

    for name in all_names:
        cap = capture_cpu.get_capture(name)
        if cap is None or cap.inputs is None:
            continue

        gpu_layer = getattr(model_gpu, name.split(".")[0], None)
        if name.startswith("layer["):
            idx = int(name.split("[")[1].split("]")[0])
            gpu_layer = model_gpu.layers[idx]
            if "." in name:
                subname = name.split(".", 1)[1]
                gpu_layer = getattr(gpu_layer, subname, None)
        elif name.startswith("model."):
            gpu_layer = getattr(model_gpu, name.split(".", 1)[1], None)

        if gpu_layer is None:
            diff_records.append({
                "name": name,
                "match": False,
                "error": "Could not find corresponding GPU layer",
            })
            if verbose:
                print(f"{name:<50} {'N/A':<14} {'N/A':<14} {'N/A':<14} {'SKIP':<8}")
            continue

        try:
            gpu_output = replay_layer(gpu_layer, cap.inputs, test_device)
            mx.eval(gpu_output)
            diffs = compare_outputs(name, gpu_output, cap.output)
            diff_records.extend(diffs)

            # Determine status for this layer
            any_fail = any(not d.get("match_loose", d.get("match", True)) for d in diffs)
            if diffs:
                worst = max(diffs, key=lambda d: d.get("max_diff", 0.0))
                status = "FAIL" if any_fail else "PASS"
                if any_fail and first_fail_name is None:
                    first_fail_name = name
                if verbose:
                    print(
                        f"{name:<50} {worst['max_diff']:<14.6e} {worst['mean_diff']:<14.6e} "
                        f"{worst['relative_diff']:<14.6e} {status:<8}"
                    )
                if any_fail and stop_after_fail:
                    break
            else:
                if verbose:
                    print(f"{name:<50} {'N/A':<14} {'N/A':<14} {'N/A':<14} {'PASS':<8}")

        except Exception as e:
            diff_records.append({
                "name": name,
                "match": False,
                "error": str(e),
            })
            if verbose:
                print(f"{name:<50} {'ERR':<14} {'ERR':<14} {'ERR':<14} {'ERROR':<8}")
            if first_fail_name is None:
                first_fail_name = name
            if stop_after_fail:
                break

    if verbose:
        print()

    cache_gpu = None
    if include_final_logits:
        # Optional end-to-end check after op-level replay.
        if verbose:
            print("Running full-model GPU forward pass for final logits...")
        cache_gpu = cache_utils.make_prompt_cache(model_gpu)
        output_gpu = model_gpu(input_ids, cache=cache_gpu)
        if isinstance(output_gpu, tuple):
            logits_gpu = output_gpu[0]
        else:
            logits_gpu = output_gpu
        if logits_gpu.ndim > 1:
            logits_gpu = logits_gpu[:, -1, :]
        mx.eval(logits_gpu)

        final_diff = compare_tensors("final_logits", logits_gpu, logits_cpu)
        if final_diff is not None:
            diff_records.append(final_diff)
        if verbose:
            print(f"  Final logits max diff: {final_diff['max_diff']:.6e}")
            print()

    # Cleanup
    del model_gpu
    del cache_gpu
    mx.eval(mx.array(0))

    all_match = all(r.get("match_loose", r.get("match", True)) for r in diff_records)
    return all_match, diff_records, first_fail_name


def print_comparison_report(diff_records: List[Dict[str, Any]], first_fail_name: Optional[str], verbose: bool = True):
    """Print a formatted comparison report."""
    print("=" * 100)
    print("COMPARISON REPORT")
    print("=" * 100)
    print()

    tensor_records = [r for r in diff_records if "max_diff" in r]
    error_records = [r for r in diff_records if "error" in r]

    if error_records:
        print("ERRORS:")
        print("-" * 100)
        for r in error_records:
            print(f"  {r['name']}: {r['error']}")
        print()

    if tensor_records:
        print(f"{'Name':<50} {'Max Diff':<14} {'Mean Diff':<14} {'Rel Diff':<14} {'Match':<8}")
        print("-" * 100)

        for r in sorted(tensor_records, key=lambda x: x["max_diff"], reverse=True):
            match_str = "PASS" if r.get("match_strict") else ("OK" if r.get("match_loose") else "FAIL")
            print(
                f"{r['name']:<50} {r['max_diff']:<14.6e} {r['mean_diff']:<14.6e} "
                f"{r['relative_diff']:<14.6e} {match_str:<8}"
            )

            if verbose and not r.get("match_loose") and "gpu_stats" in r:
                print(f"    GPU stats: min={r['gpu_stats']['min']:.6e} max={r['gpu_stats']['max']:.6e} "
                      f"mean={r['gpu_stats']['mean']:.6e} std={r['gpu_stats']['std']:.6e}")
                print(f"    CPU stats: min={r['cpu_stats']['min']:.6e} max={r['cpu_stats']['max']:.6e} "
                      f"mean={r['cpu_stats']['mean']:.6e} std={r['cpu_stats']['std']:.6e}")

        print()

        if first_fail_name:
            print(f"FIRST FAILING OP: {first_fail_name}")
            print()
        else:
            print("All selected ops match within tolerance.")
            print()

    total = len(diff_records)
    passed = sum(1 for r in diff_records if r.get("match_strict", r.get("match", True)))
    ok = sum(1 for r in diff_records if not r.get("match_strict", False) and r.get("match_loose", False))
    failed = total - passed - ok

    print("SUMMARY")
    print("-" * 100)
    print(f"Total comparisons: {total}")
    print(f"Passed (strict):   {passed}")
    print(f"Passed (loose):    {ok}")
    print(f"Failed:            {failed}")
    print()

    if failed > 0:
        print("RESULT: FAIL - deviations detected between GPU and CPU outputs")
    else:
        print("RESULT: PASS - GPU and CPU outputs match within tolerance")
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Compare GPU (Vulkan) and CPU inference op by op"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/qwen3-0.6b-bf16",
        help="Model to test (default: mlx-community/qwen3-0.6b-bf16)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, how are you?",
        help="Prompt for forward pass",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=os.environ.get("MLX_COMPARE_DEVICE", "gpu"),
        help="Test device to compare against CPU (default: gpu)",
    )
    parser.add_argument(
        "--ops",
        type=str,
        default=None,
        help="Comma-separated ops to compare. Defaults to embed_tokens,input_layernorm,self_attn,post_attention_layernorm,mlp,norm,lm_head",
    )
    parser.add_argument(
        "--start-layer",
        type=int,
        default=0,
        help="First transformer layer to inspect (default: 0)",
    )
    parser.add_argument(
        "--end-layer",
        type=int,
        default=None,
        help="Last transformer layer to inspect (default: last)",
    )
    parser.add_argument(
        "--include-layer-outputs",
        action="store_true",
        help="Also compare whole layer outputs in addition to submodule ops",
    )
    parser.add_argument(
        "--include-final-logits",
        action="store_true",
        help="Also run a full GPU forward pass and compare final logits",
    )
    parser.add_argument(
        "--stop-after-fail",
        action="store_true",
        help="Stop replay as soon as the first failing op is found",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    verbose = not args.quiet

    try:
        selected_ops = parse_ops_arg(args.ops)
        all_match, diff_records, first_fail = run_layer_by_layer_comparison(
            model_name=args.model,
            prompt=args.prompt,
            test_device_name=args.device,
            selected_ops=selected_ops,
            start_layer=args.start_layer,
            end_layer=args.end_layer,
            include_layer_outputs=args.include_layer_outputs,
            include_final_logits=args.include_final_logits,
            stop_after_fail=args.stop_after_fail,
            verbose=verbose,
        )

        if verbose:
            print()
            print_comparison_report(diff_records, first_fail, verbose=True)

        sys.exit(0 if all_match else 1)

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
