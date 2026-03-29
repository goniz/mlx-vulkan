#!/usr/bin/env python3
"""
Profile Qwen3 model inference with separate prefill and decode timing.
Uses MLX tracing/monkeypatching to show time per layer/op.

Usage:
    python mlx/backend/vulkan/profile_qwen3_vulkan.py
    python mlx/backend/vulkan/profile_qwen3_vulkan.py --model mlx-community/Qwen3.5-2B-bf16
    python mlx/backend/vulkan/profile_qwen3_vulkan.py --help

Environment variables:
    MLX_VULKAN_DEFERRED_SUBMISSION=1  # Enable deferred submission
    MLX_VULKAN_TRACE_FALLBACKS=1      # Trace CPU fallbacks
"""

import time
import sys
import os
import re
import argparse
import threading
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Tuple, Any, Optional


QWEN3_06B_BF16_GOAL = {
    "model": "mlx-community/qwen3-0.6b-bf16",
    "prefill_tokens_per_sec": 523.32,
    "decode_tokens_per_sec": 123.41,
    "source": "user-supplied benchmark update",
    "reference": "llama.cpp unsloth/Qwen3-0.6B-GGUF:BF16 on Radeon 8060S (Vulkan)",
}


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value not in ("0", "false", "False", "no", "NO", "off", "OFF")


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def should_enable_sync_trace_before_import(argv: List[str]) -> bool:
    if "MLX_VULKAN_TRACE_SYNC" in os.environ:
        return False
    if "--no-capture-sync-trace" in argv:
        return False
    return env_flag("MLX_VULKAN_PROFILE_CAPTURE_SYNC_TRACE", True)


# Enable Vulkan fallback tracing by default for profiling
os.environ.setdefault("MLX_VULKAN_TRACE_FALLBACKS", "1")
if should_enable_sync_trace_before_import(sys.argv[1:]):
    os.environ["MLX_VULKAN_TRACE_SYNC"] = "1"

import mlx.core as mx

try:
    from mlx_lm import load
    from mlx_lm.models import cache as cache_utils
except ImportError:
    print("Error: mlx_lm not installed. Install with: pip install mlx-lm")
    sys.exit(1)

class FallbackAnalyzer:
    """Analyzes Vulkan fallback messages from stderr."""

    def __init__(self):
        self.fallback_counts: Dict[str, int] = defaultdict(int)
        self.fallback_patterns = {
            "RMSNorm": re.compile(r"primitive=RMSNorm"),
            "ScaledDotProductAttention": re.compile(
                r"primitive=ScaledDotProductAttention[^V]"
            ),
            "ScaledDotProductAttentionVJP": re.compile(
                r"primitive=ScaledDotProductAttentionVJP"
            ),
            "Softmax": re.compile(r"primitive=.*Softmax"),
            "Gather": re.compile(r"primitive=Gather"),
            "RoPE": re.compile(r"primitive=RoPE"),
        }

    def analyze_stderr(self, stderr_text: str):
        """Parse stderr output and count fallbacks."""
        for line in stderr_text.split("\n"):
            self.consume_line(line)

    def consume_line(self, line: str):
        if "[vulkan-fallback]" not in line:
            return
        for op_name, pattern in self.fallback_patterns.items():
            if pattern.search(line):
                self.fallback_counts[op_name] += 1
                return
        match = re.search(r"primitive=(\S+)", line)
        if match:
            self.fallback_counts[match.group(1)] += 1

    def get_report(self) -> str:
        """Generate fallback analysis report."""
        lines = []
        lines.append("=" * 100)
        lines.append("VULKAN FALLBACK ANALYSIS")
        lines.append("=" * 100)
        lines.append(
            "\nOperations falling back to CPU (from MLX_VULKAN_TRACE_FALLBACKS output):"
        )
        lines.append("-" * 100)
        lines.append(f"{'Operation':<50} {'Fallback Count':<20}")
        lines.append("-" * 100)

        if not self.fallback_counts:
            lines.append("No fallbacks detected (great! All ops running on Vulkan GPU)")
        else:
            sorted_fallbacks = sorted(
                self.fallback_counts.items(), key=lambda x: x[1], reverse=True
            )
            for op_name, count in sorted_fallbacks:
                lines.append(f"{op_name:<50} {count:<20}")
            total = sum(self.fallback_counts.values())
            lines.append("-" * 100)
            lines.append(f"{'TOTAL FALLBACKS':<50} {total:<20}")

        lines.append("=" * 100)
        return "\n".join(lines)


class SyncTraceAnalyzer:
    """Summarize Vulkan submit/barrier activity by phase."""

    submit_re = re.compile(r"submit begin .* rec_ops=(\d+) .* reason='([^']+)'")
    hazard_boundary_re = re.compile(
        r"hazard boundary action=(submit|barrier)(?: reason=([^ ]+))?"
    )
    barrier_re = re.compile(r"barrier action=([a-zA-Z0-9_-]+)(?: reason=([^ ]+))?")
    hazard_re = re.compile(r"hazard (raw|war|waw) current=")

    def __init__(self):
        self._phase = "startup"
        self._lock = threading.Lock()
        self.submit_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        self.submit_ops: Dict[str, List[int]] = defaultdict(list)
        self.hazard_boundary_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        self.barrier_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        self.hazard_counts: Dict[Tuple[str, str], int] = defaultdict(int)

    def set_phase(self, phase: str):
        with self._lock:
            self._phase = phase

    def _current_phase(self) -> str:
        with self._lock:
            return self._phase

    def consume_line(self, line: str):
        if "[vulkan-trace" not in line:
            return

        phase = self._current_phase()

        submit_match = self.submit_re.search(line)
        if submit_match:
            rec_ops, reason = submit_match.groups()
            self.submit_counts[(phase, reason)] += 1
            self.submit_ops[phase].append(int(rec_ops))
            return

        hazard_boundary_match = self.hazard_boundary_re.search(line)
        if hazard_boundary_match:
            action, _reason = hazard_boundary_match.groups()
            self.hazard_boundary_counts[(phase, action)] += 1
            if action == "barrier":
                self.barrier_counts[(phase, "hazard-overlap")] += 1
            return

        barrier_match = self.barrier_re.search(line)
        if barrier_match:
            action, _reason = barrier_match.groups()
            self.barrier_counts[(phase, action)] += 1
            return

        hazard_match = self.hazard_re.search(line)
        if hazard_match:
            self.hazard_counts[(phase, hazard_match.group(1).upper())] += 1

    def get_report(self) -> str:
        lines = []
        lines.append("=" * 100)
        lines.append("VULKAN SUBMIT / BARRIER REPORT")
        lines.append("=" * 100)

        phases = sorted(
            {
                phase for phase, _ in self.submit_counts.keys()
            }
            | {
                phase for phase, _ in self.hazard_boundary_counts.keys()
            }
            | {
                phase for phase, _ in self.barrier_counts.keys()
            }
            | {
                phase for phase, _ in self.hazard_counts.keys()
            }
        )

        if not phases:
            lines.append("\nNo Vulkan sync trace events captured.")
            lines.append("=" * 100)
            return "\n".join(lines)

        lines.append(
            f"\n{'Phase':<12} {'Submits':<8} {'AvgOps':<8} {'HazSub':<8} {'ExpSync':<8} {'Finalize':<8} {'Thresh':<8} {'HazBar':<8} {'TailBar':<8} {'RAW':<6} {'WAR':<6} {'WAW':<6}"
        )
        lines.append("-" * 100)
        for phase in phases:
            submit_total = sum(
                count for (p, _), count in self.submit_counts.items() if p == phase
            )
            submit_ops = self.submit_ops.get(phase, [])
            avg_ops = (
                sum(submit_ops) / len(submit_ops) if submit_ops else 0.0
            )
            explicit_sync = sum(
                count
                for (p, reason), count in self.submit_counts.items()
                if p == phase and reason.startswith("explicit synchronize")
            )
            lines.append(
                f"{phase:<12} {submit_total:<8} {avg_ops:<8.2f} "
                f"{self.submit_counts[(phase, 'hazard overlap')]:<8} "
                f"{explicit_sync:<8} "
                f"{self.submit_counts[(phase, 'finalize')]:<8} "
                f"{self.submit_counts[(phase, 'threshold reached')]:<8} "
                f"{self.barrier_counts[(phase, 'hazard-overlap')]:<8} "
                f"{self.barrier_counts[(phase, 'recording-tail')]:<8} "
                f"{self.hazard_counts[(phase, 'RAW')]:<6} "
                f"{self.hazard_counts[(phase, 'WAR')]:<6} "
                f"{self.hazard_counts[(phase, 'WAW')]:<6}"
            )

        lines.append("")
        lines.append("HAZARD BOUNDARY DECISIONS")
        lines.append("-" * 100)
        for phase in phases:
            lines.append(
                f"{phase:<12} submit={self.hazard_boundary_counts[(phase, 'submit')]} barrier={self.hazard_boundary_counts[(phase, 'barrier')]}"
            )

        lines.append("")
        lines.append("EXPLICIT SYNC REASONS")
        lines.append("-" * 100)
        for phase in phases:
            phase_reasons = sorted(
                (
                    (reason, count)
                    for (p, reason), count in self.submit_counts.items()
                    if p == phase and reason.startswith("explicit synchronize")
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            if not phase_reasons:
                continue
            lines.append(f"{phase}:")
            for reason, count in phase_reasons[:8]:
                lines.append(f"  {reason:<60} {count}")

        lines.append("=" * 100)
        return "\n".join(lines)


@contextmanager
def capture_stderr_lines(line_handler, echo_vulkan_trace: bool):
    """Capture process stderr, forward selected lines, and parse backend traces."""

    sys.stderr.flush()
    original_fd = os.dup(2)
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, 2)
    os.close(write_fd)

    def forward_line(line: str):
        if line_handler is not None:
            line_handler(line)
        if echo_vulkan_trace or "[vulkan-trace" not in line:
            os.write(original_fd, (line + "\n").encode("utf-8", errors="replace"))

    def pump_stderr():
        pending = ""
        try:
            while True:
                chunk = os.read(read_fd, 4096)
                if not chunk:
                    break
                pending += chunk.decode("utf-8", errors="replace")
                while "\n" in pending:
                    line, pending = pending.split("\n", 1)
                    forward_line(line)
        finally:
            if pending:
                forward_line(pending)

    thread = threading.Thread(target=pump_stderr, daemon=True)
    thread.start()
    try:
        yield
    finally:
        sys.stderr.flush()
        os.dup2(original_fd, 2)
        thread.join()
        os.close(read_fd)
        os.close(original_fd)


class OpTracer:
    """
    Traces MLX operations using high-level timing (monkeypatching disabled to avoid breaking MLX internals).
    Captures phase-level timing for prefill and decode.
    """

    def __init__(self):
        self.phase: str = "unknown"
        self.phase_times: Dict[str, float] = defaultdict(float)
        self.phase_start: Optional[float] = None
        self.layer_times: Dict[Tuple[str, int], float] = defaultdict(float)
        self.layer_counts: Dict[Tuple[str, int], int] = defaultdict(int)
        self.submodule_times: Dict[Tuple[str, str], float] = defaultdict(float)
        self.submodule_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        self.token_times: Dict[str, List[float]] = defaultdict(list)

    @contextmanager
    def time_block(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            key = (self.phase, name)
            self.submodule_times[key] += elapsed
            self.submodule_counts[key] += 1

    def set_phase(self, phase: str):
        """Set current phase (prefill/decode)."""
        # Record previous phase time if switching
        if self.phase_start is not None and self.phase != "unknown":
            elapsed = time.perf_counter() - self.phase_start
            self.phase_times[self.phase] += elapsed

        self.phase = phase
        self.phase_start = time.perf_counter()

    def install(self):
        """Install monkey patches for tracing - DISABLED to avoid breaking MLX internals."""
        # Monkeypatching MLX operations can break internal module structure
        # Instead, we rely on MLX_VULKAN_TRACE_FALLBACKS env var for backend tracing
        pass

    @contextmanager
    def time_layer(self, layer_idx: int):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            key = (self.phase, layer_idx)
            self.layer_times[key] += elapsed
            self.layer_counts[key] += 1

    def record_token_time(self, phase: str, elapsed: float):
        self.token_times[phase].append(elapsed)

    def uninstall(self):
        """Remove monkey patches - no-op since install is disabled."""
        # Finalize last phase timing
        if self.phase_start is not None and self.phase != "unknown":
            elapsed = time.perf_counter() - self.phase_start
            self.phase_times[self.phase] += elapsed

    def get_report(self) -> str:
        """Generate profiling report."""
        lines = []
        lines.append("=" * 100)
        lines.append("PHASE TIMING REPORT")
        lines.append("=" * 100)

        if not self.phase_times:
            lines.append("\nNo phase timing captured.")
        else:
            lines.append(f"\n{'Phase':<30} {'Time (ms)':<15}")
            lines.append("-" * 100)
            for phase, total_time in sorted(self.phase_times.items()):
                lines.append(f"{phase:<30} {total_time * 1000:<15.2f}")

        if self.token_times:
            lines.append("")
            lines.append("PER-TOKEN HOST TIMING")
            lines.append("=" * 100)
            lines.append(
                f"\n{'Phase':<12} {'Count':<8} {'Total (ms)':<15} {'Avg (ms)':<15} {'Min (ms)':<15} {'Max (ms)':<15}"
            )
            lines.append("-" * 100)
            for phase, times in sorted(self.token_times.items()):
                if not times:
                    continue
                lines.append(
                    f"{phase:<12} {len(times):<8} {sum(times) * 1000:<15.2f} {(sum(times) / len(times)) * 1000:<15.2f} {min(times) * 1000:<15.2f} {max(times) * 1000:<15.2f}"
                )

        lines.append("")
        lines.append("PER-LAYER TIMING REPORT")
        lines.append("=" * 100)

        if not self.layer_times:
            lines.append("\nNo per-layer timing captured.")
        else:
            lines.append(
                f"\n{'Phase':<12} {'Layer':<8} {'Calls':<8} {'Total (ms)':<15} {'Avg (ms)':<15}"
            )
            lines.append("-" * 100)
            phase_layer_totals: Dict[str, float] = defaultdict(float)
            for (phase, _), total_time in self.layer_times.items():
                phase_layer_totals[phase] += total_time
            for (phase, layer_idx), total_time in sorted(
                self.layer_times.items(), key=lambda item: (item[0][0], item[0][1])
            ):
                count = self.layer_counts[(phase, layer_idx)]
                avg_time = total_time / count if count else 0.0
                lines.append(
                    f"{phase:<12} {layer_idx:<8} {count:<8} {total_time * 1000:<15.2f} {avg_time * 1000:<15.2f}"
                )
            lines.append("")
            lines.append("PER-PHASE LAYER ENQUEUE BREAKDOWN")
            lines.append("-" * 100)
            lines.append(
                f"{'Phase':<12} {'Layer Sum (ms)':<15} {'Other Host (ms)':<17} {'Phase Total (ms)':<17}"
            )
            lines.append("-" * 100)
            for phase, phase_total in sorted(self.phase_times.items()):
                layer_total = phase_layer_totals.get(phase, 0.0)
                other = max(0.0, phase_total - layer_total)
                lines.append(
                    f"{phase:<12} {layer_total * 1000:<15.2f} {other * 1000:<17.2f} {phase_total * 1000:<17.2f}"
                )
        if self.submodule_times:
            lines.append("")
            lines.append("HOST ENQUEUE BLOCKS")
            lines.append("=" * 100)
            lines.append(
                f"\n{'Phase':<12} {'Block':<36} {'Calls':<8} {'Total (ms)':<15} {'Avg (ms)':<15}"
            )
            lines.append("-" * 100)
            for (phase, name), total_time in sorted(
                self.submodule_times.items(), key=lambda item: (item[0][0], item[0][1])
            ):
                count = self.submodule_counts[(phase, name)]
                avg_time = total_time / count if count else 0.0
                lines.append(
                    f"{phase:<12} {name:<36} {count:<8} {total_time * 1000:<15.2f} {avg_time * 1000:<15.2f}"
                )
            lines.append("")
            lines.append(
                "Note: per-layer timings are lightweight host-side dispatch timings and do not include full GPU completion."
            )

        lines.append("=" * 100)
        return "\n".join(lines)

    def print_report(self):
        """Print profiling report."""
        print(self.get_report())


class ModelWrapper:
    """
    Wraps model for tracing. Uses __call__ interception since direct attribute replacement
    doesn't work with MLX nn.Module property-based attributes.
    """

    def __init__(self, model, tracer: OpTracer):
        self.model = model
        self.tracer = tracer
        self._original_layers = None
        self._patched_attrs = []

    def _patch_attr(self, owner, attr_name: str, label: str):
        try:
            value = getattr(owner, attr_name)
        except AttributeError:
            return
        if value is None or not callable(value):
            return
        setattr(owner, attr_name, NamedBlockWrapper(value, label, self.tracer))
        self._patched_attrs.append((owner, attr_name, value))

    def install(self):
        layers = getattr(self.model, "layers", None)
        if layers is None:
            return
        self._original_layers = list(layers)
        for idx, layer in enumerate(self._original_layers):
            for attr_name in (
                "input_layernorm",
                "self_attn",
                "post_attention_layernorm",
                "mlp",
            ):
                self._patch_attr(layer, attr_name, f"layer[{idx}].{attr_name}")
            layers[idx] = LayerWrapper(layer, idx, self.tracer)
        for attr_name in ("embed_tokens", "norm", "lm_head"):
            self._patch_attr(self.model, attr_name, f"model.{attr_name}")

    def uninstall(self):
        for owner, attr_name, value in reversed(self._patched_attrs):
            setattr(owner, attr_name, value)
        self._patched_attrs.clear()
        if self._original_layers is None:
            return
        layers = getattr(self.model, "layers", None)
        if layers is not None:
            for idx, layer in enumerate(self._original_layers):
                layers[idx] = layer
        self._original_layers = None

    def __call__(self, *args, **kwargs):
        """Forward pass with layer tracing."""
        # For transformer models, we trace at the model level since
        # layer-level wrapping requires modifying internal structure
        return self.model(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.model, name)


class LayerWrapper:
    """Thin proxy that records wall-clock time per transformer layer."""

    def __init__(self, layer, layer_idx: int, tracer: OpTracer):
        self.layer = layer
        self.layer_idx = layer_idx
        self.tracer = tracer

    def __call__(self, *args, **kwargs):
        with self.tracer.time_layer(self.layer_idx):
            return self.layer(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.layer, name)


class NamedBlockWrapper:
    """Wrap a callable model block and record host-side enqueue time."""

    def __init__(self, block, name: str, tracer: OpTracer):
        self.block = block
        self.name = name
        self.tracer = tracer

    def __call__(self, *args, **kwargs):
        with self.tracer.time_block(self.name):
            return self.block(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.block, name)


def trace_model_inference(
    model_name: str = "mlx-community/qwen3-0.6b-bf16",
    prompt: str = "Hello, how are you?",
    max_tokens: int = 5,
    verbose: bool = True,
    sync_checkpoints: bool = False,
    sync_analyzer: Optional[SyncTraceAnalyzer] = None,
) -> Tuple[Any, Any, OpTracer, Dict[str, Any]]:
    """
    Load model and trace inference with separate prefill/decode phases.

    Args:
        model_name: HuggingFace model name
        prompt: Input prompt
        max_tokens: Number of tokens to generate
        verbose: Print progress

    Returns:
        Tuple of (model, tokenizer, tracer)
    """

    if verbose:
        print(f"Loading model: {model_name}")
        print(f"Default device: {mx.default_device()}")
        print(f"GPU available: {mx.is_available(mx.gpu)}")
        print(f"CPU available: {mx.is_available(mx.cpu)}")
        print()

    # Load model
    load_start = time.perf_counter()
    model, tokenizer = load(model_name)
    load_time = (time.perf_counter() - load_start) * 1000

    if verbose:
        print(f"Model loaded in {load_time:.2f} ms")
        print(f"Model has {len(model.layers)} layers")
        print()

    # Create tracer
    tracer = OpTracer()
    tracer.install()

    # Wrap model for layer tracing
    wrapped_model = ModelWrapper(model, tracer)
    wrapped_model.install()

    try:
        # Tokenize
        if verbose:
            print(f"Prompt: '{prompt}'")

        if hasattr(tokenizer, "apply_chat_template"):
            input_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="mlx",
            )
        else:
            input_ids = tokenizer.encode(prompt, return_tensors="mlx")
        input_len = (
            input_ids.shape[1] if len(input_ids.shape) > 1 else input_ids.shape[0]
        )

        if verbose:
            print(f"Input length: {input_len} tokens")
            print()

        # === PREFILL PHASE ===
        if verbose:
            print("=" * 100)
            print("PREFILL PHASE")
            print("=" * 100)

        tracer.set_phase("prefill")
        if sync_analyzer is not None:
            sync_analyzer.set_phase("prefill")
        prefill_start = time.perf_counter()

        with tracer.time_block("make_prompt_cache"):
            prompt_cache = cache_utils.make_prompt_cache(model)
        with tracer.time_block("model_forward"):
            model_output = wrapped_model(input_ids, cache=prompt_cache)

        if isinstance(model_output, tuple):
            logits = model_output[0]
        else:
            logits = model_output

        if logits.ndim > 1:
            logits = logits[:, -1, :]

        if sync_checkpoints:
            with tracer.time_block("sync_logits_ready"):
                mx.eval(logits)

        with tracer.time_block("argmax"):
            next_token = mx.argmax(logits, axis=-1)
        if sync_checkpoints:
            with tracer.time_block("sync_next_token_ready"):
                mx.eval(next_token)
        else:
            with tracer.time_block("eval_next_token"):
                mx.eval(next_token)

        prefill_time = (time.perf_counter() - prefill_start) * 1000
        tracer.record_token_time("prefill", prefill_time / 1000.0)

        if verbose:
            print(f"Prefill completed in {prefill_time:.2f} ms")
            print(f"First token: {tokenizer.decode([next_token.item()])}")
            print()

        decode_times = []

        # === DECODE PHASE ===
        if verbose and max_tokens > 1:
            print("=" * 100)
            print("DECODE PHASE")
            print("=" * 100)

        tracer.set_phase("decode")
        if sync_analyzer is not None:
            sync_analyzer.set_phase("decode")
        generated_tokens = [next_token.item()]

        for i in range(max_tokens - 1):
            decode_start = time.perf_counter()

            with tracer.time_block("token_array"):
                next_token_arr = mx.array([[next_token.item()]])
            with tracer.time_block("model_forward"):
                model_output = wrapped_model(next_token_arr, cache=prompt_cache)

            if isinstance(model_output, tuple):
                logits = model_output[0]
            else:
                logits = model_output

            if logits.ndim > 1:
                logits = logits[:, -1, :]

            if sync_checkpoints:
                with tracer.time_block("sync_logits_ready"):
                    mx.eval(logits)

            with tracer.time_block("argmax"):
                next_token = mx.argmax(logits, axis=-1)
            if sync_checkpoints:
                with tracer.time_block("sync_next_token_ready"):
                    mx.eval(next_token)
            else:
                with tracer.time_block("eval_next_token"):
                    mx.eval(next_token)

            decode_time = (time.perf_counter() - decode_start) * 1000
            decode_times.append(decode_time)
            tracer.record_token_time("decode", decode_time / 1000.0)
            generated_tokens.append(next_token.item())

            if next_token.item() in tokenizer.eos_token_ids:
                break

            if verbose:
                print(
                    f"  Token {i + 2}: {decode_time:.2f} ms - '{tokenizer.decode([next_token.item()])}'"
                )

        if verbose and decode_times:
            avg_decode = sum(decode_times) / len(decode_times)
            print(f"\nAverage decode time: {avg_decode:.2f} ms/token")
            print(f"Decode throughput: {1000 / avg_decode:.2f} tokens/sec")

        # Generate full response
        generated_text = tokenizer.decode(generated_tokens)

        if verbose:
            print(f"\nGenerated: '{generated_text}'")

    finally:
        if sync_analyzer is not None:
            sync_analyzer.set_phase("post-run")
        wrapped_model.uninstall()
        tracer.uninstall()

    stats = {
        "input_len": input_len,
        "prefill_time_ms": prefill_time,
        "decode_times_ms": decode_times,
    }
    return model, tokenizer, tracer, stats


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(
        description="Profile Qwen3 model inference with Vulkan backend"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/qwen3-0.6b-bf16",
        choices=[
            "mlx-community/Qwen3.5-2B-bf16",
            "mlx-community/qwen3-0.6b-bf16",
        ],
        help="Model to profile (default: mlx-community/qwen3-0.6b-bf16)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt for generation (default: from env or 'Explain machine learning in simple terms.')",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max tokens to generate (default: 5)",
    )
    parser.add_argument(
        "--sync-checkpoints",
        action="store_true",
        help="Synchronize after each checkpoint for accurate timing",
    )
    parser.add_argument(
        "--no-capture-sync-trace",
        action="store_true",
        help="Disable Vulkan sync trace capture",
    )
    parser.add_argument(
        "--echo-sync-trace",
        action="store_true",
        help="Echo sync trace output to stderr",
    )
    args = parser.parse_args()

    model_name = os.environ.get("MLX_VULKAN_PROFILE_MODEL", args.model)
    prompt = args.prompt or os.environ.get(
        "MLX_VULKAN_PROFILE_PROMPT", "Explain machine learning in simple terms."
    )
    max_tokens = args.max_tokens if args.max_tokens is not None else env_int("MLX_VULKAN_PROFILE_MAX_TOKENS", 5)
    sync_checkpoints = args.sync_checkpoints or env_flag("MLX_VULKAN_PROFILE_SYNC_CHECKPOINTS", False)
    if args.no_capture_sync_trace:
        capture_sync_trace = False
    else:
        capture_sync_trace = env_flag("MLX_VULKAN_PROFILE_CAPTURE_SYNC_TRACE", True)
    echo_sync_trace = args.echo_sync_trace or env_flag("MLX_VULKAN_PROFILE_ECHO_SYNC_TRACE", False)

    if capture_sync_trace and "MLX_VULKAN_TRACE_SYNC" not in os.environ:
        os.environ["MLX_VULKAN_TRACE_SYNC"] = "1"

    fallback_analyzer = FallbackAnalyzer()
    sync_analyzer = SyncTraceAnalyzer()

    def handle_stderr_line(line: str):
        fallback_analyzer.consume_line(line)
        sync_analyzer.consume_line(line)

    print("=" * 100)
    print(f"VULKAN PROFILER - {model_name}")
    print("=" * 100)
    print()

    # Check for Vulkan backend
    device = mx.default_device()
    print(f"Active device: {device}")

    # Check environment variables
    if os.environ.get("MLX_VULKAN_DEFERRED_SUBMISSION"):
        print(
            f"MLX_VULKAN_DEFERRED_SUBMISSION: {os.environ['MLX_VULKAN_DEFERRED_SUBMISSION']}"
        )
    if os.environ.get("MLX_VULKAN_SUBMIT_ON_HAZARD"):
        print(f"MLX_VULKAN_SUBMIT_ON_HAZARD: {os.environ['MLX_VULKAN_SUBMIT_ON_HAZARD']}")
    if os.environ.get("MLX_VULKAN_TRACE_FALLBACKS"):
        print(f"MLX_VULKAN_TRACE_FALLBACKS: {os.environ['MLX_VULKAN_TRACE_FALLBACKS']}")
    if os.environ.get("MLX_VULKAN_TRACE_SYNC"):
        print(f"MLX_VULKAN_TRACE_SYNC: {os.environ['MLX_VULKAN_TRACE_SYNC']}")
    print(f"MLX_VULKAN_PROFILE_MAX_TOKENS: {max_tokens}")
    print(f"MLX_VULKAN_PROFILE_SYNC_CHECKPOINTS: {int(sync_checkpoints)}")
    print(f"MLX_VULKAN_PROFILE_CAPTURE_SYNC_TRACE: {int(capture_sync_trace)}")
    print(f"MLX_VULKAN_PROFILE_ECHO_SYNC_TRACE: {int(echo_sync_trace)}")
    print()

    # Run profiling
    if capture_sync_trace:
        with capture_stderr_lines(handle_stderr_line, echo_sync_trace):
            model, tokenizer, tracer, stats = trace_model_inference(
                model_name=model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                verbose=True,
                sync_checkpoints=sync_checkpoints,
                sync_analyzer=sync_analyzer,
            )
    else:
        model, tokenizer, tracer, stats = trace_model_inference(
            model_name=model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=True,
            sync_checkpoints=sync_checkpoints,
            sync_analyzer=sync_analyzer,
        )

    # Print detailed report
    print("\n")
    tracer.print_report()
    print("\n")
    print(sync_analyzer.get_report())
    print("\n")
    print(fallback_analyzer.get_report())

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Model: {model_name}")
    print(f"Device: {mx.default_device()}")
    print(f"Layers: {len(model.layers)}")

    # Phase timing from tracer
    print(f"\nPhase Timing:")
    for phase, total_time in sorted(tracer.phase_times.items()):
        print(f"  {phase.capitalize()}: {total_time * 1000:.2f} ms")

    if model_name == QWEN3_06B_BF16_GOAL["model"]:
        prefill_tps = 0.0
        if stats["prefill_time_ms"] > 0:
            prefill_tps = stats["input_len"] * 1000.0 / stats["prefill_time_ms"]
        decode_tps = 0.0
        if stats["decode_times_ms"]:
            decode_tps = 1000.0 / (sum(stats["decode_times_ms"]) / len(stats["decode_times_ms"]))

        print("\nQwen3 0.6B BF16 Goal:")
        print(f"  Source: {QWEN3_06B_BF16_GOAL['source']}")
        print(f"  Reference: {QWEN3_06B_BF16_GOAL['reference']}")
        print(
            f"  Prefill: {prefill_tps:.2f} t/s vs goal {QWEN3_06B_BF16_GOAL['prefill_tokens_per_sec']:.2f} t/s"
        )
        if stats["decode_times_ms"]:
            print(
                f"  Decode: {decode_tps:.2f} t/s vs goal {QWEN3_06B_BF16_GOAL['decode_tokens_per_sec']:.2f} t/s"
            )
        else:
            print(
                f"  Decode: n/a (goal {QWEN3_06B_BF16_GOAL['decode_tokens_per_sec']:.2f} t/s)"
            )

    print("\n" + "=" * 100)
    print("NOTES")
    print("=" * 100)
    print("""
Vulkan fallback messages are printed to stderr when MLX_VULKAN_TRACE_FALLBACKS=1.
To see fallback analysis, run with stderr visible:

  python mlx/backend/vulkan/profile_qwen3_vulkan.py 2>&1 | grep "vulkan-fallback"

Common fallbacks for Qwen3 on Vulkan:
  - RMSNorm (fast RMSNorm kernel needed)
  - ScaledDotProductAttention (attention kernel needed)
  - ScaledDotProductAttentionVJP (gradient attention kernel needed)
  - Softmax (for certain tensor shapes)

Deferred submission is ENABLED by default.
To disable deferred submission (use immediate submission):

  MLX_VULKAN_DEFERRED_SUBMISSION=0 python mlx/backend/vulkan/profile_qwen3_vulkan.py

Barrier-first hazard handling is now the default.
To temporarily restore the old submit-on-hazard behavior:

  MLX_VULKAN_SUBMIT_ON_HAZARD=1 python mlx/backend/vulkan/profile_qwen3_vulkan.py
""")
    print("=" * 100)


if __name__ == "__main__":
    main()
