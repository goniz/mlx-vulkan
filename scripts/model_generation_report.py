#!/usr/bin/env python3
"""Run serial generation smoke tests across a model matrix."""

import argparse
import gc
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

try:
    import mlx.core as mx
except ImportError:
    print("Error: mlx not installed.", file=sys.stderr)
    sys.exit(1)

try:
    from mlx_lm import generate as lm_generate
    from mlx_lm import load as lm_load
    from mlx_lm.sample_utils import make_sampler as lm_make_sampler
    from mlx_vlm import generate as vlm_generate
    from mlx_vlm import load as vlm_load
except ImportError:
    print(
        "Error: mlx_lm or mlx_vlm not installed. Run: ./dev.sh init-venv",
        file=sys.stderr,
    )
    sys.exit(1)


DEFAULT_MODELS = [
    "mlx-community/Qwen3-0.6B-bf16",
    "mlx-community/Qwen3-0.6B-8bit",
    "mlx-community/Qwen3.5-2B-bf16",
    "mlx-community/gemma-4-e2b-bf16",
]

DEFAULT_PROMPT = "Write one concise sentence about why Vulkan acceleration is useful."


@dataclass
class ModelResult:
    model: str
    generated_output: bool
    output_was_coherent: bool
    error_msg: str
    peak_mem_bytes: int | None
    peak_mem_gb: float | None
    output: str


def parse_models(values: list[str] | None) -> list[str]:
    if not values:
        return DEFAULT_MODELS

    models = []
    for value in values:
        models.extend(part.strip() for part in value.split(",") if part.strip())
    if not models:
        raise ValueError("--models did not contain any model names")
    return models


def looks_coherent(text: str) -> bool:
    text = text.strip()
    if not text:
        return False
    if "Traceback" in text or "RuntimeError" in text or "nan" in text.lower():
        return False
    if "\ufffd" in text:
        return False

    printable = sum(ch.isprintable() or ch.isspace() for ch in text)
    if printable / max(len(text), 1) < 0.95:
        return False

    words = re.findall(r"[A-Za-z][A-Za-z'-]*", text)
    if len(words) < 4:
        return False

    alpha_chars = sum(ch.isalpha() for ch in text)
    if alpha_chars / max(len(text), 1) < 0.35:
        return False

    lowered_words = [word.lower() for word in words]
    if len(lowered_words) >= 8:
        most_common = max(lowered_words.count(word) for word in set(lowered_words))
        if most_common / len(lowered_words) > 0.45:
            return False

    repeated_chunk = re.search(r"(.{8,}?)\1\1", text)
    return repeated_chunk is None


def bytes_to_gb(value: int | None) -> float | None:
    if value is None:
        return None
    return round(value / (1024**3), 3)


def format_mem(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def compact_error(exc: BaseException) -> str:
    message = str(exc).strip() or type(exc).__name__
    return " ".join(message.split())


def cleanup_model(model, tokenizer) -> None:
    del model
    del tokenizer
    gc.collect()
    try:
        mx.clear_cache()
    except Exception:
        pass


def run_model(
    model_name: str, prompt: str, max_tokens: int, temperature: float
) -> ModelResult:
    print(f"\n=== {model_name} ===", flush=True)
    output = ""
    error_msg = ""
    peak_mem_bytes = None
    model = None
    processor = None

    # --- mlx_lm path ---
    try:
        print("Loading with mlx_lm...", flush=True)
        model, processor = lm_load(model_name)
    except Exception as lm_load_exc:
        lm_load_error = compact_error(lm_load_exc)
        print(
            "mlx_lm load failed, trying mlx_vlm...",
            file=sys.stderr,
            flush=True,
        )

        # --- mlx_vlm fallback (load + generate) ---
        try:
            print("Loading with mlx_vlm...", flush=True)
            model, processor = vlm_load(model_name)
            print("Generating with mlx_vlm...", flush=True)
            mx.reset_peak_memory()
            result = vlm_generate(
                model,
                processor,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                verbose=False,
            )
            mx.eval(mx.array(0))
            output = result.text
            peak_mem_bytes = int(result.peak_memory * 1e9)
        except Exception as vlm_exc:
            vlm_error = compact_error(vlm_exc)
            error_msg = f"mlx_lm load: {lm_load_error}; mlx_vlm: {vlm_error}"
            try:
                peak_mem_bytes = int(mx.get_peak_memory())
            except Exception:
                peak_mem_bytes = None
            print(f"Failed: {error_msg}", file=sys.stderr, flush=True)
        cleanup_model(model, processor)
        model = None
        processor = None

    # --- mlx_lm generation (if loading succeeded) ---
    if model is not None and not error_msg:
        try:
            print("Generating with mlx_lm...", flush=True)
            mx.reset_peak_memory()
            output = lm_generate(
                model,
                processor,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=lm_make_sampler(temp=temperature),
                verbose=False,
            )
            mx.eval(mx.array(0))
            peak_mem_bytes = int(mx.get_peak_memory())
        except Exception as lm_gen_exc:
            error_msg = f"mlx_lm gen: {compact_error(lm_gen_exc)}"
            try:
                peak_mem_bytes = int(mx.get_peak_memory())
            except Exception:
                peak_mem_bytes = None
            print(f"Failed: {error_msg}", file=sys.stderr, flush=True)
        finally:
            cleanup_model(model, processor)

    if output.strip():
        print(output.strip(), flush=True)

    generated_output = bool(output.strip()) and not error_msg
    coherent = generated_output and looks_coherent(output)
    return ModelResult(
        model=model_name,
        generated_output=generated_output,
        output_was_coherent=coherent,
        error_msg=error_msg,
        peak_mem_bytes=peak_mem_bytes,
        peak_mem_gb=bytes_to_gb(peak_mem_bytes),
        output=output.strip(),
    )


def print_report(results: Iterable[ModelResult]) -> None:
    rows = list(results)
    headers = [
        "model",
        "generated_output",
        "output_was_coherent",
        "peak_mem_gb",
        "error_msg",
    ]
    table = [
        [
            result.model,
            str(result.generated_output),
            str(result.output_was_coherent),
            format_mem(result.peak_mem_gb),
            result.error_msg,
        ]
        for result in rows
    ]
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in table))
        if table
        else len(header)
        for idx, header in enumerate(headers)
    ]

    print("\nGeneration Report")
    print(" | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in table:
        print(" | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        help="Override model matrix. Accepts space-separated names or comma-separated lists.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also print the final report as JSON.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Write the final report as JSON to this path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    models = parse_models(args.models)
    results = [
        run_model(
            model_name=model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temp,
        )
        for model in models
    ]

    print_report(results)
    json_results = [asdict(result) for result in results]
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(json_results, indent=2) + "\n")
    if args.json:
        print("\nJSON Report")
        print(json.dumps(json_results, indent=2))

    failed = any(
        not result.generated_output or not result.output_was_coherent for result in results
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
