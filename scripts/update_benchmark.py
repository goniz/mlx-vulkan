#!/usr/bin/env python3
"""
Update Qwen3 benchmark baseline and print comparison.

Usage:
    python scripts/update_benchmark.py
    ./dev.sh update-benchmark

This script runs mlx_lm.benchmark for both bf16 and 8bit quantizations,
updates the baseline file, and prints a comparison with previous results.
"""

import os
import re
import subprocess
import sys
from pathlib import Path


BASELINE_FILE = Path("benchmarks/qwen3_baseline.txt")


def run_benchmark(quant: str) -> str:
    """Run mlx_lm.benchmark and return output."""
    print(f"Running {quant} benchmark...")

    env = os.environ.copy()
    env["OMPI_MCA_accelerator"] = "^rocm"  # Disable OpenMPI ROCm accelerator

    cmd = [
        "mlx_lm.benchmark",
        "--model",
        f"mlx-community/Qwen3-0.6B-{quant}",
        "-p",
        "4096",
        "-g",
        "128",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    return result.stdout + result.stderr


def parse_metrics(output: str) -> dict:
    """Parse benchmark output and extract key metrics."""
    metrics = {}

    # Look for Averages line
    match = re.search(
        r"Averages:\s+prompt_tps=([\d.]+),\s+generation_tps=([\d.]+),\s+peak_memory=([\d.]+)",
        output,
    )
    if match:
        metrics["prompt_tps"] = float(match.group(1))
        metrics["generation_tps"] = float(match.group(2))
        metrics["peak_memory"] = float(match.group(3))

    return metrics


def compare_metrics(old: dict, new: dict, label: str) -> list:
    """Generate comparison lines for a set of metrics."""
    lines = []

    for metric in ["prompt_tps", "generation_tps"]:
        old_val = old.get(metric, 0)
        new_val = new.get(metric, 0)

        if old_val and new_val:
            diff = new_val - old_val
            pct = ((new_val - old_val) / old_val) * 100
            metric_name = metric.replace("_", " ").title()
            lines.append(
                f"  {metric_name:20s} {old_val:10.2f} → {new_val:10.2f} "
                f"({diff:+.2f}, {pct:+.1f}%)"
            )

    return lines


def main():
    """Main entry point."""
    print("Updating Qwen3 benchmark baseline...")
    print()

    # Store old baseline if it exists
    old_baseline = ""
    old_metrics = {"bf16": {}, "8bit": {}}

    if BASELINE_FILE.exists():
        old_baseline = BASELINE_FILE.read_text()
        # Split into sections
        sections = old_baseline.split("\n\n")
        for section in sections:
            if "Qwen3-0.6B" in section and "8bit" not in section:
                old_metrics["bf16"] = parse_metrics(section)
            elif "Qwen3-0.6B-8bit" in section:
                old_metrics["8bit"] = parse_metrics(section)

    # Run benchmarks
    results = []

    bf16_output = run_benchmark("bf16")
    results.append(bf16_output)
    print(bf16_output)
    print()

    eightbit_output = run_benchmark("8bit")
    results.append(eightbit_output)
    print(eightbit_output)

    # Parse new metrics
    new_metrics = {
        "bf16": parse_metrics(bf16_output),
        "8bit": parse_metrics(eightbit_output),
    }

    # Save to baseline file
    BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
    new_baseline = "\n\n".join(results)
    BASELINE_FILE.write_text(new_baseline)

    print()
    print(f"Baseline updated: {BASELINE_FILE}")
    print()

    # Print comparison
    if old_baseline:
        print("=" * 70)
        print("PERFORMANCE COMPARISON")
        print("=" * 70)
        print()

        print("BF16 Changes:")
        for line in compare_metrics(old_metrics["bf16"], new_metrics["bf16"], "BF16"):
            print(line)
        print()

        print("8bit Changes:")
        for line in compare_metrics(old_metrics["8bit"], new_metrics["8bit"], "8bit"):
            print(line)
        print()

        print("=" * 70)
    else:
        print("No previous baseline found for comparison.")
        print()
        print("New metrics:")
        print(f"  BF16 Prompt TPS:      {new_metrics['bf16'].get('prompt_tps', 'N/A')}")
        print(
            f"  BF16 Generation TPS:  {new_metrics['bf16'].get('generation_tps', 'N/A')}"
        )
        print(f"  8bit Prompt TPS:      {new_metrics['8bit'].get('prompt_tps', 'N/A')}")
        print(
            f"  8bit Generation TPS:  {new_metrics['8bit'].get('generation_tps', 'N/A')}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
