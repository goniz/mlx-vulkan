#!/usr/bin/env python3
"""
Update Qwen3 benchmark results in README.md.

Usage:
    python scripts/update_benchmark.py
    ./dev.sh update-benchmark

This script runs mlx_lm.benchmark for both bf16 and 8bit quantizations,
and patches the README.md with the new results while preserving other content.
"""

import os
import re
import subprocess
import sys
from pathlib import Path


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

    return (result.stdout + result.stderr).strip()


def update_readme_section(content: str, section_name: str, new_content: str) -> str:
    """Update a specific section in the README while preserving other content."""
    # Pattern to match section header and its code block
    pattern = rf"(### {section_name}\n```\n)(.*?)(```)"

    def replacer(match):
        return f"{match.group(1)}{new_content}{match.group(3)}"

    updated = re.sub(pattern, replacer, content, flags=re.DOTALL)
    return updated


def main():
    """Main entry point."""
    print("Running Qwen3 benchmarks and updating README.md...")
    print()

    # Run benchmarks
    bf16_output = run_benchmark("bf16")
    print(bf16_output)
    print()

    eightbit_output = run_benchmark("8bit")
    print(eightbit_output)
    print()

    # Read existing README
    readme_path = Path("README.md")
    if not readme_path.exists():
        print("Error: README.md not found")
        return 1

    readme_content = readme_path.read_text()

    # Update README sections
    readme_content = update_readme_section(readme_content, "bf16", bf16_output)
    readme_content = update_readme_section(readme_content, "8bit", eightbit_output)

    # Write updated README
    readme_path.write_text(readme_content)

    print("README.md updated successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
