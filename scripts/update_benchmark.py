#!/usr/bin/env python3
"""Run Vulkan benchmarks and regenerate benchmark history reports."""

import argparse
import csv
import html
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_DIR = ROOT / "benchmarks"
RESULTS_CSV = BENCHMARKS_DIR / "results.csv"
BENCHMARK_README = BENCHMARKS_DIR / "README.md"
ROOT_README = ROOT / "README.md"
PROMPT_GRAPH = BENCHMARKS_DIR / "prompt_tps.svg"
GENERATION_GRAPH = BENCHMARKS_DIR / "generation_tps.svg"

DEFAULT_VARIANTS = [
    ("mlx-community/Qwen3-0.6B-bf16", "bf16"),
    ("mlx-community/Qwen3-0.6B-8bit", "8bit"),
]

FIELDNAMES = [
    "timestamp_utc",
    "github_run_id",
    "github_run_number",
    "github_run_attempt",
    "github_workflow",
    "ci_event",
    "ci_ref",
    "run_url",
    "mlx_vulkan_commit",
    "mlx_commit",
    "model_name",
    "model_bits",
    "prompt_tokens",
    "generation_tokens",
    "batch_size",
    "prompt_tps",
    "generation_tps",
    "peak_memory_gb",
]

AVERAGES_RE = re.compile(
    r"Averages:\s+prompt_tps=(?P<prompt>[0-9.]+),\s+"
    r"generation_tps=(?P<generation>[0-9.]+),\s+"
    r"peak_memory=(?P<memory>[0-9.]+)"
)
TIMING_RE = re.compile(
    r"Timing with prompt_tokens=(?P<prompt_tokens>\d+),\s+"
    r"generation_tokens=(?P<generation_tokens>\d+),\s+"
    r"batch_size=(?P<batch_size>\d+)\."
)


def run_git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def current_metadata(timestamp: str) -> dict[str, str]:
    repository = os.environ.get("GITHUB_REPOSITORY", "")
    server_url = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    run_url = (
        f"{server_url}/{repository}/actions/runs/{run_id}"
        if repository and run_id
        else ""
    )

    return {
        "timestamp_utc": timestamp,
        "github_run_id": run_id,
        "github_run_number": os.environ.get("GITHUB_RUN_NUMBER", ""),
        "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", ""),
        "github_workflow": os.environ.get("GITHUB_WORKFLOW", ""),
        "ci_event": os.environ.get("GITHUB_EVENT_NAME", "local"),
        "ci_ref": os.environ.get("GITHUB_REF_NAME", os.environ.get("GITHUB_REF", "")),
        "run_url": run_url,
        "mlx_vulkan_commit": run_git(["rev-parse", "HEAD"]),
        "mlx_commit": run_git(["-C", "mlx", "rev-parse", "HEAD"]),
    }


def run_benchmark(
    model: str,
    prompt_tokens: int,
    generation_tokens: int,
    batch_size: int,
) -> str:
    print(f"Running benchmark for {model}...")

    env = os.environ.copy()
    env["OMPI_MCA_accelerator"] = "^rocm"
    env.setdefault("MLX_MPI_LIBNAME", "/dev/null")

    cmd = [
        "mlx_lm.benchmark",
        "--model",
        model,
        "-p",
        str(prompt_tokens),
        "-g",
        str(generation_tokens),
        "-b",
        str(batch_size),
    ]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)

    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"Benchmark failed for {model} with exit code {result.returncode}"
        )

    return result.stdout


def parse_benchmark_output(output: str) -> dict[str, str]:
    timing = TIMING_RE.search(output)
    averages = AVERAGES_RE.search(output)
    if not timing or not averages:
        raise ValueError(f"Could not parse mlx_lm.benchmark output:\n{output}")

    return {
        "prompt_tokens": timing.group("prompt_tokens"),
        "generation_tokens": timing.group("generation_tokens"),
        "batch_size": timing.group("batch_size"),
        "prompt_tps": averages.group("prompt"),
        "generation_tps": averages.group("generation"),
        "peak_memory_gb": averages.group("memory"),
    }


def append_rows(rows: list[dict[str, str]]) -> None:
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    needs_header = not RESULTS_CSV.exists() or RESULTS_CSV.stat().st_size == 0
    with RESULTS_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if needs_header:
            writer.writeheader()
        writer.writerows(rows)


def load_rows() -> list[dict[str, str]]:
    if not RESULTS_CSV.exists():
        return []
    with RESULTS_CSV.open(newline="") as f:
        return list(csv.DictReader(f))


def write_empty_results_file() -> None:
    if RESULTS_CSV.exists():
        return
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_CSV.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()


def short_commit(commit: str) -> str:
    return commit[:7] if commit else ""


def model_label(row: dict[str, str]) -> str:
    model = row.get("model_name", "")
    name = model.rsplit("/", 1)[-1] or model
    bits = row.get("model_bits", "")
    if bits and name.lower().endswith(bits.lower()):
        return name
    return f"{name} {bits}".strip()


def run_key(row: dict[str, str]) -> str:
    run_id = row.get("github_run_id", "")
    if run_id:
        attempt = row.get("github_run_attempt", "")
        return f"{run_id}.{attempt}" if attempt else run_id
    return row.get("timestamp_utc", "")


def safe_float(row: dict[str, str], key: str) -> float | None:
    try:
        return float(row.get(key, ""))
    except ValueError:
        return None


def no_data_svg(title: str) -> str:
    return f"""<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"900\" height=\"320\" viewBox=\"0 0 900 320\" role=\"img\" aria-label=\"{html.escape(title)}\">
  <rect width=\"900\" height=\"320\" fill=\"#0f172a\" rx=\"12\"/>
  <text x=\"450\" y=\"145\" text-anchor=\"middle\" font-family=\"Inter,Segoe UI,sans-serif\" font-size=\"24\" fill=\"#e2e8f0\">{html.escape(title)}</text>
  <text x=\"450\" y=\"180\" text-anchor=\"middle\" font-family=\"Inter,Segoe UI,sans-serif\" font-size=\"16\" fill=\"#94a3b8\">No benchmark data recorded yet</text>
</svg>
"""


def timestamp_label(timestamp: str) -> str:
    if "T" not in timestamp:
        return timestamp
    return timestamp.replace("T", " ").replace("Z", "")[:16]


def generate_svg(
    rows: list[dict[str, str]], metric: str, title: str, output_path: Path
) -> None:
    points_by_label: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
    run_timestamps: dict[str, str] = {}

    for row in rows:
        value = safe_float(row, metric)
        key = run_key(row)
        if value is None or not key:
            continue
        run_timestamps.setdefault(key, row.get("timestamp_utc", key))
        points_by_label[model_label(row)].append(
            (key, row.get("timestamp_utc", key), value)
        )

    if not points_by_label:
        output_path.write_text(no_data_svg(title))
        return

    ordered_runs = sorted(run_timestamps, key=lambda key: run_timestamps[key])
    x_index = {key: i for i, key in enumerate(ordered_runs)}
    values = [value for points in points_by_label.values() for _, _, value in points]
    max_value = max(values)
    y_max = max_value * 1.12 if max_value > 0 else 1.0

    width = 900
    height = 420
    margin_left = 82
    margin_right = 230
    margin_top = 54
    margin_bottom = 94
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    palette = ["#60a5fa", "#f97316", "#34d399", "#f43f5e", "#a78bfa", "#facc15"]

    def x_pos(key: str) -> float:
        if len(ordered_runs) == 1:
            return margin_left + plot_width / 2
        return margin_left + plot_width * x_index[key] / (len(ordered_runs) - 1)

    def y_pos(value: float) -> float:
        return margin_top + plot_height * (1 - value / y_max)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f'<rect width="{width}" height="{height}" fill="#0f172a" rx="12"/>',
        f'<text x="{margin_left}" y="32" font-family="Inter,Segoe UI,sans-serif" font-size="22" font-weight="700" fill="#f8fafc">{html.escape(title)}</text>',
    ]

    for i in range(6):
        value = y_max * i / 5
        y = y_pos(value)
        parts.append(
            f'<line x1="{margin_left}" y1="{y:.1f}" x2="{margin_left + plot_width}" y2="{y:.1f}" stroke="#1e293b" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{margin_left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Inter,Segoe UI,sans-serif" font-size="12" fill="#94a3b8">{value:.1f}</text>'
        )

    parts.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#475569"/>'
    )
    parts.append(
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#475569"/>'
    )

    label_stride = max(1, math.ceil(len(ordered_runs) / 6))
    for i, key in enumerate(ordered_runs):
        if i % label_stride != 0 and i != len(ordered_runs) - 1:
            continue
        x = x_pos(key)
        label = timestamp_label(run_timestamps[key])
        parts.append(
            f'<text x="{x:.1f}" y="{height - 34}" text-anchor="end" transform="rotate(-32 {x:.1f} {height - 34})" font-family="Inter,Segoe UI,sans-serif" font-size="12" fill="#94a3b8">{html.escape(label)}</text>'
        )

    for series_index, (label, points) in enumerate(sorted(points_by_label.items())):
        color = palette[series_index % len(palette)]
        ordered_points = sorted(points, key=lambda point: (x_index[point[0]], point[1]))
        coords = [(x_pos(key), y_pos(value), value) for key, _, value in ordered_points]
        if len(coords) > 1:
            polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in coords)
            parts.append(
                f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'
            )
        for x, y, value in coords:
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" stroke="#0f172a" stroke-width="2"><title>{html.escape(label)}: {value:.3f}</title></circle>'
            )

        legend_y = margin_top + 22 + series_index * 24
        legend_x = margin_left + plot_width + 32
        parts.append(f'<circle cx="{legend_x}" cy="{legend_y}" r="5" fill="{color}"/>')
        parts.append(
            f'<text x="{legend_x + 14}" y="{legend_y + 4}" font-family="Inter,Segoe UI,sans-serif" font-size="13" fill="#cbd5e1">{html.escape(label)}</text>'
        )

    parts.append("</svg>")
    output_path.write_text("\n".join(parts) + "\n")


def latest_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    latest: dict[tuple[str, str], dict[str, str]] = {}
    for row in sorted(rows, key=lambda r: r.get("timestamp_utc", "")):
        latest[(row.get("model_name", ""), row.get("model_bits", ""))] = row
    return [latest[key] for key in sorted(latest)]


def latest_table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "No benchmark results recorded yet."

    lines = [
        "| Model | Bits | Prompt TPS | Generation TPS | Peak memory (GB) | mlx-vulkan | mlx | Run |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in latest_rows(rows):
        run = row.get("ci_event") or "local"
        if row.get("run_url"):
            run = f"[run]({row['run_url']})"
        lines.append(
            "| {model} | {bits} | {prompt} | {generation} | {memory} | {vulkan} | {mlx} | {run} |".format(
                model=row.get("model_name", ""),
                bits=row.get("model_bits", ""),
                prompt=row.get("prompt_tps", ""),
                generation=row.get("generation_tps", ""),
                memory=row.get("peak_memory_gb", ""),
                vulkan=short_commit(row.get("mlx_vulkan_commit", "")),
                mlx=short_commit(row.get("mlx_commit", "")),
                run=run,
            )
        )
    return "\n".join(lines)


def build_benchmark_readme(rows: list[dict[str, str]]) -> str:
    return f"""# Vulkan Benchmark History

This file is generated by `scripts/update_benchmark.py`. Do not edit it by hand.

Benchmark data is stored in `results.csv`. Graphs are regenerated from that file whenever `./dev.sh update-benchmark` runs.

## Prompt Throughput

![Prompt TPS](prompt_tps.svg)

## Generation Throughput

![Generation TPS](generation_tps.svg)

## Latest Results

{latest_table(rows)}
"""


def build_root_benchmark_section(rows: list[dict[str, str]]) -> str:
    return f"""## Benchmark Results

CI benchmark history from AMD Radeon 8060S (Strix Halo). Detailed data is in `benchmarks/results.csv`.

![Prompt TPS](benchmarks/prompt_tps.svg)

![Generation TPS](benchmarks/generation_tps.svg)

### Latest Results

{latest_table(rows)}
"""


def replace_benchmark_section(readme: str, section: str) -> str:
    pattern = re.compile(
        r"^## Benchmark Results\n.*?(?=^##\s|\Z)", re.MULTILINE | re.DOTALL
    )
    if pattern.search(readme):
        return pattern.sub(section.rstrip() + "\n", readme)
    return readme.rstrip() + "\n\n" + section.rstrip() + "\n"


def regenerate_reports() -> None:
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    write_empty_results_file()
    rows = load_rows()
    generate_svg(rows, "prompt_tps", "Prompt throughput (tokens/sec)", PROMPT_GRAPH)
    generate_svg(
        rows, "generation_tps", "Generation throughput (tokens/sec)", GENERATION_GRAPH
    )
    BENCHMARK_README.write_text(build_benchmark_readme(rows))

    if ROOT_README.exists():
        readme = ROOT_README.read_text()
    else:
        readme = "# mlx-vulkan\n"
    ROOT_README.write_text(
        replace_benchmark_section(readme, build_root_benchmark_section(rows))
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-only",
        action="store_true",
        help="Regenerate graphs and README files without running benchmarks.",
    )
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--generation-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.refresh_only:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        metadata = current_metadata(timestamp)
        rows = []
        for model, bits in DEFAULT_VARIANTS:
            output = run_benchmark(
                model,
                prompt_tokens=args.prompt_tokens,
                generation_tokens=args.generation_tokens,
                batch_size=args.batch_size,
            )
            rows.append(
                {
                    **metadata,
                    "model_name": model,
                    "model_bits": bits,
                    **parse_benchmark_output(output),
                }
            )
            print()
        append_rows(rows)

    regenerate_reports()
    print(f"Updated {RESULTS_CSV.relative_to(ROOT)}")
    print(f"Updated {BENCHMARK_README.relative_to(ROOT)}")
    print(f"Updated {ROOT_README.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
