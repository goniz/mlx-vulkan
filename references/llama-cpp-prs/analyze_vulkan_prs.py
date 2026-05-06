#!/usr/bin/env python3
"""
Analyze llama.cpp Vulkan PRs for performance-related changes.
"""

import csv
import re
import os
import sys
from collections import Counter

# Increase CSV field size limit to handle large diff fields
csv.field_size_limit(sys.maxsize)

# Keywords that indicate performance-related PRs
PERF_KEYWORDS = [
    "perf",
    "performance",
    "speed",
    "fast",
    "faster",
    "accelerate",
    "optimize",
    "optimization",
    "efficiency",
    "memory",
    "throughput",
    "latency",
    "shader",
    "kernel",
    "simd",
    "vectorize",
    "parallel",
    "cache",
    "bandwidth",
    "flops",
    "compute",
    "reduce",
    "improve",
    "boost",
    "scale",
    "scalable",
    "vec",
    "unroll",
    "prefetch",
    "coalesce",
    "shared memory",
    "local memory",
    "register",
]


def score_performance_relevance(row):
    """
    Score a PR based on how likely it is to be performance-related.
    Returns a tuple (score, details) where higher score = more performance-relevant.
    """
    pr_number = row.get("pr_number", "")
    pr_title = row.get("pr_title", "").lower()
    pr_body = row.get("pr_body", "").lower()
    pr_comments = row.get("pr_comments", "").lower()
    pr_diff = row.get("pr_diff", "").lower()

    score = 0
    details = []

    # Check title - title matches are weighted heavily
    for keyword in PERF_KEYWORDS:
        if keyword in pr_title:
            score += 5
            details.append(f"Title contains '{keyword}'")

    # Check body for performance keywords
    for keyword in PERF_KEYWORDS:
        if keyword in pr_body:
            score += 2
            details.append(f"Body mentions '{keyword}'")

    # Check comments
    for keyword in PERF_KEYWORDS:
        if keyword in pr_comments:
            score += 1

    # Look for benchmark results (tokens/sec, GB/s, speedup)
    benchmark_patterns = [
        r"\d+\.?\d*\s*t/s",  # tokens per second
        r"\d+\.?\d*\s*tok/s",
        r"\d+\.?\d*\s*gb/s",
        r"\d+\.?\d*\s*gflops",
        r"\d+\.?\d*\s*ms",
        r"\d+\.?\d*\s*x\s*faster",
        r"speedup[:\s]+\d+",
        r"improvement[:\s]+\d+",
        r"benchmark",
        r"profil",
    ]

    combined_text = pr_title + " " + pr_body + " " + pr_comments
    for pattern in benchmark_patterns:
        if re.search(pattern, combined_text, re.IGNORECASE):
            score += 3
            details.append(f"Contains benchmark pattern '{pattern}'")

    # Check for kernel/shader changes in diff
    if "kernel" in pr_diff or "shader" in pr_diff:
        score += 2
        details.append("Diff contains kernel/shader changes")

    # Check diff size - larger changes might indicate significant refactoring
    diff_size = len(pr_diff)
    if diff_size > 5000:
        score += 1
        details.append(f"Large diff ({diff_size} chars)")

    # Specific high-value keywords
    high_value_keywords = [
        "vkCmd",
        "pipeline",
        "descriptor",
        "barrier",
        "memory pool",
        "subgroup",
        "warp",
        "workgroup",
        "dispatch",
    ]
    for keyword in high_value_keywords:
        if keyword in pr_diff or keyword in pr_title:
            score += 3
            details.append(f"Contains Vulkan performance keyword '{keyword}'")

    return score, details, diff_size


def analyze_prs(csv_path):
    """Parse CSV and identify top performance PRs."""

    prs = []

    print(f"Parsing {csv_path}...")

    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader):
            if i % 500 == 0:
                print(f"  Processed {i} PRs...", end="\r")

            score, details, diff_size = score_performance_relevance(row)

            if score > 5:  # Only keep PRs with meaningful performance scores
                prs.append(
                    {
                        "number": row.get("pr_number", "N/A"),
                        "title": row.get("pr_title", "N/A"),
                        "body": row.get("pr_body", ""),
                        "comments": row.get("pr_comments", ""),
                        "diff": row.get("pr_diff", ""),
                        "score": score,
                        "details": details,
                        "diff_size": diff_size,
                    }
                )

    print(f"\nTotal PRs analyzed: {i + 1}")
    print(f"Performance-related PRs found: {len(prs)}")

    # Sort by score (descending)
    prs.sort(key=lambda x: x["score"], reverse=True)

    return prs


def extract_key_changes(diff_text, max_lines=30):
    """Extract a summary of key changes from the diff."""
    lines = diff_text.split("\n")
    changes = []

    # Look for added/modified lines in key files
    for line in lines:
        if line.startswith("+") and not line.startswith("+++"):
            # Skip comment lines and empty additions
            stripped = line[1:].strip()
            if stripped and not stripped.startswith("//"):
                if any(
                    keyword in line.lower()
                    for keyword in [
                        "vk",
                        "shader",
                        "kernel",
                        "dispatch",
                        "pipeline",
                        "memory",
                        "buffer",
                        "optimize",
                    ]
                ):
                    changes.append(stripped[:100])  # Limit line length

        if len(changes) >= max_lines:
            break

    return changes[:15]  # Return top 15 relevant changes


def print_top_prs(prs, n=5):
    """Print detailed info about top N PRs."""

    print("\n" + "=" * 80)
    print(f"TOP {n} VULKAN PERFORMANCE PRs")
    print("=" * 80)

    for i, pr in enumerate(prs[:n], 1):
        print(f"\n{'#' * 80}")
        print(f"#{i}. PR #{pr['number']}: {pr['title']}")
        print(f"{'#' * 80}")
        print(f"\nPerformance Score: {pr['score']}")
        print(f"Diff Size: {pr['diff_size']} characters")

        print(f"\nKey Performance Indicators:")
        for detail in pr["details"][:10]:  # Show top 10 details
            print(f"  - {detail}")

        # Extract a brief from the body
        body_preview = pr["body"][:500].replace("\n", " ")
        if body_preview:
            print(f"\nPR Description (preview):")
            print(f"  {body_preview}...")

        # Show key code changes
        print(f"\nKey Code Changes (from diff):")
        changes = extract_key_changes(pr["diff"])
        for change in changes[:8]:
            print(f"  + {change}")

        print()


def main():
    csv_path = (
        "/home/goniz/Work/mlx/mlx-vulkan/references/llama.cpp-vulkan-pr-details.csv"
    )

    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        return

    prs = analyze_prs(csv_path)
    print_top_prs(prs, n=5)

    # Also print some statistics
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)

    # Count keyword occurrences
    all_text = " ".join([pr["title"] + " " + pr["body"] for pr in prs])
    keyword_counts = Counter()

    for keyword in PERF_KEYWORDS:
        count = all_text.lower().count(keyword)
        if count > 0:
            keyword_counts[keyword] = count

    print("\nTop Performance Keywords in High-Scoring PRs:")
    for keyword, count in keyword_counts.most_common(15):
        print(f"  {keyword}: {count} occurrences")


if __name__ == "__main__":
    main()
