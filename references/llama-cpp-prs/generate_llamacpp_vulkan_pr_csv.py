#!/usr/bin/env python3
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


csv.field_size_limit(sys.maxsize)


REPO_DIR = Path(__file__).resolve().parent / "llama.cpp"
OUTPUT_CSV = Path(__file__).resolve().parent / "llama.cpp-vulkan-pr-details.csv"
OWNER_REPO = "ggml-org/llama.cpp"
TARGET_PATHS = [
    "ggml/src/ggml-vulkan/ggml-vulkan.cpp",
    "ggml/include/ggml-vulkan.h",
]


MAX_RETRIES = 6


def run(cmd, cwd=None, check=True):
    for attempt in range(1, MAX_RETRIES + 1):
        result = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode == 0 or not check:
            return result.stdout

        combined = f"{result.stdout}\n{result.stderr}".lower()
        if "rate limit" in combined or "secondary rate limit" in combined:
            sleep_for = rate_limit_sleep_seconds(cwd)
            print(
                f"rate limited running {' '.join(cmd)}; sleeping {sleep_for}s (attempt {attempt}/{MAX_RETRIES})",
                file=sys.stderr,
            )
            time.sleep(sleep_for)
            continue

        if attempt < MAX_RETRIES and (
            "http 502" in combined
            or "http 503" in combined
            or "http 504" in combined
            or "connection reset" in combined
            or "tls" in combined
            or "timeout" in combined
        ):
            sleep_for = min(2**attempt, 30)
            print(
                f"transient failure running {' '.join(cmd)}; sleeping {sleep_for}s (attempt {attempt}/{MAX_RETRIES})",
                file=sys.stderr,
            )
            time.sleep(sleep_for)
            continue

        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    raise RuntimeError(f"command failed after retries: {' '.join(cmd)}")


def rate_limit_sleep_seconds(cwd):
    result = subprocess.run(
        ["gh", "api", "rate_limit"],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode == 0:
        try:
            data = json.loads(result.stdout)
            reset_at = data["resources"]["core"]["reset"]
            return max(5, int(reset_at - time.time()) + 5)
        except Exception:
            pass
    return 60


def git_commits_for_paths():
    cmd = ["git", "log", "--format=%H", "origin/master", "--", *TARGET_PATHS]
    out = run(cmd, cwd=REPO_DIR)
    return [line.strip() for line in out.splitlines() if line.strip()]


def gh_json(path, paginate=False):
    cmd = ["gh", "api"]
    if paginate:
        cmd.append("--paginate")
    cmd.extend([path])
    out = run(cmd, cwd=REPO_DIR)
    chunks = []
    decoder = json.JSONDecoder()
    idx = 0
    text = out.strip()
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        obj, idx = decoder.raw_decode(text, idx)
        chunks.append(obj)
    if not paginate:
        return chunks[0] if chunks else None
    merged = []
    for chunk in chunks:
        if isinstance(chunk, list):
            merged.extend(chunk)
        else:
            merged.append(chunk)
    return merged


def build_diff_from_files(pr_number):
    files = gh_json(
        f"repos/{OWNER_REPO}/pulls/{pr_number}/files?per_page=100",
        paginate=True,
    )
    parts = []
    for file_info in files:
        filename = file_info.get("filename", "")
        previous_filename = file_info.get("previous_filename")
        status = file_info.get("status", "")
        additions = file_info.get("additions", 0)
        deletions = file_info.get("deletions", 0)
        changes = file_info.get("changes", 0)
        patch = file_info.get("patch", "")

        header = [
            f"diff --github {filename}",
            f"status: {status}",
            f"additions: {additions}",
            f"deletions: {deletions}",
            f"changes: {changes}",
        ]
        if previous_filename:
            header.append(f"previous_filename: {previous_filename}")
        if patch:
            header.append(patch)
        else:
            header.append("<no textual patch available>")

        parts.append("\n".join(header))

    return "\n\n".join(parts)


def pr_numbers_from_commits(commits):
    prs = set()
    for i, sha in enumerate(commits, 1):
        if i == 1 or i % 25 == 0 or i == len(commits):
            print(f"resolving commits: {i}/{len(commits)}", file=sys.stderr)
        data = gh_json(f"repos/{OWNER_REPO}/commits/{sha}/pulls")
        for pr in data or []:
            if pr.get("merged_at"):
                prs.add(pr["number"])
    return sorted(prs)


def collect_comments(pr_number):
    issue_comments = gh_json(
        f"repos/{OWNER_REPO}/issues/{pr_number}/comments?per_page=100",
        paginate=True,
    )
    review_comments = gh_json(
        f"repos/{OWNER_REPO}/pulls/{pr_number}/comments?per_page=100",
        paginate=True,
    )
    comments = []
    for comment in issue_comments:
        comments.append(
            {
                "type": "issue_comment",
                "author": comment.get("user", {}).get("login"),
                "created_at": comment.get("created_at"),
                "body": comment.get("body", ""),
            }
        )
    for comment in review_comments:
        comments.append(
            {
                "type": "review_comment",
                "author": comment.get("user", {}).get("login"),
                "created_at": comment.get("created_at"),
                "path": comment.get("path"),
                "body": comment.get("body", ""),
            }
        )
    comments.sort(key=lambda c: (c.get("created_at") or "", c.get("type") or ""))
    return comments


def collect_pr_row(pr_number, current, total):
    print(f"fetching PRs: {current}/{total} (#{pr_number})", file=sys.stderr)
    pr = gh_json(f"repos/{OWNER_REPO}/pulls/{pr_number}")
    comments = collect_comments(pr_number)
    diff = build_diff_from_files(pr_number)
    return {
        "pr_number": pr["number"],
        "pr_title": pr.get("title", ""),
        "pr_body": pr.get("body", "") or "",
        "pr_comments": json.dumps(comments, ensure_ascii=True),
        "pr_diff": diff,
    }


def read_existing_pr_numbers(csv_path):
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    existing = set()
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            value = row.get("pr_number")
            if value:
                existing.add(int(value))
    return existing


def main():
    commits = git_commits_for_paths()
    print(f"found {len(commits)} commits touching target files", file=sys.stderr)
    pr_numbers = pr_numbers_from_commits(commits)
    print(f"found {len(pr_numbers)} merged PRs", file=sys.stderr)

    fieldnames = ["pr_number", "pr_title", "pr_body", "pr_comments", "pr_diff"]
    existing = read_existing_pr_numbers(OUTPUT_CSV)
    mode = "a" if OUTPUT_CSV.exists() and OUTPUT_CSV.stat().st_size > 0 else "w"

    with OUTPUT_CSV.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
            f.flush()

        for index, pr_number in enumerate(pr_numbers, 1):
            if pr_number in existing:
                print(
                    f"fetching PRs: {index}/{len(pr_numbers)} (#{pr_number}) [skip existing]",
                    file=sys.stderr,
                )
                continue
            row = collect_pr_row(pr_number, index, len(pr_numbers))
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())

    print(f"wrote {OUTPUT_CSV}", file=sys.stderr)


if __name__ == "__main__":
    main()
