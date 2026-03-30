#!/usr/bin/env python3
"""
Fetch non-resolved PR review comments (including line comments/threads)
for the current branch from GitHub.

Usage:
    python scripts/fetch_pr_comments.py
    python scripts/fetch_pr_comments.py --repo owner/repo
    python scripts/fetch_pr_comments.py --pr-number 16 --repo owner/repo
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd, cwd=None, check=True):
    """Run a shell command and return output."""
    result = subprocess.run(
        cmd, shell=True, cwd=cwd, capture_output=True, text=True, check=check
    )
    return result.stdout.strip()


def get_current_branch(cwd=None):
    """Get the current git branch name."""
    return run_cmd("git rev-parse --abbrev-ref HEAD", cwd=cwd)


def get_repo_from_remote(cwd=None):
    """Extract owner/repo from git remote origin URL."""
    remote_url = run_cmd("git remote get-url origin", cwd=cwd)

    # Handle HTTPS: https://github.com/owner/repo.git
    # Handle SSH: git@github.com:owner/repo.git
    if remote_url.startswith("https://github.com/"):
        repo = remote_url.replace("https://github.com/", "").replace(".git", "")
    elif remote_url.startswith("git@github.com:"):
        repo = remote_url.replace("git@github.com:", "").replace(".git", "")
    else:
        raise ValueError(f"Unsupported remote URL format: {remote_url}")

    return repo


def find_pr_for_branch(branch, repo):
    """Find open PR number for the given branch."""
    cmd = f'gh pr list --head "{branch}" --repo "{repo}" --json number,state --jq ".[0].number"'
    result = run_cmd(cmd, check=False)

    if not result:
        # Try with different repo formats
        parts = repo.split("/")
        if len(parts) == 2:
            # Try searching in the mlx submodule context
            cmd = f'gh pr list --head "{branch}" --json number,state --jq ".[0].number"'
            result = run_cmd(cmd, check=False)

    if not result or result == "null":
        return None

    return int(result)


def fetch_review_threads(pr_number, repo):
    """Fetch review threads with GraphQL to get resolution status."""

    # Extract owner and repo
    owner, repo_name = repo.split("/")

    # GraphQL query to get review threads with resolution status
    query = """
    query($owner: String!, $repo: String!, $pr_number: Int!) {
      repository(owner: $owner, name: $repo) {
        pullRequest(number: $pr_number) {
          reviewThreads(first: 100) {
            nodes {
              id
              isResolved
              isCollapsed
              comments(first: 100) {
                nodes {
                  id
                  body
                  path
                  line
                  state
                  author {
                    login
                  }
                  createdAt
                  commit {
                    oid
                  }
                  originalCommit {
                    oid
                  }
                }
              }
            }
          }
        }
      }
    }
    """

    variables = {"owner": owner, "repo": repo_name, "pr_number": pr_number}

    # Use gh api graphql
    json_query = json.dumps({"query": query, "variables": variables})
    cmd = f'gh api graphql -f query="{query}" -F owner="{owner}" -F repo="{repo_name}" -F pr_number={pr_number}'

    # Build the proper command
    cmd = [
        "gh",
        "api",
        "graphql",
        "-f",
        f"query={query}",
        "-F",
        f"owner={owner}",
        "-F",
        f"repo={repo_name}",
        "-F",
        f"pr_number={pr_number}",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        print(f"Error fetching review threads: {result.stderr}", file=sys.stderr)
        # Fallback to REST API
        return fetch_review_comments_rest(pr_number, repo)

    data = json.loads(result.stdout)

    if "errors" in data:
        print(f"GraphQL errors: {data['errors']}", file=sys.stderr)
        return fetch_review_comments_rest(pr_number, repo)

    return data


def fetch_review_comments_rest(pr_number, repo):
    """Fallback: Fetch review comments using REST API (no resolution status)."""
    cmd = f"gh api repos/{repo}/pulls/{pr_number}/comments --paginate"
    result = run_cmd(cmd, check=False)

    if not result:
        return {
            "data": {"repository": {"pullRequest": {"reviewThreads": {"nodes": []}}}}
        }

    comments = json.loads(result)

    # Transform to similar structure as GraphQL
    nodes = []
    for comment in comments:
        # In REST API, state can be: null (active), "OUTDATED", "RESOLVED"
        # But this is actually per-comment, not per-thread
        # For simplicity, we treat non-outdated as active
        if comment.get("state") != "OUTDATED":
            nodes.append(
                {
                    "isResolved": False,
                    "comments": {
                        "nodes": [
                            {
                                "id": comment["id"],
                                "body": comment["body"],
                                "path": comment["path"],
                                "line": comment.get("line")
                                or comment.get("original_line"),
                                "state": comment.get("state"),
                                "author": {
                                    "login": comment.get("user", {}).get(
                                        "login", "unknown"
                                    )
                                },
                                "createdAt": comment.get("created_at"),
                            }
                        ]
                    },
                }
            )

    return {
        "data": {"repository": {"pullRequest": {"reviewThreads": {"nodes": nodes}}}}
    }


def format_comment(comment, thread_resolved=False):
    """Format a single comment for display."""
    lines = []

    author = comment.get("author", {}).get("login", "unknown")
    path = comment.get("path", "unknown")
    line = comment.get("line", "N/A")
    body = comment.get("body", "")
    created_at = comment.get("createdAt", "unknown")
    state = comment.get("state")

    lines.append(f"Author: @{author}")
    lines.append(f"File: {path}:{line}")
    lines.append(f"Created: {created_at}")
    if state:
        lines.append(f"State: {state}")
    if thread_resolved:
        lines.append("Status: RESOLVED")
    else:
        lines.append("Status: ACTIVE")
    lines.append("")
    lines.append(body)
    lines.append("")
    lines.append("-" * 80)

    return "\n".join(lines)


def process_review_data(data):
    """Process review threads data and extract non-resolved comments."""
    threads = (
        data.get("data", {})
        .get("repository", {})
        .get("pullRequest", {})
        .get("reviewThreads", {})
        .get("nodes", [])
    )

    active_comments = []
    resolved_comments = []

    for thread in threads:
        is_resolved = thread.get("isResolved", False)
        comments = thread.get("comments", {}).get("nodes", [])

        for comment in comments:
            # Skip outdated comments (code changed)
            if comment.get("state") == "OUTDATED":
                continue

            formatted = format_comment(comment, thread_resolved=is_resolved)

            if is_resolved:
                resolved_comments.append(formatted)
            else:
                active_comments.append(formatted)

    return active_comments, resolved_comments


def main():
    parser = argparse.ArgumentParser(
        description="Fetch non-resolved PR review comments for the current branch"
    )
    parser.add_argument(
        "--repo", help="Repository in format owner/repo (auto-detected if not provided)"
    )
    parser.add_argument(
        "--pr-number",
        type=int,
        help="PR number (auto-detected from current branch if not provided)",
    )
    parser.add_argument(
        "--submodule", help="Submodule directory to use for branch detection"
    )
    parser.add_argument(
        "--include-resolved", action="store_true", help="Also show resolved comments"
    )
    parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )

    args = parser.parse_args()

    # Determine working directory
    cwd = Path.cwd()
    if args.submodule:
        cwd = cwd / args.submodule

    # Get repository info
    if args.repo:
        repo = args.repo
    else:
        try:
            repo = get_repo_from_remote(cwd=cwd)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            print("Please provide --repo owner/repo", file=sys.stderr)
            sys.exit(1)

    # Get PR number
    if args.pr_number:
        pr_number = args.pr_number
    else:
        branch = get_current_branch(cwd=cwd)
        print(f"Current branch: {branch}")
        pr_number = find_pr_for_branch(branch, repo)

        if not pr_number:
            print(f"No open PR found for branch '{branch}' in {repo}", file=sys.stderr)
            sys.exit(1)

    print(f"Repository: {repo}")
    print(f"PR: #{pr_number}")
    print(f"Fetching review comments...")
    print()

    # Fetch review data
    data = fetch_review_threads(pr_number, repo)
    active, resolved = process_review_data(data)

    # Output results
    if args.format == "json":
        output = {
            "repository": repo,
            "pr_number": pr_number,
            "active_comments": active,
            "resolved_comments": resolved if args.include_resolved else [],
        }
        print(json.dumps(output, indent=2))
    else:
        # Text format
        if active:
            print(f"=== ACTIVE/NON-RESOLVED COMMENTS ({len(active)} threads) ===")
            print()
            for comment_text in active:
                print(comment_text)
        else:
            print("No active (non-resolved) review comments found.")

        if args.include_resolved and resolved:
            print()
            print(f"=== RESOLVED COMMENTS ({len(resolved)} threads) ===")
            print()
            for comment_text in resolved:
                print(comment_text)

        # Summary
        print()
        print(
            f"Summary: {len(active)} active comments, {len(resolved)} resolved comments"
        )


if __name__ == "__main__":
    main()
