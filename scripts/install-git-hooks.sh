#!/bin/sh

set -eu

repo_root=$(git rev-parse --show-toplevel)
hook_dir=$(git rev-parse --git-path hooks)

mkdir -p "$hook_dir"

cat > "$hook_dir/pre-commit" <<'EOF'
#!/bin/sh

set -eu

repo_root=$(git rev-parse --show-toplevel)
exec "$repo_root/scripts/check-reference-submodules.sh"
EOF

chmod +x "$hook_dir/pre-commit"

printf '%s\n' "Installed pre-commit hook at $hook_dir/pre-commit"
