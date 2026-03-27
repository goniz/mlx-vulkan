#!/bin/sh

set -eu

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

unset GIT_DIR
unset GIT_WORK_TREE
unset GIT_INDEX_FILE
unset GIT_PREFIX

if ! git rev-parse --git-path modules >/dev/null 2>&1; then
    exit 0
fi

git diff --cached --quiet -- references || {
    printf '%s\n' 'ERROR: commits may not change gitlinks under references/.' >&2
    printf '%s\n' 'Revert the staged submodule update before committing.' >&2
    exit 1
}

dirty_submodules=''

for path in references/*; do
    [ -d "$path" ] || continue
    [ -e "$path/.git" ] || continue

    if ! git -C "$path" diff --quiet --ignore-submodules=all || \
       ! git -C "$path" diff --cached --quiet --ignore-submodules=all || \
       [ -n "$(git -C "$path" ls-files --others --exclude-standard)" ]; then
        dirty_submodules="$dirty_submodules
$path"
    fi
done

if [ -n "$dirty_submodules" ]; then
    printf '%s\n' 'ERROR: references submodules contain local file changes:' >&2
    printf '%s\n' "$dirty_submodules" | sed '/^$/d' >&2
    printf '%s\n' 'Clean or discard those changes before committing.' >&2
    exit 1
fi

exit 0
