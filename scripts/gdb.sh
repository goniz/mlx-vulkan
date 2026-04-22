#!/usr/bin/env bash
set -euo pipefail

PID="${1:-1053616}"

exec gdb -q -batch \
  -ex "set pagination off" \
  -ex "set print thread-events off" \
  -ex "attach ${PID}" \
  -ex "info threads" \
  -ex "thread apply all bt" \
  -ex "detach" \
  -ex "quit"
