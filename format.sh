#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
find src/tenkai/templates -type f \
    \( -name '*.h' -o -name '*.hpp' -o -name '*.cu' -o -name '*.cuh' \) \
    -exec uv run --locked clang-format -style=Chromium -i {} +

uv run --locked ruff format src tests bench tools
uv run --locked ruff check src tests bench tools
