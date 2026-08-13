#!/usr/bin/env bash
# Copyright 2026 Alibaba Group.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Minimal CANN 8.5 Ascend operator profiler wrapper.
#
# Usage:
#   bash tools/profile_ascend.sh profile_driver.py --output-dir profiles/v1
#   PROFILE_DEVICE=npu:0 bash tools/profile_ascend.sh profile_driver.py -- --shape 1
#
# MSPROF_BIN and PROFILE_PYTHON may select non-default toolchain binaries. The
# CANN-version-dependent msprof files are kept intact below <output-dir>/msprof;
# this wrapper deliberately does not infer NCU-style counters from them.

set -euo pipefail

TARGET_FILE=""
OUTPUT_DIR="./profiles/v0"
TARGET_ARGS=()

usage() {
    cat <<'EOF'
Usage: profile_ascend.sh <python-target> [--output-dir DIR] [-- TARGET_ARGS...]

Environment:
    MSPROF_BIN       msprof executable (default: resolve msprof from PATH)
    PROFILE_PYTHON   Python executable used for the target (default: python3)
    PROFILE_DEVICE   torch device consumed by profile_driver.py (use npu:0)

Output:
    <output-dir>/msprof/                 raw CANN msprof op artifacts
    <output-dir>/msprof.log              profiler stdout/stderr
    <output-dir>/profile_summary.json    explicit structured-metrics status
    <output-dir>/artifact_manifest.json  raw artifact paths and sizes
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)
            if [[ $# -lt 2 ]]; then
                echo "profile_ascend.sh: --output-dir requires a value" >&2
                exit 2
            fi
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --)
            shift
            TARGET_ARGS=("$@")
            break
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "profile_ascend.sh: unknown option: $1" >&2
            exit 2
            ;;
        *)
            if [[ -n "$TARGET_FILE" ]]; then
                echo "profile_ascend.sh: unexpected argument: $1 (use -- before target arguments)" >&2
                exit 2
            fi
            TARGET_FILE="$1"
            shift
            ;;
    esac
done

if [[ -z "$TARGET_FILE" ]]; then
    echo "profile_ascend.sh: a Python target is required" >&2
    usage >&2
    exit 2
fi
if [[ ! -f "$TARGET_FILE" ]]; then
    echo "profile_ascend.sh: target does not exist: $TARGET_FILE" >&2
    exit 2
fi
if [[ -z "$OUTPUT_DIR" ]]; then
    echo "profile_ascend.sh: output directory must not be empty" >&2
    exit 2
fi

MSPROF_BIN_PATH="${MSPROF_BIN:-}"
if [[ -z "$MSPROF_BIN_PATH" ]]; then
    MSPROF_BIN_PATH="$(command -v msprof || true)"
fi
if [[ -z "$MSPROF_BIN_PATH" || ! -x "$MSPROF_BIN_PATH" ]]; then
    echo "profile_ascend.sh: msprof was not found; source the CANN 8.5 environment" >&2
    exit 127
fi

PROFILE_PYTHON_BIN="${PROFILE_PYTHON:-python3}"
if ! command -v "$PROFILE_PYTHON_BIN" >/dev/null 2>&1; then
    echo "profile_ascend.sh: Python executable was not found: $PROFILE_PYTHON_BIN" >&2
    exit 127
fi
export PROFILE_DEVICE="${PROFILE_DEVICE:-npu:0}"

mkdir -p -- "$OUTPUT_DIR"
MSPROF_ARTIFACT_DIR="$OUTPUT_DIR/msprof"
mkdir -p -- "$MSPROF_ARTIFACT_DIR"

echo "[profile_ascend] collecting CANN msprof op artifacts in $MSPROF_ARTIFACT_DIR"
set +e
"$MSPROF_BIN_PATH" op \
    --output="$MSPROF_ARTIFACT_DIR" \
    "$PROFILE_PYTHON_BIN" "$TARGET_FILE" "${TARGET_ARGS[@]}" \
    2>&1 | tee -- "$OUTPUT_DIR/msprof.log"
MSPROF_STATUS=${PIPESTATUS[0]}
set -e

"$PROFILE_PYTHON_BIN" - "$OUTPUT_DIR" "$MSPROF_STATUS" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
status = int(sys.argv[2])
summary = {
    "profiler": "msprof",
    "exit_code": status,
    "metrics_status": "unavailable",
    "artifact_root": "msprof",
    "note": (
        "Raw CANN msprof op artifacts are retained; structured metric parsing "
        "is unavailable because the schema is CANN-version dependent."
    ),
}
(root / "profile_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)

artifacts = []
for path in sorted(root.rglob("*")):
    if path.is_symlink() or not path.is_file() or path.name == "artifact_manifest.json":
        continue
    artifacts.append(
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
        }
    )
(root / "artifact_manifest.json").write_text(
    json.dumps({"profiler": "msprof", "artifacts": artifacts}, indent=2) + "\n",
    encoding="utf-8",
)
PY

echo "[profile_ascend] profile complete; structured metrics: unavailable"
exit "$MSPROF_STATUS"
