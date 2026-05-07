#!/usr/bin/env bash
# Run a script inside the unitree-py39 conda environment.
#
# Usage:
#   ./scripts/run_unitree.sh setup              # create the env (first time)
#   ./scripts/run_unitree.sh scripts/test_b1_z1.py [args...]
#
# The script automatically sources the .env file so B1_SDK_PATH and network
# vars are available before any imports.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="unitree-py39"
ENV_FILE="$REPO_ROOT/envs/unitree.yml"
SDK_PATH="${B1_SDK_PATH:-/home/liam/installs/unitree_legged_sdk/lib/python/amd64}"

# ── helpers ──────────────────────────────────────────────────────────────────

die() { echo "ERROR: $*" >&2; exit 1; }

find_conda() {
    for candidate in \
        "$CONDA_EXE" \
        "$(command -v conda 2>/dev/null)" \
        "$HOME/miniconda3/bin/conda" \
        "$HOME/anaconda3/bin/conda" \
        "/opt/conda/bin/conda"
    do
        [ -x "$candidate" ] && echo "$candidate" && return
    done
    die "conda not found — install Miniconda or set CONDA_EXE"
}

CONDA="$(find_conda)"
CONDA_BASE="$("$CONDA" info --base)"

# ── setup subcommand ─────────────────────────────────────────────────────────

if [[ "${1:-}" == "setup" ]]; then
    echo "==> Creating conda env '$ENV_NAME' from $ENV_FILE"
    "$CONDA" env create -f "$ENV_FILE" --name "$ENV_NAME" || \
        "$CONDA" env update -f "$ENV_FILE" --name "$ENV_NAME"

    # Verify the SDK bindings are importable in the new env
    echo "==> Checking robot_interface.so is loadable"
    "$CONDA_BASE/envs/$ENV_NAME/bin/python" - <<EOF
import sys
sys.path.insert(0, "$SDK_PATH")
try:
    import robot_interface
    print("  OK — robot_interface loaded from $SDK_PATH")
except ImportError as e:
    print(f"  WARN — could not import robot_interface: {e}")
    print("  Make sure B1_SDK_PATH points to the directory containing robot_interface.cpython-39-*.so")
EOF
    echo "==> Setup complete.  Run with:"
    echo "    ./scripts/run_unitree.sh scripts/test_b1_z1.py"
    exit 0
fi

# ── run subcommand ────────────────────────────────────────────────────────────

SCRIPT="${1:-}"
[ -z "$SCRIPT" ] && die "Usage: $0 setup | $0 <script.py> [args...]"
shift

# Resolve to absolute path
[[ "$SCRIPT" != /* ]] && SCRIPT="$REPO_ROOT/$SCRIPT"
[ -f "$SCRIPT" ] || die "Script not found: $SCRIPT"

# Check env exists
"$CONDA" env list | grep -q "^$ENV_NAME " || \
    die "Env '$ENV_NAME' not found — run: $0 setup"

PYTHON="$CONDA_BASE/envs/$ENV_NAME/bin/python"
[ -x "$PYTHON" ] || die "Python not found in env: $PYTHON"

# Load .env into shell so child process inherits them
if [ -f "$REPO_ROOT/.env" ]; then
    set -o allexport
    # shellcheck disable=SC1090
    source <(grep -v '^\s*#' "$REPO_ROOT/.env" | grep -v '^\s*$')
    set +o allexport
fi

# Prepend SDK path and repo root to PYTHONPATH
export PYTHONPATH="$SDK_PATH:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "==> Running in $ENV_NAME: $SCRIPT $*"
exec "$PYTHON" "$SCRIPT" "$@"
