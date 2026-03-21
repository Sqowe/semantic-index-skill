#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

echo "Setting up semantic-index environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "Created virtual environment at $VENV_DIR"
fi

# Activate and install core dependencies
source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q
pip install -r "$SCRIPT_DIR/requirements.txt" -q

# Install HuggingFace dependencies if requested or if config says huggingface
if [ "${1:-}" = "--with-huggingface" ]; then
    echo "Installing HuggingFace local embedding dependencies..."
    pip install -r "$SCRIPT_DIR/requirements-huggingface.txt" -q
elif [ -f "${2:-.index/config.json}" ]; then
    # Auto-detect from config if it exists
    PROVIDER=$(python3 -c "
import json, sys
try:
    cfg = json.load(open(sys.argv[1]))
    print(cfg.get('embedding', {}).get('provider', ''))
except: pass
" "${2:-.index/config.json}" 2>/dev/null || true)
    if [ "$PROVIDER" = "huggingface" ]; then
        echo "Config uses HuggingFace provider. Installing local embedding dependencies..."
        pip install -r "$SCRIPT_DIR/requirements-huggingface.txt" -q
    fi
fi

echo "Setup complete. Dependencies installed."
echo "Virtual environment: $VENV_DIR"
