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

# Parse arguments: consume known flags, first non-flag arg is config path
INSTALL_HF=false
INSTALL_OFFICE=false
CONFIG_PATH=""

for arg in "$@"; do
    case "$arg" in
        --with-huggingface) INSTALL_HF=true ;;
        --with-office) INSTALL_OFFICE=true ;;
        -*)
            echo "Unknown flag: $arg" >&2
            exit 1
            ;;
        *)
            # First non-flag argument is the config path
            if [ -z "$CONFIG_PATH" ]; then
                CONFIG_PATH="$arg"
            fi
            ;;
    esac
done

# Default config path if not provided
CONFIG_PATH="${CONFIG_PATH:-.index/config.json}"

# Install HuggingFace dependencies if requested or if config says huggingface
if [ "$INSTALL_HF" = true ]; then
    echo "Installing HuggingFace local embedding dependencies..."
    pip install -r "$SCRIPT_DIR/requirements-huggingface.txt" -q
elif [ -f "$CONFIG_PATH" ]; then
    PROVIDER=$(python3 -c "
import json, sys
try:
    cfg = json.load(open(sys.argv[1]))
    print(cfg.get('embedding', {}).get('provider', ''))
except: pass
" "$CONFIG_PATH" 2>/dev/null || true)
    if [ "$PROVIDER" = "huggingface" ]; then
        echo "Config uses HuggingFace provider. Installing local embedding dependencies..."
        pip install -r "$SCRIPT_DIR/requirements-huggingface.txt" -q
    fi
fi

# Install office document dependencies if requested
if [ "$INSTALL_OFFICE" = true ]; then
    echo "Installing office document extraction dependencies (PDF, DOCX, PPTX)..."
    pip install -r "$SCRIPT_DIR/requirements-office.txt" -q
fi

echo "Setup complete. Dependencies installed."
echo "Virtual environment: $VENV_DIR"
