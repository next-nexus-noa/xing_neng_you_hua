#!/usr/bin/env bash
set -euo pipefail

# Power is sampled on the client side during the measured benchmark window.
# Keep the server lifecycle identical to the known-good gradient server sweep.
# The actual vLLM Ascend launch flags and environment setup live in
# run_gradient_server.sh in the same scripts directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_gradient_server.sh" "$@"
