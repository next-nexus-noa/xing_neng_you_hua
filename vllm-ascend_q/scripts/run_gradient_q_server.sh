#!/usr/bin/env bash
set -euo pipefail

# Power is sampled on the client side during the measured benchmark window.
# The server lifecycle is identical to the gradient server sweep, so keep this
# as a thin entrypoint for a matched run_gradient_power_client.sh workflow.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_gradient_server.sh" "$@"
