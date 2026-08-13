#!/usr/bin/env sh
# modelo - single-file installer for macOS / Linux
#
#   Usage:  curl -fsSL <base-url>/install.sh | sh
#
# Downloads the one-file "modelo" program and installs it as a global command
# in ${HOME}/.local/bin (or $MODELO_INSTALL_DIR), then installs the Python
# dependencies it needs.
set -e

# ---------------------------------------------------------------------------
# Override the raw base URL that hosts install.sh and modelo.py
#   - GitHub repo:   https://raw.githubusercontent.com/USER/REPO/main
#   - Gist:          https://gist.githubusercontent.com/USER/GIST_ID/raw
# Set MODELO_URL in your environment to override the value below.
# ---------------------------------------------------------------------------
BASE_URL="${MODELO_URL:-https://raw.githubusercontent.com/Merunism19/modelo/main}"

# Where the "modelo" executable lands (edit or override via MODELO_INSTALL_DIR).
INSTALL_DIR="${MODELO_INSTALL_DIR:-$HOME/.local/bin}"

# --- Check prerequisites -----------------------------------------------------
command -v curl >/dev/null 2>&1 || { echo "error: curl is required"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "error: python3 is required"; exit 1; }

# --- Download the program ----------------------------------------------------
mkdir -p "$INSTALL_DIR"
echo ">> Downloading modelo to $INSTALL_DIR/modelo"
curl -fsSL "$BASE_URL/modelo.py" -o "$INSTALL_DIR/modelo"
chmod +x "$INSTALL_DIR/modelo"

# --- Install Python dependencies ---------------------------------------------
echo ">> Installing Python dependencies (typer, huggingface_hub, requests)"
if ! (pip install -q --user typer huggingface_hub requests 2>/dev/null \
      || pip3 install -q --user typer huggingface_hub requests 2>/dev/null); then
    echo "warning: could not auto-install Python deps."
    echo "         run manually:  pip install --user typer huggingface_hub requests"
fi

# --- Done --------------------------------------------------------------------
cat <<EOF

modelo installed.
  - Make sure $INSTALL_DIR is on your PATH:
      echo 'export PATH="\$HOME/.local/bin:\$PATH"' >> ~/.bashrc && source ~/.bashrc
  - Try it:
      modelo --help
  - Optional (to serve models):
      pip install --user llama-cpp-python uvicorn
EOF