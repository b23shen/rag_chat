#!/usr/bin/env bash
# Build pgvector from source and install it into an EDB-installed PostgreSQL on macOS.
#
# EDB's pg_config points at a macOS SDK path from EDB's build machine, which doesn't
# exist locally, so PG_SYSROOT is set to the SDK from the Xcode Command Line Tools.
#
# Usage: bash scripts/install_pgvector.sh
# Env:   PG_CONFIG (default /Library/PostgreSQL/18/bin/pg_config), PGVECTOR_VERSION (default v0.8.7)
set -euo pipefail

PG_CONFIG="${PG_CONFIG:-/Library/PostgreSQL/18/bin/pg_config}"
VERSION="${PGVECTOR_VERSION:-v0.8.7}"

[ -x "$PG_CONFIG" ] || { echo "pg_config not found at $PG_CONFIG (set PG_CONFIG)"; exit 1; }
SDK="$(xcrun --show-sdk-path)" || { echo "Install the Xcode Command Line Tools: xcode-select --install"; exit 1; }

WORK="$(mktemp -d)"
echo "Building pgvector $VERSION for $("$PG_CONFIG" --version) in $WORK"
git clone --quiet --branch "$VERSION" --depth 1 https://github.com/pgvector/pgvector.git "$WORK/pgvector" 2>/dev/null
cd "$WORK/pgvector"

if ! make PG_CONFIG="$PG_CONFIG" PG_SYSROOT="$SDK" >"$WORK/build.log" 2>&1; then
  tail -20 "$WORK/build.log"
  echo "Build failed; full log: $WORK/build.log"
  exit 1
fi

echo "Installing into $("$PG_CONFIG" --pkglibdir). sudo will ask for your Mac password."
if ! sudo make install PG_CONFIG="$PG_CONFIG" PG_SYSROOT="$SDK" >"$WORK/install.log" 2>&1; then
  tail -20 "$WORK/install.log"
  echo "Install failed; full log: $WORK/install.log"
  exit 1
fi

echo "pgvector $VERSION installed. No restart needed. Next: bash scripts/init_db.sh"
