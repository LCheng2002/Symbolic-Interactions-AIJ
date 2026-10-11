#!/usr/bin/env bash
# Download the interaction data (~2.8 GB) from Zenodo and verify integrity.
#
# Usage:
#   ./download_data.sh
#
# The record id is the numeric part of the Zenodo DOI
# (10.5281/zenodo.<record-id>); the published record for this package is
# https://doi.org/10.5281/zenodo.22052714
set -euo pipefail

PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIZ_DIR="$PKG_ROOT/visualization"
TARBALL="interactions.tar.gz"

ZENODO_RECORD="${ZENODO_RECORD:-22052714}"
URL="https://zenodo.org/records/${ZENODO_RECORD}/files/${TARBALL}?download=1"

cd "$PKG_ROOT"
echo "[1/3] downloading $URL"
curl -L --fail --retry 3 -o "$TARBALL" "$URL"

echo "[2/3] extracting into visualization/"
tar -xzf "$TARBALL" -C "$VIZ_DIR"

echo "[3/3] verifying checksums (this takes a few minutes)"
cd "$VIZ_DIR"
sha256sum --check CHECKSUMS.sha256

echo "Done. Interaction data is ready under visualization/interactions/."
