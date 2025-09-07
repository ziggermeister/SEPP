#!/usr/bin/env bash
set -euo pipefail

# Create directories
mkdir -p vendor/vanguard vendor/spdr vendor/ishares

# Helper to (re)create a CSV with the default header
make_csv() {
  local path="$1"
  # only write if file doesn't exist, or force with --force
  if [[ ! -f "$path" || "${FORCE:-}" == "1" ]]; then
    printf "Date,Value\n" > "$path"
    echo "[OK] wrote header to $path"
  else
    echo "[SKIP] $path already exists (use FORCE=1 to overwrite)"
  fi
}

# Vanguard
make_csv vendor/vanguard/VTI_TR_or_NAV.csv
make_csv vendor/vanguard/BND_TR_or_NAV.csv
make_csv vendor/vanguard/VGIT_TR_or_NAV.csv

# SPDR
make_csv vendor/spdr/SCHD_TR_or_NAV.csv
make_csv vendor/spdr/GLD_TR_or_NAV.csv

# iShares
make_csv vendor/ishares/QQQ_TR_or_NAV.csv