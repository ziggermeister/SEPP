#!/usr/bin/env bash
set -euo pipefail

echo "[*] Patching run_portfolios.py for case-insensitive presets..."
python - <<'PY'
from pathlib import Path
import re

f = Path("run_portfolios.py")
src = f.read_text()

pattern = re.compile(r"# --- Load scoring presets.*?eng\.WEIGHTS.*?print\(.*?\)", re.S)
replacement = """# --- Load scoring presets and pick one (case-insensitive) ---
import json
from pathlib import Path

def load_scoring_config(path="config/scoring.json"):
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Missing scoring config: {path}")
    try:
        return json.loads(p.read_text())
    except Exception as e:
        raise SystemExit(f"Failed to parse {path}: {e}")

scoring_cfg = load_scoring_config()
presets = scoring_cfg.get("presets", {})

if not presets:
    raise SystemExit("No presets found under 'presets' in config/scoring.json.")

# Build a case-insensitive map of preset names -> canonical key
ci_map = {k.casefold(): k for k in presets.keys()}

requested = (args.preset or "SEPP").strip()
canon_key = ci_map.get(requested.casefold())

if not canon_key:
    available = ", ".join(sorted(presets.keys()))
    raise SystemExit(
        f"Preset '{requested}' not found in config/scoring.json. "
        f"Available: {available}"
    )

# Override engine weights for this run
eng.WEIGHTS = presets[canon_key]
print(f"Using preset: {canon_key}")"""

new_src, n = pattern.subn(replacement, src)
if n == 0:
    raise SystemExit("Patch failed: could not find scoring preset block in run_portfolios.py")

f.write_text(new_src)
print("[+] run_portfolios.py patched.")
PY

echo "[*] Patching sepp_engine.py period_score for safe dict lookups..."
python - <<'PY'
from pathlib import Path
import re

f = Path("sepp_engine.py")
src = f.read_text()

pattern = re.compile(r"score\s*=\s*sum\(weights\[k\]\s*\*\s*parts\[k\]\s*for k in weights\.keys\(\)\)")
replacement = "score = sum(weights.get(k, 0.0) * parts.get(k, 0.0) for k in weights.keys())"

new_src, n = pattern.subn(replacement, src)
if n == 0:
    raise SystemExit("Patch failed: could not find period_score line in sepp_engine.py")

f.write_text(new_src)
print("[+] sepp_engine.py patched.")
PY

echo "[*] Running formatters..."
ruff check . --fix || true
black . || true
isort . || true

echo "[*] Done."
