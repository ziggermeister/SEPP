#!/usr/bin/env bash
# bootstrap.sh — non-interactive repo sync + deps install for Colab
set -euo pipefail

# ---------- Config (override via env if needed) ----------
: "${GITHUB_TOKEN:?Missing GITHUB_TOKEN env var}"
REPO_URL="${REPO_URL:-https://github.com/ziggermeister/SEPP.git}"
PROJECT_DIR="${PROJECT_DIR:-/content/SEPP}"
BRANCH="${BRANCH:-main}"

echo "[bootstrap] Repo: $REPO_URL"
echo "[bootstrap] Dir : $PROJECT_DIR"
echo "[bootstrap] Branch: $BRANCH"

# Avoid interactive git prompts and cached helpers
export GIT_TERMINAL_PROMPT=0
git config --global credential.helper ""

if [ -d "$PROJECT_DIR/.git" ]; then
  echo "[bootstrap] Repo exists — pulling latest…"
  cd "$PROJECT_DIR"
  git -c http.https://github.com/.extraheader="Authorization: Bearer ${GITHUB_TOKEN}" \
      -c http.https://codeload.github.com/.extraheader="Authorization: Bearer ${GITHUB_TOKEN}" \
      fetch origin
  git checkout "$BRANCH"
  git -c http.https://github.com/.extraheader="Authorization: Bearer ${GITHUB_TOKEN}" \
      -c http.https://codeload.github.com/.extraheader="Authorization: Bearer ${GITHUB_TOKEN}" \
      pull --rebase
else
  echo "[bootstrap] Cloning fresh…"
  rm -rf "$PROJECT_DIR"
  git -c http.https://github.com/.extraheader="Authorization: Bearer ${GITHUB_TOKEN}" \
      -c http.https://codeload.github.com/.extraheader="Authorization: Bearer ${GITHUB_TOKEN}" \
      clone --branch "$BRANCH" "$REPO_URL" "$PROJECT_DIR"
  cd "$PROJECT_DIR"
fi

# Set git identity for any commits you make from Colab
git config user.name  "Vivek Bhatnagar"
git config user.email "bhatnagar.vivek@gmail.com"

echo "[bootstrap] Installing Python deps…"
if [ -f requirements.txt ]; then
  pip install -q -r requirements.txt
else
  pip install -q numpy scipy pandas yfinance alpha_vantage matplotlib
fi

mkdir -p runs tests/param_packs data

echo "[bootstrap] Done. On branch: $(git branch --show-current)"
