#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
ENV_FILE="${ROOT_DIR}/.env"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required but was not found in PATH." >&2
  exit 1
fi

python3 -m venv "${VENV_DIR}"

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
python -m pip install -e "${ROOT_DIR}"

if [[ ! -f "${ENV_FILE}" ]]; then
  cat >"${ENV_FILE}" <<'EOF'
OPENAI_API_KEY=
OPENAI_BASE_URL=
OPENAI_MODEL=
EOF
  echo "Created .env template at ${ENV_FILE}"
else
  echo ".env already exists at ${ENV_FILE}; leaving it unchanged."
fi

cat <<EOF
Bootstrap complete.

Next steps:
  edit .env and fill in OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL
  source .venv/bin/activate
  python -m weakness_driven_problem_synthesis.run --help
EOF
