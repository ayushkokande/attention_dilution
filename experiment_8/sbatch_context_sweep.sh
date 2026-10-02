#!/usr/bin/env bash
# Portable launcher for the maintained context stage.
# Pass the same arguments shown by: python -m attention_dilution context --help
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
exec "${PYTHON_BIN:-python}" -m attention_dilution context "$@"
