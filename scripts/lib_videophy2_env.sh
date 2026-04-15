#!/usr/bin/env bash

resolve_videophy2_python() {
  local project_root="$1"
  local env_python="${VIDEOPHY2_PYTHON:-}"
  local default_candidate=""

  default_candidate="$(cd "${project_root}/../.." && pwd)/.conda_envs/phys-videophy/bin/python"

  if [[ -n "${env_python}" ]]; then
    printf '%s\n' "${env_python}"
    return 0
  fi
  if [[ -x "${default_candidate}" ]]; then
    printf '%s\n' "${default_candidate}"
    return 0
  fi
  command -v python
}

verify_videophy2_python() {
  local python_bin="$1"
  local project_root="$2"
  local resolved_python=""

  if [[ "${python_bin}" == */* ]]; then
    if [[ ! -x "${python_bin}" ]]; then
      echo "[ERROR] VideoPhy-2 python is not executable: ${python_bin}" >&2
      return 1
    fi
    resolved_python="${python_bin}"
  else
    if ! resolved_python="$(command -v "${python_bin}")"; then
      echo "[ERROR] VideoPhy-2 python not found on PATH: ${python_bin}" >&2
      return 1
    fi
  fi

  PYTHONPATH="${project_root}/src:${PYTHONPATH:-}" "${resolved_python}" - <<'PY'
import importlib
mods = ["sentencepiece", "transformers", "physical_consistency.cli.run_videophy2"]
for name in mods:
    importlib.import_module(name)
PY
}
