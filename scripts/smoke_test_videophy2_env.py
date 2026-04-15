"""Quick VideoPhy-2 runtime preflight for the dedicated eval environment."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

BOOTSTRAP_PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = BOOTSTRAP_PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from physical_consistency.common.defaults import PROJECT_ROOT
from physical_consistency.common.path_config import resolve_path_config


def _default_videophy2_python(project_root: Path) -> str:
    candidate = (project_root / ".." / ".." / ".conda_envs" / "phys-videophy" / "bin" / "python").resolve()
    if candidate.exists():
        return str(candidate)
    return sys.executable


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str]) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test the dedicated VideoPhy-2 env.")
    parser.add_argument("--env_file", type=str, default=str(PROJECT_ROOT / "configs" / "path_config_cluster.env"))
    parser.add_argument("--python_bin", type=str, default="")
    parser.add_argument("--videophy_repo_dir", type=str, default="")
    args = parser.parse_args()

    path_cfg = resolve_path_config(env_file=args.env_file)
    project_root = PROJECT_ROOT
    python_bin = args.python_bin or os.environ.get("VIDEOPHY2_PYTHON") or _default_videophy2_python(project_root)
    repo_dir = Path(args.videophy_repo_dir or path_cfg.videophy_repo_dir)
    if not repo_dir.exists():
        repo_dir = project_root / "third_party" / "videophy"
    repo_entry = repo_dir / "VIDEOPHY2" / "inference.py"
    if not repo_entry.exists():
        raise FileNotFoundError(f"VideoPhy-2 repo not found: {repo_entry}")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{project_root / 'src'}:{env.get('PYTHONPATH', '')}".rstrip(":")

    _run(
        [
            python_bin,
            "-c",
            (
                "import sentencepiece, transformers, physical_consistency.cli.run_videophy2; "
                "print('python', __import__('sys').executable)"
            ),
        ],
        env=env,
    )
    _run([python_bin, "inference.py", "--help"], cwd=repo_entry.parent, env=env)
    print(f"[SMOKE OK] VideoPhy-2 runtime ready via {python_bin}")


if __name__ == "__main__":
    main()
