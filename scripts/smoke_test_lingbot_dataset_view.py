"""Smoke-test LingBot dataset-view materialization against eval_batch.py."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from physical_consistency.common.defaults import CONFIG_DIR, PROJECT_ROOT
from physical_consistency.common.io import write_csv_rows
from physical_consistency.common.path_config import resolve_path_config
from physical_consistency.datasets.manifest_builder import materialize_dataset_view


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test LingBot dataset view materialization.")
    parser.add_argument("--env_file", type=str, default=str(CONFIG_DIR / "path_config_cluster.env"))
    parser.add_argument("--eval_batch_path", type=str, default="")
    return parser.parse_args()


def _resolve_eval_batch_path(args: argparse.Namespace) -> Path:
    if args.eval_batch_path:
        candidate = Path(args.eval_batch_path)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"eval_batch.py not found at {candidate}")

    try:
        path_cfg = resolve_path_config(args, env_file=args.env_file)
    except Exception:
        path_cfg = None
    if path_cfg is not None:
        candidate = Path(path_cfg.finetune_code_dir) / "eval_batch.py"
        if candidate.exists():
            return candidate

    fallback = PROJECT_ROOT.parent / "code" / "finetune_v3" / "lingbot-csgo-finetune" / "eval_batch.py"
    if fallback.exists():
        return fallback
    raise FileNotFoundError("Could not resolve eval_batch.py; pass --eval_batch_path explicitly.")


def _load_eval_batch_module(eval_batch_path: Path):
    spec = importlib.util.spec_from_file_location("lingbot_eval_batch_smoke", eval_batch_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from {eval_batch_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    eval_batch_path = _resolve_eval_batch_path(args)
    eval_batch = _load_eval_batch_module(eval_batch_path)

    with tempfile.TemporaryDirectory(prefix="lingbot_view_smoke_") as tmpdir:
        tmp_root = Path(tmpdir)
        dataset_dir = tmp_root / "processed_csgo_v3"
        train_clip_dir = dataset_dir / "train" / "clips" / "clip_smoke_0000"
        train_clip_dir.mkdir(parents=True)
        (train_clip_dir / "image.jpg").write_text("image", encoding="utf-8")

        rows = [{"clip_path": "val/clips/clip_smoke_0000", "prompt": "smoke prompt"}]
        write_csv_rows(dataset_dir / "metadata_train.csv", rows, list(rows[0].keys()))
        write_csv_rows(dataset_dir / "metadata_val.csv", rows, list(rows[0].keys()))

        manifest_path = tmp_root / "manifest.csv"
        write_csv_rows(manifest_path, rows, list(rows[0].keys()))

        view_dir = materialize_dataset_view(dataset_dir, manifest_path, tmp_root / "view")
        clips = eval_batch.load_clip_list(str(view_dir), "val")
        if len(clips) != 1:
            raise RuntimeError(f"Expected 1 clip from eval_batch.load_clip_list(), got {len(clips)}")

        clip_dir = Path(clips[0]["clip_dir"])
        image_path = Path(eval_batch.ensure_first_frame(str(clip_dir), 480, 832))
        if not image_path.exists():
            raise FileNotFoundError(f"Smoke test failed: ensure_first_frame returned missing path {image_path}")

        print(
            "[SMOKE OK] LingBot dataset view resolved clip media:",
            f"clip_dir={clip_dir}",
            f"image_path={image_path}",
            f"eval_batch_path={eval_batch_path}",
        )


if __name__ == "__main__":
    main()
