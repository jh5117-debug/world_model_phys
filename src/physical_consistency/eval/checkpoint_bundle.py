"""Helpers for building explicit dual-model eval bundles."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from physical_consistency.common.io import ensure_dir, write_json

MODEL_BRANCHES = ("low_noise_model", "high_noise_model")


def validate_dual_model_checkpoint(path: str | Path) -> tuple[bool, list[str]]:
    """Validate that both dual-model branches are present and loadable."""
    root = Path(path)
    errors: list[str] = []
    if not root.exists():
        errors.append(f"Checkpoint path does not exist: {root}")
        return False, errors

    for branch in MODEL_BRANCHES:
        branch_dir = root / branch
        if not branch_dir.exists():
            errors.append(f"Missing directory: {branch_dir}")
            continue
        config_path = branch_dir / "config.json"
        weight_bin = branch_dir / "diffusion_pytorch_model.bin"
        if not config_path.exists():
            errors.append(f"Missing config: {config_path}")
        if not weight_bin.exists():
            errors.append(f"Missing weights: {weight_bin}")
    return len(errors) == 0, errors


def materialize_eval_checkpoint_bundle(
    *,
    ft_ckpt_dir: str | Path,
    output_root: str | Path,
    experiment_name: str,
    stage1_ckpt_dir: str | Path = "",
    companion_ckpt_dir: str | Path = "",
    allow_stage1_fallback: bool = False,
) -> Path:
    """Create a full two-branch eval directory and make branch provenance explicit."""
    ft_root = Path(ft_ckpt_dir).resolve()
    stage1_root = Path(stage1_ckpt_dir).resolve() if stage1_ckpt_dir else None
    companion_root = Path(companion_ckpt_dir).resolve() if companion_ckpt_dir else None

    bundle_key = hashlib.sha256(
        "|".join(
            [
                str(ft_root),
                str(stage1_root) if stage1_root else "",
                str(companion_root) if companion_root else "",
                experiment_name,
                str(int(allow_stage1_fallback)),
            ]
        ).encode("utf-8")
    ).hexdigest()[:16]
    bundle_dir = (
        Path(output_root)
        / "cache"
        / "eval_ckpt_bundles"
        / f"{experiment_name}_{bundle_key}"
    )
    ensure_dir(bundle_dir)

    manifest: dict[str, dict[str, str]] = {}
    for branch in MODEL_BRANCHES:
        source_dir = ft_root / branch
        source_kind = "finetuned"
        if not source_dir.exists():
            if companion_root is not None:
                source_dir = companion_root / branch
                source_kind = "companion"
            elif allow_stage1_fallback and stage1_root is not None:
                source_dir = stage1_root / branch
                source_kind = "stage1_fallback"
            else:
                raise FileNotFoundError(
                    "Incomplete fine-tuned checkpoint bundle. "
                    f"Missing {branch} under {ft_root}. "
                    "Refusing to silently fall back to the base model."
                )

        branch_config = source_dir / "config.json"
        branch_weight = source_dir / "diffusion_pytorch_model.bin"
        if not branch_config.exists() or not branch_weight.exists():
            raise FileNotFoundError(
                f"Branch {branch} under {source_dir} is incomplete for evaluation."
            )

        dst = bundle_dir / branch
        if dst.is_symlink() or dst.exists():
            if dst.is_dir() and not dst.is_symlink():
                for child in dst.iterdir():
                    if child.is_symlink() or child.is_file():
                        child.unlink()
                    elif child.is_dir():
                        raise RuntimeError(f"Unexpected directory inside existing bundle: {child}")
                dst.rmdir()
            else:
                dst.unlink()
        os.symlink(source_dir, dst, target_is_directory=True)
        manifest[branch] = {
            "source": str(source_dir),
            "source_kind": source_kind,
        }

    bundle_manifest = {
        "experiment_name": experiment_name,
        "ft_ckpt_dir": str(ft_root),
        "stage1_ckpt_dir": str(stage1_root) if stage1_root else "",
        "companion_ckpt_dir": str(companion_root) if companion_root else "",
        "allow_stage1_fallback": allow_stage1_fallback,
        "branches": manifest,
    }
    write_json(bundle_dir / "bundle_manifest.json", bundle_manifest)

    ok, errors = validate_dual_model_checkpoint(bundle_dir)
    if not ok:
        raise FileNotFoundError("\n".join(errors))
    return bundle_dir
