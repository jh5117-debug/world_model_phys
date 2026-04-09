"""Lineage metadata and Stage-1 contract verification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from physical_consistency.common.git_utils import get_git_commit
from physical_consistency.common.io import write_json
from physical_consistency.eval.checkpoint_bundle import validate_dual_model_checkpoint


@dataclass(slots=True)
class LineageRecord:
    """Minimal provenance record for isolated physical-consistency runs."""

    experiment_group: str
    parent_stage: str
    parent_stage1_ckpt: str
    base_model_dir: str
    dataset_dir: str
    config_path: str
    config_hash: str
    created_at: str
    git_commit: str
    notes: str = ""

    @classmethod
    def create(
        cls,
        *,
        parent_stage1_ckpt: str,
        base_model_dir: str,
        dataset_dir: str,
        config_path: str,
        config_hash: str,
        project_root: str | Path,
        notes: str = "",
    ) -> "LineageRecord":
        """Build a lineage record with current timestamp and git commit."""
        return cls(
            experiment_group="physical_consistency",
            parent_stage="stage1",
            parent_stage1_ckpt=str(parent_stage1_ckpt),
            base_model_dir=str(base_model_dir),
            dataset_dir=str(dataset_dir),
            config_path=str(config_path),
            config_hash=config_hash,
            created_at=datetime.now(timezone.utc).isoformat(),
            git_commit=get_git_commit(project_root),
            notes=notes,
        )

    def write(self, output_path: str | Path) -> None:
        """Write lineage JSON to disk."""
        write_json(output_path, asdict(self))


def verify_stage1_checkpoint(path: str | Path) -> tuple[bool, list[str]]:
    """Verify the required Stage-1 checkpoint structure exists."""
    return validate_dual_model_checkpoint(path)
