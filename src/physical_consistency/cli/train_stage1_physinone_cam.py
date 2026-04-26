"""CLI wrapper for pure Stage-1 PhysInOne camera training."""

from __future__ import annotations

from physical_consistency.stages.stage1_physinone_cam.runner import main as _main


def main() -> None:
    _main()


if __name__ == "__main__":
    main()
