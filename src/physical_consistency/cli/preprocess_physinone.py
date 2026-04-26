"""CLI wrapper for PhysInOne preprocessing."""

from __future__ import annotations

from physical_consistency.stages.stage1_physinone_cam.preprocess import main as _main


def main() -> None:
    _main()


if __name__ == "__main__":
    main()
