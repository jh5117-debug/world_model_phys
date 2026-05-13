"""CLI wrapper for PhysInOne moving-camera act preprocessing."""

from __future__ import annotations

from physical_consistency.stages.stage1_physinone_cam.preprocess_moving_act import main as _main


def main() -> None:
    _main()


if __name__ == "__main__":
    main()
