"""CLI wrapper for Stage-2 Phy_CSGO action training."""

from __future__ import annotations

from physical_consistency.stages.stage2_phycsgo_act.runner import main as _main


def main() -> None:
    _main()


if __name__ == "__main__":
    main()
