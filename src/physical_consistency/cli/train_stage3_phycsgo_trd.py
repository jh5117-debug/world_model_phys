"""CLI wrapper for Stage-3 Phy_CSGO TRD training."""

from __future__ import annotations

from physical_consistency.stages.stage3_phycsgo_trd.runner import main as _main


def main() -> None:
    _main()


if __name__ == "__main__":
    main()
