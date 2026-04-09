"""Thin CLI wrapper around the CSGO metrics suite."""

from __future__ import annotations

from physical_consistency.eval.csgo_metrics import main as _main


def main() -> None:
    """Console-script compatible entrypoint."""
    _main()


if __name__ == "__main__":
    main()
