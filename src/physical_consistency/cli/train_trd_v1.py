"""Thin CLI wrapper for TRD-v1 training."""

from __future__ import annotations

from physical_consistency.trainers.trd_v1 import main as _main


def main() -> None:
    """Console-script compatible entrypoint."""
    _main()


if __name__ == "__main__":
    main()
