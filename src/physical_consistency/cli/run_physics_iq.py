"""Thin CLI wrapper around Physics-IQ-style paired evaluation."""

from __future__ import annotations

from physical_consistency.eval.physics_iq import main as _main


def main() -> None:
    """Console-script compatible entrypoint."""
    _main()


if __name__ == "__main__":
    main()
