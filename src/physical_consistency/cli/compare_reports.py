"""Thin CLI wrapper around report aggregation."""

from __future__ import annotations

from physical_consistency.eval.aggregate import main as _main


def main() -> None:
    """Console-script compatible entrypoint."""
    _main()


if __name__ == "__main__":
    main()
