"""Thin CLI wrapper around VideoPhy-2 evaluation."""

from __future__ import annotations

from physical_consistency.eval.videophy2 import main as _main


def main() -> None:
    """Console-script compatible entrypoint."""
    _main()


if __name__ == "__main__":
    main()
