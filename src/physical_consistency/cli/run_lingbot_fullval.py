"""Thin CLI wrapper around the chunked LingBot full-val pipeline."""

from __future__ import annotations

from physical_consistency.eval.lingbot_fullval import main as _main


def main() -> None:
    """Console-script compatible entrypoint."""
    _main()


if __name__ == "__main__":
    main()
