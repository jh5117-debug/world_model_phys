"""CLI entrypoint for generation-only LingBot sharded inference."""

from physical_consistency.eval.lingbot_generate import main as _main


def main() -> None:
    _main()


if __name__ == "__main__":
    main()
