"""Quick preflight for FLUX-based first-frame generation."""

from __future__ import annotations

import argparse
import importlib
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test the FLUX first-frame environment.")
    parser.add_argument("--model_family", type=str, default="flux2", choices=["flux1", "flux2"])
    args = parser.parse_args()

    required = ["torch", "transformers", "accelerate", "diffusers", "PIL"]
    for name in required:
        importlib.import_module(name if name != "PIL" else "PIL.Image")

    from diffusers import Flux2Pipeline, FluxPipeline

    if args.model_family == "flux2":
        _ = Flux2Pipeline
    else:
        _ = FluxPipeline

    print(f"[SMOKE OK] FLUX first-frame environment ready via {sys.executable}")


if __name__ == "__main__":
    main()
