#!/usr/bin/env python
"""Generate reusable first-frame images from a VideoPhy-2 prompt manifest."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

BOOTSTRAP_PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = BOOTSTRAP_PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from physical_consistency.common.defaults import PROJECT_ROOT
from physical_consistency.eval.flux_first_frames import (
    build_first_frame_jobs,
    generate_first_frames,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate FLUX first-frame images from prompt CSV.")
    parser.add_argument("--manifest_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_manifest_csv", type=str, default="")
    parser.add_argument("--model_id", type=str, default="black-forest-labs/FLUX.2-dev")
    parser.add_argument("--turbo_lora_id", type=str, default="fal/FLUX.2-dev-Turbo")
    parser.add_argument("--turbo_weight_name", type=str, default="flux.2-turbo-lora.safetensors")
    parser.add_argument("--disable_turbo", action="store_true")
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_inference_steps", type=int, default=0)
    parser.add_argument("--guidance_scale", type=float, default=-1.0)
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--enable_model_cpu_offload", action="store_true")
    parser.add_argument("--image_filename", type=str, default="image.jpg")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_sequence_length", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_manifest_csv = (
        Path(args.output_manifest_csv).resolve()
        if args.output_manifest_csv
        else output_dir / "generated_first_frames.csv"
    )
    turbo_lora_id = "" if args.disable_turbo else args.turbo_lora_id

    jobs = build_first_frame_jobs(
        manifest_csv=Path(args.manifest_csv).resolve(),
        output_dir=output_dir,
        base_seed=args.base_seed,
        image_filename=args.image_filename,
        max_samples=args.max_samples,
    )
    rows = generate_first_frames(
        jobs,
        output_manifest_csv=output_manifest_csv,
        model_id=args.model_id,
        turbo_lora_id=turbo_lora_id,
        turbo_weight_name=args.turbo_weight_name,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        torch_dtype_name=args.torch_dtype,
        device=args.device,
        enable_model_cpu_offload=args.enable_model_cpu_offload,
        skip_existing=args.skip_existing,
        max_sequence_length=args.max_sequence_length,
    )

    logging.info("Generated %d first-frame images -> %s", len(rows), output_dir)
    logging.info("Manifest -> %s", output_manifest_csv)
    logging.info("Model id -> %s", args.model_id)
    if turbo_lora_id:
        logging.info("Turbo LoRA -> %s", turbo_lora_id)
    else:
        logging.info("Turbo disabled")
    logging.info(
        "This script is intended for conditioned VideoPhy-style evaluation, not standard prompt-only VideoPhy-2."
    )


if __name__ == "__main__":
    main()
