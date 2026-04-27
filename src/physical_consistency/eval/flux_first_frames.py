"""Generate reusable first-frame images from VideoPhy-2 prompt manifests."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from physical_consistency.common.io import read_csv_rows, write_csv_rows

TURBO_SIGMAS = [1.0, 0.6509, 0.4374, 0.2932, 0.1893, 0.1108, 0.0495, 0.00031]


@dataclass(slots=True)
class FluxFirstFrameJob:
    """One prompt-to-image generation job."""

    sample_id: str
    prompt: str
    clip_dir: Path
    image_path: Path
    prompt_mode: str
    is_hard: str
    source_manifest: str
    seed: int


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "y"}


def _stable_seed(sample_id: str, base_seed: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**31)


def build_first_frame_jobs(
    *,
    manifest_csv: str | Path,
    output_dir: str | Path,
    base_seed: int,
    image_filename: str = "image.jpg",
    max_samples: int = 0,
) -> list[FluxFirstFrameJob]:
    """Map a prompt manifest into deterministic per-sample image targets."""
    manifest_path = Path(manifest_csv).resolve()
    root = Path(output_dir).resolve()
    jobs: list[FluxFirstFrameJob] = []
    for idx, row in enumerate(read_csv_rows(manifest_path)):
        sample_id = (row.get("sample_id") or "").strip()
        prompt = (row.get("prompt") or "").strip()
        if not sample_id or not prompt:
            continue
        clip_dir = root / "samples" / sample_id
        jobs.append(
            FluxFirstFrameJob(
                sample_id=sample_id,
                prompt=prompt,
                clip_dir=clip_dir,
                image_path=clip_dir / image_filename,
                prompt_mode=(row.get("source_mode") or row.get("prompt_mode") or "").strip(),
                is_hard="1" if _truthy(row.get("is_hard")) else "0",
                source_manifest=str(manifest_path),
                seed=_stable_seed(sample_id, base_seed),
            )
        )
        if max_samples > 0 and len(jobs) >= max_samples:
            break
    return jobs


def load_flux_pipeline(
    *,
    model_id: str,
    turbo_lora_id: str = "",
    turbo_weight_name: str = "flux.2-turbo-lora.safetensors",
    torch_dtype_name: str = "bfloat16",
    device: str = "cuda:0",
    enable_model_cpu_offload: bool = False,
    disable_progress_bar: bool = True,
) -> Any:
    """Load a FLUX pipeline from diffusers using official model ids."""
    import torch

    try:
        from diffusers import Flux2Pipeline, FluxPipeline
    except ImportError as exc:
        raise ImportError(
            "diffusers with FLUX support is required. Install/upgrade diffusers before running "
            "first-frame generation."
        ) from exc

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if torch_dtype_name not in dtype_map:
        raise ValueError(f"Unsupported torch dtype name: {torch_dtype_name}")

    upper_model_id = model_id.upper()
    pipeline_cls = Flux2Pipeline if "FLUX.2" in upper_model_id else FluxPipeline
    pipeline = pipeline_cls.from_pretrained(
        model_id,
        torch_dtype=dtype_map[torch_dtype_name],
    )
    if turbo_lora_id:
        pipeline.load_lora_weights(
            turbo_lora_id,
            weight_name=turbo_weight_name,
        )
    if disable_progress_bar and hasattr(pipeline, "set_progress_bar_config"):
        pipeline.set_progress_bar_config(disable=True)
    if enable_model_cpu_offload and hasattr(pipeline, "enable_model_cpu_offload"):
        pipeline.enable_model_cpu_offload()
    elif hasattr(pipeline, "to"):
        pipeline.to(device)
    return pipeline


def _default_steps_and_guidance(*, model_id: str, turbo_lora_id: str) -> tuple[int, float]:
    if turbo_lora_id:
        return 8, 2.5
    if "FLUX.2" in model_id.upper():
        return 50, 4.0
    return 50, 3.5


def generate_first_frames(
    jobs: list[FluxFirstFrameJob],
    *,
    output_manifest_csv: str | Path,
    model_id: str,
    turbo_lora_id: str = "",
    turbo_weight_name: str = "flux.2-turbo-lora.safetensors",
    height: int,
    width: int,
    num_inference_steps: int = 0,
    guidance_scale: float = -1.0,
    torch_dtype_name: str = "bfloat16",
    device: str = "cuda:0",
    enable_model_cpu_offload: bool = False,
    skip_existing: bool = True,
    max_sequence_length: int = 0,
    pipeline: Any | None = None,
) -> list[dict[str, str]]:
    """Generate one first-frame image per prompt and write a reusable manifest."""
    import torch

    if not jobs:
        write_csv_rows(output_manifest_csv, [], [])
        return []

    if pipeline is None:
        pipeline = load_flux_pipeline(
            model_id=model_id,
            turbo_lora_id=turbo_lora_id,
            turbo_weight_name=turbo_weight_name,
            torch_dtype_name=torch_dtype_name,
            device=device,
            enable_model_cpu_offload=enable_model_cpu_offload,
        )

    default_steps, default_guidance = _default_steps_and_guidance(
        model_id=model_id,
        turbo_lora_id=turbo_lora_id,
    )
    steps = num_inference_steps if num_inference_steps > 0 else default_steps
    guide = guidance_scale if guidance_scale >= 0 else default_guidance

    rows: list[dict[str, str]] = []
    for job in jobs:
        job.clip_dir.mkdir(parents=True, exist_ok=True)
        if not (skip_existing and job.image_path.exists() and job.image_path.stat().st_size > 0):
            generator = torch.Generator(device=device).manual_seed(job.seed)
            kwargs: dict[str, Any] = {
                "prompt": job.prompt,
                "height": height,
                "width": width,
                "guidance_scale": guide,
                "num_inference_steps": steps,
                "generator": generator,
            }
            if max_sequence_length > 0:
                kwargs["max_sequence_length"] = max_sequence_length
            if turbo_lora_id:
                kwargs["sigmas"] = TURBO_SIGMAS
            result = pipeline(**kwargs)
            image = result.images[0]
            image.save(job.image_path, quality=95)
        rows.append(
            {
                "sample_id": job.sample_id,
                "prompt": job.prompt,
                "clip_dir": str(job.clip_dir),
                "image_path": str(job.image_path),
                "prompt_mode": job.prompt_mode,
                "is_hard": job.is_hard,
                "source_manifest": job.source_manifest,
                "seed": str(job.seed),
                "model_id": model_id,
                "turbo_lora_id": turbo_lora_id,
                "num_inference_steps": str(steps),
                "guidance_scale": f"{guide:.4f}",
                "height": str(height),
                "width": str(width),
            }
        )

    fieldnames = list(rows[0].keys()) if rows else []
    write_csv_rows(output_manifest_csv, rows, fieldnames)
    return rows
