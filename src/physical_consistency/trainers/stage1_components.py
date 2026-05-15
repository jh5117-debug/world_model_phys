"""Reusable Stage-1 helpers for isolated physical-consistency training."""

from __future__ import annotations

import contextlib
import csv
import importlib
import logging
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader, Dataset

LOGGER = logging.getLogger(__name__)
_SDPA_MATH_LOGGED = False
_NUMERIC_AUDIT_COUNTERS: dict[str, int] = {}


def _should_log_rank_zero() -> bool:
    rank = os.environ.get("RANK", "")
    return rank in {"", "0"}


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        LOGGER.warning("Invalid integer for %s=%r; using %s", name, raw, default)
        return int(default)


def _numeric_audit_enabled(name: str) -> bool:
    return _env_flag("PC_NUMERIC_AUDIT") or _env_flag(name)


def _audit_tensor_numerics(
    label: str,
    tensor: torch.Tensor,
    *,
    key: str,
    enabled: bool,
    limit_env: str = "PC_NUMERIC_AUDIT_LIMIT",
) -> None:
    if not enabled or not torch.is_tensor(tensor) or not tensor.is_floating_point():
        return
    limit = _env_int(limit_env, 16)
    count = _NUMERIC_AUDIT_COUNTERS.get(key, 0)
    if count >= limit:
        return
    _NUMERIC_AUDIT_COUNTERS[key] = count + 1

    with torch.no_grad():
        detached = tensor.detach()
        finite = bool(torch.isfinite(detached).all().item())
        if detached.numel() == 0:
            max_abs = 0.0
            mean_abs = 0.0
        else:
            detached_float = detached.float()
            abs_values = detached_float.abs()
            max_abs = float(
                torch.nan_to_num(abs_values, nan=float("inf"), posinf=float("inf"), neginf=float("inf")).max().item()
            )
            mean_abs = float(torch.nan_to_num(abs_values, nan=0.0, posinf=0.0, neginf=0.0).mean().item())

    if _should_log_rank_zero():
        LOGGER.info(
            "PC_NUMERIC_AUDIT label=%s shape=%s dtype=%s device=%s requires_grad=%s finite=%s "
            "max_abs=%.6g mean_abs=%.6g",
            label,
            tuple(tensor.shape),
            tensor.dtype,
            tensor.device,
            tensor.requires_grad,
            finite,
            max_abs,
            mean_abs,
        )
    if not finite:
        raise FloatingPointError(
            f"Non-finite tensor detected during numeric audit: {label} "
            f"shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}"
        )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.expanduser())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path.expanduser())
    return deduped


def _existing_wan_import_root(path: Path) -> bool:
    return (path / "wan" / "modules" / "model.py").is_file()


def _candidate_lingbot_import_roots(
    configured_dir: str | os.PathLike[str] | None,
    *,
    project_root: Path | None = None,
) -> list[Path]:
    repo_root = Path(project_root or _repo_root()).resolve()
    explicit_candidates: list[Path] = []
    fallback_candidates: list[Path] = []

    def add_variants(bucket: list[Path], raw: str | os.PathLike[str] | None) -> None:
        if raw in {"", None}:
            return
        base = Path(raw).expanduser()
        bucket.extend(
            [
                base,
                base / "lingbot-world",
                base / "code" / "lingbot-world",
            ]
        )

    add_variants(explicit_candidates, configured_dir)
    env_lingbot_code_dir = os.environ.get("LINGBOT_CODE_DIR", "")
    if env_lingbot_code_dir and env_lingbot_code_dir != str(configured_dir or ""):
        add_variants(explicit_candidates, env_lingbot_code_dir)

    add_variants(fallback_candidates, repo_root / "links" / "lingbot_code")
    add_variants(fallback_candidates, repo_root / "third_party" / "lingbot_restore" / "code" / "lingbot-world")
    add_variants(
        fallback_candidates,
        repo_root.parents[1] / "code" / "lingbot-world" if len(repo_root.parents) > 1 else None,
    )

    swap_candidates: list[Path] = []
    for candidate in fallback_candidates:
        candidate_str = str(candidate)
        for src, dst in (("nvme03", "nvme04"), ("nvme04", "nvme03")):
            needle = f"/{src}/"
            if needle not in candidate_str:
                continue
            swapped = candidate_str.replace(needle, f"/{dst}/", 1)
            add_variants(swap_candidates, swapped)
            swap_candidates.append(Path(swapped).expanduser())

    fallback_candidates.extend(swap_candidates)
    if explicit_candidates:
        return _dedupe_paths(explicit_candidates + fallback_candidates)
    return _dedupe_paths(fallback_candidates)


def _resolve_lingbot_import_root(
    configured_dir: str | os.PathLike[str] | None,
    *,
    project_root: Path | None = None,
) -> tuple[Path, list[str]]:
    attempted: list[str] = []
    for candidate in _candidate_lingbot_import_roots(configured_dir, project_root=project_root):
        attempted.append(str(candidate))
        if _existing_wan_import_root(candidate):
            return candidate.resolve(), attempted

    attempted_list = "\n  - ".join(attempted) if attempted else "<none>"
    raise FileNotFoundError(
        "Unable to locate a LingBot checkout that exposes wan/modules/model.py.\n"
        f"Configured lingbot_code_dir={configured_dir!r}\n"
        f"Checked candidate import roots:\n  - {attempted_list}\n"
        "Set LINGBOT_CODE_DIR to the LingBot repo root (the directory that contains wan/)."
    )


def _normalize_stage1_precision_profile(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"", "auto", "native_lowp", "native", "lowp"}:
        return "native_lowp"
    if normalized in {"mixed_safe", "safe_mixed", "safe"}:
        return "mixed_safe"
    if normalized in {"fp32", "float32", "full_fp32"}:
        return "fp32"
    LOGGER.warning("Unknown Stage1 precision profile %r; falling back to native_lowp", value)
    return "native_lowp"


def _normalize_stage1_lowp_dtype_name(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"", "bf16", "bfloat16"}:
        return "bf16"
    if normalized in {"fp16", "float16", "half"}:
        return "fp16"
    LOGGER.warning("Unknown Stage1 low-precision dtype %r; falling back to bf16", value)
    return "bf16"


def resolve_stage1_precision_profile() -> str:
    if _env_flag("PC_STAGE1_FORCE_FP32"):
        return "fp32"
    return _normalize_stage1_precision_profile(os.environ.get("PC_STAGE1_PRECISION_PROFILE"))


def resolve_stage1_low_precision_dtype() -> torch.dtype:
    normalized = _normalize_stage1_lowp_dtype_name(os.environ.get("PC_STAGE1_LOWP_DTYPE"))
    return torch.float16 if normalized == "fp16" else torch.bfloat16


def _stage1_force_fp32() -> bool:
    return resolve_stage1_precision_profile() == "fp32"


def _stage1_mixed_safe() -> bool:
    return resolve_stage1_precision_profile() == "mixed_safe"


def _attention_compute_dtype(default_dtype: torch.dtype) -> torch.dtype:
    if _env_flag("PC_FORCE_ATTN_FP32") or _stage1_force_fp32():
        return torch.float32
    return default_dtype


def configure_stage1_precision_env(
    profile: str | None = None,
    lowp_dtype: str | None = None,
) -> dict[str, str]:
    """Materialize a stable Stage-1 precision policy via environment defaults."""

    requested_profile = _normalize_stage1_precision_profile(
        profile if profile not in {"", None} else os.environ.get("PC_STAGE1_PRECISION_PROFILE")
    )
    requested_lowp_dtype = _normalize_stage1_lowp_dtype_name(
        lowp_dtype if lowp_dtype not in {"", None} else os.environ.get("PC_STAGE1_LOWP_DTYPE")
    )

    if _env_flag("PC_STAGE1_FORCE_FP32"):
        requested_profile = "fp32"

    os.environ["PC_STAGE1_PRECISION_PROFILE"] = requested_profile
    os.environ["PC_STAGE1_LOWP_DTYPE"] = requested_lowp_dtype

    defaults: dict[str, str] = {}
    if requested_profile == "fp32":
        defaults.update(
            {
                "PC_STAGE1_FORCE_FP32": "1",
                "PC_VAE_FORCE_FP32": "1",
                "PC_FORCE_LORA_FP32": "1",
                "PC_LORA_DISABLE_AUTOCAST": "1",
                "PC_FORCE_SDPA_FALLBACK": "1",
                "PC_FORCE_SDPA_MATH": "1",
                "PC_FORCE_ATTN_FP32": "1",
            }
        )
    elif requested_profile == "mixed_safe":
        defaults.update(
            {
                "PC_VAE_FORCE_FP32": "1",
                "PC_FORCE_LORA_FP32": "1",
                "PC_LORA_DISABLE_AUTOCAST": "1",
                "PC_FORCE_SDPA_FALLBACK": "1",
            }
        )

    for name, value in defaults.items():
        os.environ.setdefault(name, value)

    effective = {
        "profile": resolve_stage1_precision_profile(),
        "lowp_dtype": _normalize_stage1_lowp_dtype_name(os.environ.get("PC_STAGE1_LOWP_DTYPE")),
        "force_fp32": str(_stage1_force_fp32()),
        "force_sdpa_fallback": str(_env_flag("PC_FORCE_SDPA_FALLBACK")),
        "force_sdpa_math": str(_env_flag("PC_FORCE_SDPA_MATH")),
        "force_attn_fp32": str(_env_flag("PC_FORCE_ATTN_FP32")),
        "force_lora_fp32": str(_env_flag("PC_FORCE_LORA_FP32")),
        "disable_lora_autocast": str(_env_flag("PC_LORA_DISABLE_AUTOCAST")),
        "force_vae_fp32": str(_env_flag("PC_VAE_FORCE_FP32")),
    }
    if _should_log_rank_zero():
        LOGGER.info("Effective Stage1 precision policy: %s", effective)
    return effective


@contextlib.contextmanager
def _sdpa_kernel_context():
    """Optionally force PyTorch SDPA fallback to its math backend."""
    if not _env_flag("PC_FORCE_SDPA_MATH"):
        yield
        return

    global _SDPA_MATH_LOGGED
    if not _SDPA_MATH_LOGGED and _should_log_rank_zero():
        LOGGER.info(
            "Using PyTorch SDPA math backend for Stage1 attention "
            "(PC_FORCE_SDPA_MATH=1)"
        )
        _SDPA_MATH_LOGGED = True

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except Exception:
        sdpa_kernel = None
        SDPBackend = None

    if sdpa_kernel is not None and SDPBackend is not None:
        with sdpa_kernel(SDPBackend.MATH):
            yield
        return

    cuda_backends = getattr(torch.backends, "cuda", None)
    if cuda_backends is not None and hasattr(cuda_backends, "sdp_kernel"):
        with cuda_backends.sdp_kernel(
            enable_flash=False,
            enable_mem_efficient=False,
            enable_math=True,
        ):
            yield
        return

    yield


def _env_list_allows(name: str, value: str | None) -> bool:
    spec = os.environ.get(name, "").strip()
    if not spec or spec.lower() in {"all", "*"}:
        return True
    if value is None:
        return False
    allowed = {part.strip() for part in spec.split(",") if part.strip()}
    return value in allowed


def _wan_trace_enabled(block: torch.nn.Module | None = None) -> bool:
    if not _env_flag("PC_WAN_BLOCK_TRACE"):
        return False
    rank = os.environ.get("RANK")
    local_rank = os.environ.get("LOCAL_RANK")
    if not (
        _env_list_allows("PC_WAN_BLOCK_TRACE_RANKS", rank)
        or _env_list_allows("PC_WAN_BLOCK_TRACE_RANKS", local_rank)
    ):
        return False
    if block is None:
        return True
    block_index = getattr(block, "_pc_wan_block_index", None)
    return _env_list_allows(
        "PC_WAN_BLOCK_TRACE_BLOCKS",
        None if block_index is None else str(block_index),
    )


def _wan_trace_tensor(tensor: torch.Tensor) -> str:
    return (
        f"shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"device={tensor.device} requires_grad={tensor.requires_grad}"
    )


def _wan_trace_memory(device: torch.device | None) -> str:
    if device is None or device.type != "cuda" or not torch.cuda.is_available():
        return ""
    try:
        allocated = torch.cuda.memory_allocated(device) / (1024**3)
        reserved = torch.cuda.memory_reserved(device) / (1024**3)
        max_allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
    except Exception:
        return ""
    return f" mem_alloc={allocated:.2f}GiB mem_reserved={reserved:.2f}GiB mem_max={max_allocated:.2f}GiB"


def _wan_trace(
    event: str,
    *,
    block: torch.nn.Module | None = None,
    tensors: dict[str, torch.Tensor] | None = None,
    detail: str = "",
    sync: bool = False,
) -> None:
    if not _wan_trace_enabled(block):
        return

    device = None
    if tensors:
        device = next((tensor.device for tensor in tensors.values() if torch.is_tensor(tensor)), None)
    if sync and _env_flag("PC_WAN_BLOCK_TRACE_SYNC") and device is not None and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)

    rank = os.environ.get("RANK", "?")
    local_rank = os.environ.get("LOCAL_RANK", "?")
    block_index = getattr(block, "_pc_wan_block_index", "?") if block is not None else "?"
    parts = [
        "[PC_WAN_TRACE]",
        f"rank={rank}",
        f"local_rank={local_rank}",
        f"pid={os.getpid()}",
        f"block={block_index}",
        f"event={event}",
    ]
    if detail:
        parts.append(detail)
    if tensors:
        parts.extend(f"{name}:{_wan_trace_tensor(tensor)}" for name, tensor in tensors.items())
    parts.append(_wan_trace_memory(device))
    print(" ".join(part for part in parts if part), file=sys.stderr, flush=True)


def _patch_wan_rope_apply_to_preserve_dtype(wan_model_module) -> None:
    """Patch Wan RoPE to avoid a large fp32 q/k tensor before flash-attn."""

    original_rope_apply = getattr(wan_model_module, "rope_apply", None)
    if original_rope_apply is None:
        LOGGER.warning("Wan model module has no rope_apply; skipping RoPE dtype patch")
        return
    if getattr(wan_model_module, "_pc_rope_preserve_dtype_patched", False):
        return

    def _rope_apply_preserve_dtype(x: torch.Tensor, grid_sizes: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        out_dtype = x.dtype
        device_type = x.device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            n, c = x.size(2), x.size(3) // 2
            split_freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

            output = []
            for i, (f, h, w) in enumerate(grid_sizes.tolist()):
                f, h, w = int(f), int(h), int(w)
                seq_len = f * h * w

                x_i = torch.view_as_complex(
                    x[i, :seq_len].to(torch.float32).reshape(seq_len, n, -1, 2)
                )
                freqs_i = torch.cat(
                    [
                        split_freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                        split_freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                        split_freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                    ],
                    dim=-1,
                ).reshape(seq_len, 1, -1)

                x_i = torch.view_as_real(x_i * freqs_i.to(torch.complex64)).flatten(2)
                if seq_len < x.size(1):
                    x_i = torch.cat([x_i, x[i, seq_len:].to(dtype=x_i.dtype)], dim=0)
                output.append(x_i.to(out_dtype))
            return torch.stack(output)

    wan_model_module._pc_original_rope_apply = original_rope_apply
    wan_model_module.rope_apply = _rope_apply_preserve_dtype
    wan_model_module._pc_rope_preserve_dtype_patched = True
    if _should_log_rank_zero():
        LOGGER.info("Patched Wan rope_apply to preserve q/k dtype for training memory")


def _patch_flash_attention_sdpa_fallback(wan_attention_module, wan_model_module=None) -> None:
    """Replace Wan flash_attention with SDPA fallback when requested.

    Wan's model module imports flash_attention by value, so patching only
    wan.modules.attention is not enough after wan.modules.model has loaded.
    """
    if not (_env_flag("PC_FORCE_SDPA_FALLBACK") or _stage1_mixed_safe()):
        return

    sdpa_fn = getattr(wan_attention_module, "_sdpa_fallback", None)
    if sdpa_fn is None:
        LOGGER.warning(
            "PC_FORCE_SDPA_FALLBACK=1 but wan.modules.attention has no "
            "_sdpa_fallback; cannot patch flash_attention"
        )
        return

    patched_flash_attention = getattr(wan_attention_module, "flash_attention")
    if not getattr(wan_attention_module, "_pc_sdpa_fallback_patched", False):
        original_flash_attention = patched_flash_attention

        def _flash_attention_via_sdpa(
            q, k, v,
            q_lens=None, k_lens=None,
            dropout_p=0., softmax_scale=None, q_scale=None,
            causal=False, window_size=(-1, -1),
            deterministic=False, dtype=torch.bfloat16, version=None,
        ):
            compute_dtype = _attention_compute_dtype(dtype)
            with _sdpa_kernel_context():
                return sdpa_fn(
                    q=q, k=k, v=v,
                    q_lens=q_lens, k_lens=k_lens,
                    dropout_p=dropout_p, softmax_scale=softmax_scale,
                    q_scale=q_scale, causal=causal, dtype=compute_dtype,
                )

        wan_attention_module._pc_original_flash_attention = original_flash_attention
        wan_attention_module.flash_attention = _flash_attention_via_sdpa
        wan_attention_module._pc_sdpa_fallback_patched = True
        patched_flash_attention = _flash_attention_via_sdpa

    patched_targets = ["wan.modules.attention"]
    if wan_model_module is not None and hasattr(wan_model_module, "flash_attention"):
        if getattr(wan_model_module, "flash_attention") is not patched_flash_attention:
            wan_model_module._pc_original_flash_attention = wan_model_module.flash_attention
            wan_model_module.flash_attention = patched_flash_attention
        wan_model_module._pc_sdpa_fallback_patched = True
        patched_targets.append("wan.modules.model")

    if _should_log_rank_zero():
        LOGGER.info(
            "Patched %s flash_attention -> SDPA fallback "
            "(flash-attn backward will NOT be used; compute_dtype=%s)",
            "+".join(patched_targets),
            _attention_compute_dtype(resolve_stage1_low_precision_dtype()),
        )


def _patch_sdpa_fallback_precision(wan_attention_module) -> None:
    """Make the SDPA fallback run in fp32 when the Stage-1 stability profile asks for it."""

    original_sdpa_fallback = getattr(wan_attention_module, "_sdpa_fallback", None)
    if original_sdpa_fallback is None:
        return
    if getattr(wan_attention_module, "_pc_sdpa_precision_patched", False):
        return

    def _sdpa_fallback_with_precision(
        q,
        k,
        v,
        q_lens=None,
        k_lens=None,
        dropout_p=0.,
        softmax_scale=None,
        q_scale=None,
        causal=False,
        dtype=torch.bfloat16,
    ):
        return original_sdpa_fallback(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            dtype=_attention_compute_dtype(dtype),
        )

    wan_attention_module._pc_original_sdpa_fallback = original_sdpa_fallback
    wan_attention_module._sdpa_fallback = _sdpa_fallback_with_precision
    wan_attention_module._pc_sdpa_precision_patched = True
    if _should_log_rank_zero():
        LOGGER.info(
            "Patched wan.modules.attention._sdpa_fallback for Stage1 precision policy "
            "(compute_dtype=%s)",
            _attention_compute_dtype(resolve_stage1_low_precision_dtype()),
        )


MODEL_SUBFOLDERS = ("low_noise_model", "high_noise_model")
MODEL_TYPE_TO_SUBFOLDER = {
    "low": "low_noise_model",
    "high": "high_noise_model",
}


class CSGODataset(Dataset):
    """Load preprocessed CSGO clips for training or validation."""

    def __init__(
        self,
        dataset_dir: str,
        *,
        split: str = "train",
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
        repeat: int = 1,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.repeat = repeat
        self.keep_aspect = _env_flag("PC_CSGO_KEEP_ASPECT")

        csv_path = os.path.join(dataset_dir, f"metadata_{split}.csv")
        with open(csv_path, "r", encoding="utf-8") as handle:
            self.samples = list(csv.DictReader(handle))
        if not self.samples:
            raise ValueError(f"No samples found in {csv_path}")
        LOGGER.info(
            "Loaded %s %s samples (repeat=%s, size=%sx%s, keep_aspect=%s)",
            len(self.samples),
            split,
            repeat,
            self.height,
            self.width,
            self.keep_aspect,
        )

    def __len__(self) -> int:
        return len(self.samples) * self.repeat

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        import cv2

        sample = self.samples[index % len(self.samples)]
        clip_dir = os.path.join(self.dataset_dir, sample["clip_path"])
        video_path = os.path.join(clip_dir, "video.mp4")

        cap = cv2.VideoCapture(video_path)
        frames = []
        source_height = None
        source_width = None
        while len(frames) < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if source_height is None or source_width is None:
                source_height = int(frame.shape[0])
                source_width = int(frame.shape[1])
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self._resize_frame(frame)
            frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 127.5 - 1.0
            frames.append(frame)
        cap.release()
        while len(frames) < self.num_frames:
            frames.append(frames[-1].clone())
        video_tensor = torch.stack(frames, dim=1)

        poses = np.load(os.path.join(clip_dir, "poses.npy"))
        actions = np.load(os.path.join(clip_dir, "action.npy"))
        intrinsics = np.load(os.path.join(clip_dir, "intrinsics.npy"))

        return {
            "clip_name": os.path.basename(clip_dir),
            "video": video_tensor,
            "prompt": sample["prompt"],
            "poses": torch.from_numpy(self._pad_or_truncate(poses)).float(),
            "actions": torch.from_numpy(self._pad_or_truncate(actions)).float(),
            "intrinsics": torch.from_numpy(self._pad_or_truncate(intrinsics)).float(),
            "source_height": int(source_height or self.height),
            "source_width": int(source_width or self.width),
        }

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        import cv2

        if not self.keep_aspect:
            return cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LANCZOS4)

        src_h, src_w = frame.shape[:2]
        if src_h <= 0 or src_w <= 0:
            raise ValueError(f"Invalid video frame shape: {frame.shape}")
        scale = min(float(self.width) / float(src_w), float(self.height) / float(src_h))
        new_w = min(self.width, max(1, int(round(src_w * scale))))
        new_h = min(self.height, max(1, int(round(src_h * scale))))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
        resized = cv2.resize(frame, (new_w, new_h), interpolation=interpolation)

        canvas = np.zeros((self.height, self.width, frame.shape[2]), dtype=frame.dtype)
        top = max((self.height - new_h) // 2, 0)
        left = max((self.width - new_w) // 2, 0)
        canvas[top : top + new_h, left : left + new_w] = resized
        return canvas

    def _pad_or_truncate(self, array: np.ndarray) -> np.ndarray:
        if len(array) >= self.num_frames:
            return array[: self.num_frames]
        rep_shape = (self.num_frames - len(array),) + (1,) * (array.ndim - 1)
        pad = np.tile(array[-1:], rep_shape)
        return np.concatenate([array, pad], axis=0)


@dataclass(slots=True)
class TimestepSample:
    """One sampled timestep record."""

    index: int
    sigma: float
    timestep: torch.Tensor
    weight: float
    branch: str


class LingBotStage1Helper:
    """Load models and shared components the same way as Stage-1 training."""

    def __init__(self, args) -> None:
        self.args = args
        self.device = torch.device("cpu")
        self.boundary = 0.947
        self.num_train_timesteps = 1000
        self.vae_stride = (4, 8, 8)
        self.patch_size = (1, 2, 2)
        self.shift = 10.0
        self._t5_cache: dict[str, list[torch.Tensor]] = {}
        self.keep_t5_on_gpu = _env_flag("PC_STAGE1_KEEP_T5_ON_GPU")
        self._build_schedule()

    def _build_schedule(self) -> None:
        sigmas_linear = torch.linspace(1.0, 0.0, self.num_train_timesteps + 1)[:-1]
        self.sigmas = self.shift * sigmas_linear / (1 + (self.shift - 1) * sigmas_linear)
        self.timesteps_schedule = self.sigmas * self.num_train_timesteps
        max_timestep = self.boundary * self.num_train_timesteps
        self.low_noise_indices = torch.where(self.timesteps_schedule < max_timestep)[0]
        self.high_noise_indices = torch.where(self.timesteps_schedule >= max_timestep)[0]

        y = torch.exp(-2 * ((self.timesteps_schedule - self.num_train_timesteps / 2) / self.num_train_timesteps) ** 2)
        y_shifted = y - y.min()
        training_weights = y_shifted * (self.num_train_timesteps / y_shifted.sum())
        self.all_indices = torch.arange(self.num_train_timesteps)
        self.all_weights = training_weights / training_weights.mean()
        self.low_noise_weights = training_weights[self.low_noise_indices] / training_weights[self.low_noise_indices].mean()
        self.high_noise_weights = training_weights[self.high_noise_indices] / training_weights[self.high_noise_indices].mean()

    def bootstrap_imports(self) -> None:
        """Import LingBot modules lazily from the shared code checkout."""
        start_time = time.perf_counter()
        configured_dir = getattr(self.args, "lingbot_code_dir", "")
        LOGGER.info("Stage1 helper: resolving LingBot import root (configured=%r)", configured_dir)
        lingbot_root, attempted = _resolve_lingbot_import_root(configured_dir)
        self.args.lingbot_code_dir = str(lingbot_root)
        if str(lingbot_root) not in sys.path:
            sys.path.insert(0, str(lingbot_root))
        importlib.invalidate_caches()
        LOGGER.info(
            "Resolved LingBot import root to %s (configured=%r, attempted=%s)",
            lingbot_root,
            configured_dir,
            attempted,
        )
        LOGGER.info("Stage1 helper: importing Wan runtime modules from %s", lingbot_root)
        import wan.modules.model as wan_model_module
        from wan.modules.model import WanModel
        from wan.modules.t5 import T5EncoderModel
        from wan.modules.vae2_1 import Wan2_1_VAE
        from wan.utils.cam_utils import (
            compute_relative_poses,
            get_Ks_transformed,
            get_plucker_embeddings,
            interpolate_camera_poses,
        )

        _patch_wan_rope_apply_to_preserve_dtype(wan_model_module)
        import wan.modules.attention as wan_attention_module
        _patch_sdpa_fallback_precision(wan_attention_module)
        _patch_flash_attention_sdpa_fallback(wan_attention_module, wan_model_module)
        self.WanModel = WanModel
        self.T5EncoderModel = T5EncoderModel
        self.Wan2_1_VAE = Wan2_1_VAE
        self.cam_utils = {
            "interpolate_camera_poses": interpolate_camera_poses,
            "compute_relative_poses": compute_relative_poses,
            "get_plucker_embeddings": get_plucker_embeddings,
            "get_Ks_transformed": get_Ks_transformed,
        }
        LOGGER.info(
            "Stage1 helper: Wan runtime imports ready in %.2fs",
            time.perf_counter() - start_time,
        )

    def ensure_runtime_components(self, device: torch.device) -> None:
        """Load the shared VAE/T5 runtime once."""
        LOGGER.info("Stage1 helper: ensuring runtime components on %s", device)
        self.bootstrap_imports()
        self.device = device
        if getattr(self, "vae", None) is None:
            start_time = time.perf_counter()
            LOGGER.info(
                "Stage1 helper: initializing VAE from %s",
                os.path.join(self.args.base_model_dir, "Wan2.1_VAE.pth"),
            )
            self.vae = self.Wan2_1_VAE(
                vae_pth=os.path.join(self.args.base_model_dir, "Wan2.1_VAE.pth"),
                device=self.device,
            )
            LOGGER.info("Stage1 helper: VAE ready in %.2fs", time.perf_counter() - start_time)
            if _env_flag("PC_VAE_FORCE_FP32"):
                if hasattr(self.vae, "dtype"):
                    self.vae.dtype = torch.float32
                module = getattr(self.vae, "model", None)
                if module is not None and hasattr(module, "float"):
                    module.float()
                elif hasattr(self.vae, "float"):
                    self.vae.float()
                LOGGER.info("Forced VAE runtime to fp32 for numerical stability (PC_VAE_FORCE_FP32=1)")
        if getattr(self, "t5", None) is None:
            # Keep T5 on CPU by default and only move it to GPU on cache misses.
            start_time = time.perf_counter()
            LOGGER.info(
                "Stage1 helper: initializing T5 from %s",
                os.path.join(self.args.base_model_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
            )
            self.t5 = self.T5EncoderModel(
                text_len=512,
                dtype=torch.bfloat16,
                device=torch.device("cpu"),
                checkpoint_path=os.path.join(self.args.base_model_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
                tokenizer_path=os.path.join(self.args.base_model_dir, "google", "umt5-xxl"),
            )
            module = getattr(self.t5, "model", None)
            if module is not None and hasattr(module, "to"):
                if self.keep_t5_on_gpu:
                    module.to(self.device)
                    LOGGER.info("Keeping T5 model on %s (PC_STAGE1_KEEP_T5_ON_GPU=1)", self.device)
                else:
                    module.to("cpu")
            LOGGER.info("Stage1 helper: T5 ready in %.2fs", time.perf_counter() - start_time)

    def load_model(
        self,
        device: torch.device,
        model_type: str,
        checkpoint_dir: str | Path | None = None,
        *,
        control_type: str = "act",
    ):
        """Load one target Stage-1 branch plus shared VAE/T5 runtime."""
        LOGGER.info(
            "Stage1 helper: load_model start (model_type=%s checkpoint_dir=%s control_type=%s device=%s)",
            model_type,
            checkpoint_dir or self.args.stage1_ckpt_dir,
            control_type,
            device,
        )
        self.ensure_runtime_components(device)
        subfolder = get_model_subfolder(model_type)
        checkpoint_root = str(checkpoint_dir or self.args.stage1_ckpt_dir)
        control_type = str(control_type).strip().lower()
        if control_type not in {"act", "cam"}:
            raise ValueError(f"Unsupported control_type: {control_type}")
        LOGGER.info("Loading %s from %s", subfolder, checkpoint_root)
        model_dtype = torch.float32 if _stage1_force_fp32() else resolve_stage1_low_precision_dtype()
        start_time = time.perf_counter()
        model = self.WanModel.from_pretrained(
            checkpoint_root,
            subfolder=subfolder,
            torch_dtype=model_dtype,
            control_type=control_type,
        )
        LOGGER.info(
            "Stage1 helper: WanModel.from_pretrained finished in %.2fs (subfolder=%s dtype=%s)",
            time.perf_counter() - start_time,
            subfolder,
            model_dtype,
        )
        if _stage1_force_fp32() and hasattr(model, "float"):
            model.float()
            LOGGER.info("Forced Stage1 student model to fp32 for numerical stability (PC_STAGE1_FORCE_FP32=1)")
        if getattr(self.args, "student_memory_efficient_modulation", True):
            start_time = time.perf_counter()
            LOGGER.info("Stage1 helper: applying memory-efficient modulation patch to %s", subfolder)
            apply_memory_efficient_wan_block_patch(
                model,
                subfolder,
                ffn_chunk_size=getattr(self.args, "student_ffn_chunk_size", None),
                norm_chunk_size=getattr(self.args, "student_norm_chunk_size", None),
            )
            LOGGER.info(
                "Stage1 helper: memory-efficient modulation patch ready in %.2fs",
                time.perf_counter() - start_time,
            )
        if getattr(self.args, "student_tuning_mode", "full") == "lora":
            start_time = time.perf_counter()
            LOGGER.info("Stage1 helper: applying LoRA adapters to %s", subfolder)
            apply_lora_to_wan_model(
                model,
                model_name=subfolder,
                rank=getattr(self.args, "student_lora_rank", 16),
                alpha=getattr(self.args, "student_lora_alpha", 16),
                dropout=getattr(self.args, "student_lora_dropout", 0.0),
                block_start=getattr(self.args, "student_lora_block_start", 0),
                lora_chunk_size=getattr(self.args, "student_lora_chunk_size", None),
                merge_mode=getattr(self.args, "student_lora_merge_mode", "inplace"),
            )
            LOGGER.info("Stage1 helper: LoRA adapters ready in %.2fs", time.perf_counter() - start_time)
        model.train()
        LOGGER.info("Stage1 helper: load_model complete for %s", subfolder)
        return model

    def release_runtime_components(self) -> None:
        """Drop non-trainable helper runtime objects to free GPU memory."""
        for attr_name in ("vae", "t5"):
            obj = getattr(self, attr_name, None)
            if obj is None:
                continue
            module = getattr(obj, "model", None)
            if module is not None and hasattr(module, "to"):
                try:
                    module.to("cpu")
                except Exception:
                    LOGGER.debug("Failed to move %s.model to CPU before release", attr_name, exc_info=True)
            elif hasattr(obj, "to"):
                try:
                    obj.to("cpu")
                except Exception:
                    LOGGER.debug("Failed to move %s to CPU before release", attr_name, exc_info=True)
            setattr(self, attr_name, None)
        self._t5_cache.clear()
        self.device = torch.device("cpu")

    @torch.no_grad()
    def encode_video(self, video_tensor: torch.Tensor) -> torch.Tensor:
        vae_numeric_audit = _numeric_audit_enabled("PC_VAE_NUMERIC_AUDIT")
        _audit_tensor_numerics(
            "vae_encode_input",
            video_tensor,
            key="vae",
            enabled=vae_numeric_audit,
        )
        if _env_flag("PC_VAE_FORCE_FP32"):
            video_tensor = video_tensor.float()
            device_type = self.device.type
            with torch.amp.autocast(device_type=device_type, enabled=False):
                latent = self.vae.encode([video_tensor.to(self.device)])[0]
        else:
            latent = self.vae.encode([video_tensor.to(self.device)])[0]
        _audit_tensor_numerics(
            "vae_encode_latent",
            latent,
            key="vae",
            enabled=vae_numeric_audit,
        )
        return latent

    @torch.no_grad()
    def encode_text(self, prompt: str) -> list[torch.Tensor]:
        force_fp32 = _stage1_force_fp32()
        if prompt in self._t5_cache:
            cached = [tensor.to(self.device) for tensor in self._t5_cache[prompt]]
            if force_fp32:
                cached = [tensor.float() for tensor in cached]
            return cached
        self.t5.model.to(self.device)
        context = self.t5([prompt], self.device)
        if not self.keep_t5_on_gpu:
            self.t5.model.cpu()
        self._t5_cache[prompt] = [tensor.cpu() for tensor in context]
        outputs = [tensor.to(self.device) for tensor in context]
        if force_fp32:
            outputs = [tensor.float() for tensor in outputs]
        return outputs

    @torch.no_grad()
    def prepare_y(self, video_tensor: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        vae_numeric_audit = _numeric_audit_enabled("PC_VAE_NUMERIC_AUDIT")
        lat_h, lat_w = latent.shape[2], latent.shape[3]
        frame_total = video_tensor.shape[1]
        height, width = video_tensor.shape[2], video_tensor.shape[3]
        first_frame = video_tensor[:, 0:1]
        zeros = torch.zeros(3, frame_total - 1, height, width, device=video_tensor.device)
        y_input = torch.concat([first_frame, zeros], dim=1)
        _audit_tensor_numerics(
            "vae_prepare_y_input",
            y_input,
            key="vae",
            enabled=vae_numeric_audit,
        )
        if _env_flag("PC_VAE_FORCE_FP32"):
            y_input = y_input.float()
            device_type = self.device.type
            with torch.amp.autocast(device_type=device_type, enabled=False):
                y_latent = self.vae.encode([y_input.to(self.device)])[0]
        else:
            y_latent = self.vae.encode([y_input.to(self.device)])[0]
        _audit_tensor_numerics(
            "vae_prepare_y_latent",
            y_latent,
            key="vae",
            enabled=vae_numeric_audit,
        )

        # Wan consumes a 4-channel temporal mask aligned to the latent timeline, not the raw frame count.
        mask = torch.zeros(4, y_latent.shape[1], lat_h, lat_w, device=self.device, dtype=y_latent.dtype)
        mask[:, 0] = 1
        return torch.concat([mask, y_latent])

    @torch.no_grad()
    def prepare_control_signal(
        self,
        poses: torch.Tensor,
        actions: torch.Tensor | None,
        intrinsics: torch.Tensor,
        height: int,
        width: int,
        lat_f: int,
        lat_h: int,
        lat_w: int,
        *,
        control_type: str = "act",
        source_height: int | None = None,
        source_width: int | None = None,
    ) -> dict[str, tuple[torch.Tensor, ...]]:
        interpolate_camera_poses = self.cam_utils["interpolate_camera_poses"]
        compute_relative_poses = self.cam_utils["compute_relative_poses"]
        get_plucker_embeddings = self.cam_utils["get_plucker_embeddings"]
        get_Ks_transformed = self.cam_utils["get_Ks_transformed"]

        control_numeric_audit = _numeric_audit_enabled("PC_CONTROL_NUMERIC_AUDIT")
        control_trace = control_numeric_audit or _env_flag("PC_CONTROL_TRACE")

        def _trace_control(label: str, **values) -> None:
            if not control_trace:
                return
            rank = os.environ.get("RANK", "?")
            details = " ".join(f"{key}={value}" for key, value in values.items())
            LOGGER.info("[CONTROL TRACE] rank=%s label=%s %s", rank, label, details)

        control_type = str(control_type).strip().lower()
        if control_type not in {"act", "cam"}:
            raise ValueError(f"Unsupported control_type: {control_type}")
        num_frames = poses.shape[0]
        source_height = int(source_height or height)
        source_width = int(source_width or width)
        _trace_control(
            "start",
            control_type=control_type,
            num_frames=num_frames,
            height=height,
            width=width,
            source_height=source_height,
            source_width=source_width,
            lat_f=lat_f,
            lat_h=lat_h,
            lat_w=lat_w,
        )
        _audit_tensor_numerics("control_poses", poses, key="control", enabled=control_numeric_audit)
        _audit_tensor_numerics("control_intrinsics", intrinsics, key="control", enabled=control_numeric_audit)
        if actions is not None:
            _audit_tensor_numerics("control_actions", actions, key="control", enabled=control_numeric_audit)

        ks = get_Ks_transformed(
            intrinsics,
            height_org=source_height,
            width_org=source_width,
            height_resize=height,
            width_resize=width,
            height_final=height,
            width_final=width,
        )
        _trace_control("after_get_Ks_transformed", ks_shape=tuple(ks.shape), ks_dtype=ks.dtype, ks_device=ks.device)
        _audit_tensor_numerics("control_ks", ks, key="control", enabled=control_numeric_audit)
        ks_single = ks[0]
        c2ws_infer = interpolate_camera_poses(
            src_indices=np.linspace(0, num_frames - 1, num_frames),
            src_rot_mat=poses[:, :3, :3].cpu().numpy(),
            src_trans_vec=poses[:, :3, 3].cpu().numpy(),
            tgt_indices=np.linspace(0, num_frames - 1, lat_f),
        )
        _trace_control("after_interpolate_camera_poses", c2ws_shape=tuple(c2ws_infer.shape))
        c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)
        _trace_control(
            "after_compute_relative_poses",
            c2ws_shape=tuple(c2ws_infer.shape),
            c2ws_dtype=c2ws_infer.dtype,
            c2ws_device=c2ws_infer.device,
        )
        _audit_tensor_numerics("control_c2ws_relative", c2ws_infer, key="control", enabled=control_numeric_audit)
        ks_repeated = ks_single.repeat(len(c2ws_infer), 1).to(self.device)
        c2ws_infer = c2ws_infer.to(self.device)
        cond_dtype = (
            torch.float32
            if (_stage1_force_fp32() or _stage1_mixed_safe())
            else resolve_stage1_low_precision_dtype()
        )

        only_rays_d = control_type == "act"
        plucker = get_plucker_embeddings(
            c2ws_infer,
            ks_repeated,
            height,
            width,
            only_rays_d=only_rays_d,
        )
        _trace_control(
            "after_get_plucker_embeddings",
            plucker_shape=tuple(plucker.shape),
            plucker_dtype=plucker.dtype,
            plucker_device=plucker.device,
        )
        _audit_tensor_numerics("control_plucker_raw", plucker, key="control", enabled=control_numeric_audit)
        plucker = rearrange(
            plucker,
            "f (h c1) (w c2) c -> (f h w) (c c1 c2)",
            c1=int(height // lat_h),
            c2=int(width // lat_w),
        )[None]
        plucker = rearrange(plucker, "b (f h w) c -> b c f h w", f=lat_f, h=lat_h, w=lat_w).to(cond_dtype)
        _trace_control(
            "after_plucker_rearrange",
            plucker_shape=tuple(plucker.shape),
            plucker_dtype=plucker.dtype,
            plucker_device=plucker.device,
        )
        _audit_tensor_numerics("control_plucker", plucker, key="control", enabled=control_numeric_audit)
        if control_type == "cam":
            return {"c2ws_plucker_emb": (plucker,)}

        if actions is None:
            raise ValueError("actions must be provided when control_type='act'")
        action_indices = np.linspace(0, len(actions) - 1, len(c2ws_infer)).round().astype(int)
        wasd = actions[action_indices].to(self.device)
        _trace_control("after_action_select", wasd_shape=tuple(wasd.shape), wasd_dtype=wasd.dtype, wasd_device=wasd.device)
        _audit_tensor_numerics("control_wasd", wasd, key="control", enabled=control_numeric_audit)

        wasd_tensor = wasd[:, None, None, :].repeat(1, height, width, 1)
        wasd_tensor = rearrange(
            wasd_tensor,
            "f (h c1) (w c2) c -> (f h w) (c c1 c2)",
            c1=int(height // lat_h),
            c2=int(width // lat_w),
        )[None]
        wasd_tensor = rearrange(
            wasd_tensor,
            "b (f h w) c -> b c f h w",
            f=lat_f,
            h=lat_h,
            w=lat_w,
        ).to(cond_dtype)
        _trace_control(
            "after_wasd_rearrange",
            wasd_shape=tuple(wasd_tensor.shape),
            wasd_dtype=wasd_tensor.dtype,
            wasd_device=wasd_tensor.device,
        )
        _audit_tensor_numerics("control_wasd_tensor", wasd_tensor, key="control", enabled=control_numeric_audit)

        control = torch.cat([plucker, wasd_tensor], dim=1)
        _trace_control(
            "done",
            control_shape=tuple(control.shape),
            control_dtype=control.dtype,
            control_device=control.device,
        )
        _audit_tensor_numerics("control_concat", control, key="control", enabled=control_numeric_audit)
        return {"c2ws_plucker_emb": (control,)}

    def sample_timestep(self, model_type: str) -> TimestepSample:
        """Sample a valid noise step for the selected model."""
        if model_type == "dual":
            indices = self.all_indices
            weights = self.all_weights
        elif model_type == "high":
            indices = self.high_noise_indices
            weights = self.high_noise_weights
        else:
            indices = self.low_noise_indices
            weights = self.low_noise_weights
        local_idx = torch.randint(len(indices), (1,)).item()
        idx = int(indices[local_idx].item())
        sigma = float(self.sigmas[idx].item())
        timestep = self.timesteps_schedule[idx].to(self.device).unsqueeze(0)
        weight = float(weights[local_idx].item())
        return TimestepSample(
            index=idx,
            sigma=sigma,
            timestep=timestep,
            weight=weight,
            branch=self.branch_for_timestep_index(idx),
        )

    def branch_for_timestep_index(self, timestep_index: int) -> str:
        """Return the model branch implied by one sampled timestep."""
        timestep_value = float(self.timesteps_schedule[timestep_index].item())
        boundary = self.boundary * self.num_train_timesteps
        return "high" if timestep_value >= boundary else "low"


def apply_gradient_checkpointing(
    model,
    model_name: str = "model",
    *,
    use_reentrant: bool = False,
    skip_block_indices: set[int] | None = None,
    memory_efficient_mode: str = "full",
) -> None:
    """Apply the same DiT block checkpointing strategy as Stage-1."""
    from functools import wraps
    import torch.utils.checkpoint as checkpoint_module

    patched = 0
    inner_patched = 0
    memory_efficient_patched = 0
    memory_efficient_skipped = 0
    skipped = 0
    skip_block_indices = set(skip_block_indices or ())
    memory_efficient_mode = str(memory_efficient_mode).strip().lower()
    if memory_efficient_mode not in {"full", "inner", "none"}:
        raise ValueError(
            "memory_efficient_mode must be one of full, inner, none; "
            f"got {memory_efficient_mode}"
        )
    block_container = getattr(model, "blocks", None)
    if block_container is None:
        LOGGER.warning("No transformer blocks found for %s", model_name)
        return

    def _make_ckpt(fn):
        @wraps(fn)
        def _wrapped(x, e, seq_lens, grid_sizes, freqs, context, context_lens, dit_cond_dict=None):
            def _forward(*tensor_args):
                return fn(*tensor_args, dit_cond_dict=dit_cond_dict)

            checkpoint_args = (x, e, seq_lens, grid_sizes, freqs, context, context_lens)
            if use_reentrant and not any(
                torch.is_tensor(arg) and arg.requires_grad for arg in checkpoint_args
            ):
                # Reentrant checkpointing drops parameter gradients when no input
                # requires grad. LoRA tuning freezes the student inputs, so mark
                # the actual hidden state as a local grad anchor.
                checkpoint_args = (
                    x.detach().requires_grad_(True),
                    e,
                    seq_lens,
                    grid_sizes,
                    freqs,
                    context,
                    context_lens,
                )

            checkpoint_kwargs = {"use_reentrant": use_reentrant}
            if not use_reentrant:
                # ZeRO-3 can expose sharded parameters as empty tensors during
                # non-reentrant recompute, which trips PyTorch's metadata check
                # even though DeepSpeed gathers them for the actual math.
                checkpoint_kwargs["determinism_check"] = "none"

            def _run_checkpoint():
                return checkpoint_module.checkpoint(
                    _forward,
                    *checkpoint_args,
                    **checkpoint_kwargs,
                )

            if use_reentrant:
                return _run_checkpoint()

            # Non-reentrant checkpointing stops recomputation once saved
            # tensors have been rebuilt. With ZeRO-3 parameter offload that
            # can interrupt a Wan block before DeepSpeed has balanced all
            # parameter gather/release hooks, which has shown up as native
            # SIGFPEs during the first backward. Force a full block replay.
            early_stop = getattr(checkpoint_module, "set_checkpoint_early_stop", None)
            if early_stop is None:
                return _run_checkpoint()
            with early_stop(False):
                return _run_checkpoint()

        return _wrapped

    for block_index, block in enumerate(block_container):
        if block_index in skip_block_indices:
            block._pc_gradient_checkpointing_skipped = True
            block._pc_checkpoint_skip_reason = "requested"
            block._pc_checkpoint_mode = "skipped"
            skipped += 1
            continue
        if getattr(block, "_pc_gradient_checkpointing_patched", False):
            continue

        is_memory_efficient = bool(getattr(block, "_pc_memory_efficient_modulation_patched", False))
        if is_memory_efficient:
            if memory_efficient_mode == "inner":
                block._pc_inner_gradient_checkpointing = True
                block._pc_checkpoint_use_reentrant = use_reentrant
                block._pc_gradient_checkpointing_patched = True
                block._pc_checkpoint_mode = "inner"
                inner_patched += 1
                continue
            if memory_efficient_mode == "none":
                block._pc_inner_gradient_checkpointing = False
                block._pc_checkpoint_use_reentrant = use_reentrant
                block._pc_gradient_checkpointing_patched = True
                block._pc_checkpoint_mode = "none"
                memory_efficient_skipped += 1
                continue
            # The memory-efficient block already chunks expensive FFN math. Wrap
            # the whole block so attention/norm/camera activations are replayed
            # instead of retained across all Wan layers.
            block._pc_inner_gradient_checkpointing = False
            memory_efficient_patched += 1

        original_forward = block.forward

        block.forward = _make_ckpt(original_forward)
        block._pc_gradient_checkpointing_patched = True
        block._pc_checkpoint_use_reentrant = use_reentrant
        block._pc_checkpoint_mode = "full"
        patched += 1
    if _should_log_rank_zero():
        if inner_patched:
            LOGGER.info(
                "Gradient checkpointing enabled inner FFN/camera checkpointing for %s "
                "memory-efficient Wan blocks in %s (use_reentrant=%s; attention remains outside checkpoint)",
                inner_patched,
                model_name,
                use_reentrant,
            )
        if memory_efficient_patched:
            LOGGER.info(
                "Gradient checkpointing wrapped %s memory-efficient Wan blocks in %s "
                "(use_reentrant=%s; full block replay)",
                memory_efficient_patched,
                model_name,
                use_reentrant,
            )
        if memory_efficient_skipped:
            LOGGER.info(
                "Gradient checkpointing left %s memory-efficient Wan blocks uncheckpointed in %s",
                memory_efficient_skipped,
                model_name,
            )
        if patched:
            LOGGER.info(
                "Gradient checkpointing patched %s blocks for %s (use_reentrant=%s)",
                patched,
                model_name,
                use_reentrant,
            )
        if skipped:
            LOGGER.info(
                "Gradient checkpointing skipped %s blocks for %s (indices=%s)",
                skipped,
                model_name,
                ",".join(str(index) for index in sorted(skip_block_indices)),
            )
        if not patched and not inner_patched and not memory_efficient_skipped:
            LOGGER.warning("No transformer blocks were patched for gradient checkpointing in %s", model_name)


def apply_memory_efficient_wan_block_patch(
    model,
    model_name: str = "model",
    *,
    ffn_chunk_size: int | None = None,
    norm_chunk_size: int | None = None,
) -> None:
    """Avoid full fp32 modulation tensors and optionally chunk FFN activations."""

    if ffn_chunk_size is not None and ffn_chunk_size <= 0:
        ffn_chunk_size = None
    if norm_chunk_size is not None and norm_chunk_size <= 0:
        norm_chunk_size = None

    norm_patched = 0
    if norm_chunk_size is not None:
        norm_patched = _patch_sequence_norm_forward(model, int(norm_chunk_size))

    def _squeeze_gate(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim >= 3 and tensor.shape[2] == 1:
            return tensor.squeeze(2)
        return tensor

    def _use_fp32_sensitive_path(reference: torch.Tensor) -> bool:
        return reference.device.type == "cuda" and (_stage1_mixed_safe() or _stage1_force_fp32())

    def _modulate(normed: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
        normed = normed.to(target_dtype)
        shift = _squeeze_gate(shift).to(device=normed.device, dtype=target_dtype)
        scale = _squeeze_gate(scale).to(device=normed.device, dtype=target_dtype)
        return normed * (1 + scale) + shift

    def _slice_sequence(tensor: torch.Tensor, start: int, stop: int) -> torch.Tensor:
        if tensor.ndim < 3 or tensor.shape[1] == 1:
            return tensor
        return tensor[:, start:stop]

    def _apply_gated_residual(base: torch.Tensor, value: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        output_dtype = base.dtype
        compute_dtype = torch.float32 if _use_fp32_sensitive_path(base) else value.dtype
        base_work = base.to(compute_dtype)
        value_work = value.to(compute_dtype)
        gate_work = _squeeze_gate(gate).to(device=value.device, dtype=compute_dtype)
        output = torch.addcmul(base_work, value_work, gate_work)
        return output if output.dtype == output_dtype else output.to(output_dtype)

    def _run_checkpointed_inner(
        block: torch.nn.Module,
        fn,
        *args: torch.Tensor,
    ) -> torch.Tensor:
        if not getattr(block, "_pc_inner_gradient_checkpointing", False) or not torch.is_grad_enabled():
            return fn(*args)

        import torch.utils.checkpoint as checkpoint_module

        use_reentrant = bool(getattr(block, "_pc_checkpoint_use_reentrant", False))
        checkpoint_args = args
        needs_anchor = use_reentrant and not any(
            torch.is_tensor(arg) and arg.requires_grad for arg in checkpoint_args
        )
        if needs_anchor:
            anchor_source = next((arg for arg in checkpoint_args if torch.is_tensor(arg)), None)
            if anchor_source is None:
                return fn(*args)
            anchor = torch.ones(
                (),
                device=anchor_source.device,
                dtype=anchor_source.dtype,
                requires_grad=True,
            )
            checkpoint_args = (*checkpoint_args, anchor)

            def _forward(*inner_args):
                *fn_args, anchor_arg = inner_args
                output = fn(*fn_args)
                return output + anchor_arg.to(dtype=output.dtype) * 0

        else:
            def _forward(*inner_args):
                return fn(*inner_args)

        checkpoint_kwargs = {"use_reentrant": use_reentrant}
        if not use_reentrant:
            checkpoint_kwargs["determinism_check"] = "none"

        def _run_checkpoint():
            return checkpoint_module.checkpoint(_forward, *checkpoint_args, **checkpoint_kwargs)

        if use_reentrant:
            return _run_checkpoint()

        early_stop = getattr(checkpoint_module, "set_checkpoint_early_stop", None)
        if early_stop is None:
            return _run_checkpoint()
        with early_stop(False):
            return _run_checkpoint()

    def _extract_control_tensor(dit_cond_dict) -> torch.Tensor | None:
        if not dit_cond_dict or "c2ws_plucker_emb" not in dit_cond_dict:
            return None
        value = dit_cond_dict["c2ws_plucker_emb"]
        if isinstance(value, (tuple, list)):
            if not value:
                return None
            value = value[0]
        return value if torch.is_tensor(value) else None

    def _run_module_layers(module: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        # Running nested Sequentials layer-by-layer avoids a large ZeRO-3 fetch on the
        # outer container, which otherwise gathers the whole FFN at once.
        if isinstance(module, torch.nn.Sequential):
            value = inputs
            for submodule in module:
                value = _run_module_layers(submodule, value)
            return value
        return module(inputs)

    def _run_ffn_residual(
        block: torch.nn.Module,
        ffn: torch.nn.Module,
        norm: torch.nn.Module,
        residual: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
        gate: torch.Tensor,
        target_dtype: torch.dtype,
        ) -> torch.Tensor:
        if ffn_chunk_size is None or residual.shape[1] <= ffn_chunk_size:
            def _ffn_body(
                residual_arg: torch.Tensor,
                shift_arg: torch.Tensor,
                scale_arg: torch.Tensor,
                gate_arg: torch.Tensor,
            ):
                ffn_input_dtype = torch.float32 if _use_fp32_sensitive_path(residual_arg) else target_dtype
                ffn_input = _modulate(norm(residual_arg), shift_arg, scale_arg, ffn_input_dtype)
                y = _run_module_layers(ffn, ffn_input)
                return _apply_gated_residual(residual_arg, y, gate_arg)

            return _run_checkpointed_inner(block, _ffn_body, residual, shift, scale, gate)

        output = torch.empty_like(residual)
        for start in range(0, residual.shape[1], ffn_chunk_size):
            stop = min(start + ffn_chunk_size, residual.shape[1])
            residual_chunk = residual[:, start:stop]
            shift_chunk = _slice_sequence(shift, start, stop)
            scale_chunk = _slice_sequence(scale, start, stop)
            gate_chunk = _slice_sequence(gate, start, stop)

            def _ffn_chunk_body(
                residual_arg: torch.Tensor,
                shift_arg: torch.Tensor,
                scale_arg: torch.Tensor,
                gate_arg: torch.Tensor,
            ):
                ffn_input_dtype = torch.float32 if _use_fp32_sensitive_path(residual_arg) else target_dtype
                ffn_input_chunk = _modulate(norm(residual_arg), shift_arg, scale_arg, ffn_input_dtype)
                y_chunk = _run_module_layers(ffn, ffn_input_chunk)
                return _apply_gated_residual(residual_arg, y_chunk, gate_arg)

            output[:, start:stop] = _run_checkpointed_inner(
                block,
                _ffn_chunk_body,
                residual_chunk,
                shift_chunk,
                scale_chunk,
                gate_chunk,
            )
        return output

    def _run_camera_injection(block: torch.nn.Module, x: torch.Tensor, dit_cond_dict, target_dtype: torch.dtype) -> torch.Tensor:
        c2ws = _extract_control_tensor(dit_cond_dict)
        if c2ws is None:
            _wan_trace("camera_skip_no_control", block=block, tensors={"x": x})
            return x
        if c2ws.ndim < 3:
            _wan_trace(
                "camera_skip_bad_control_rank",
                block=block,
                tensors={"x": x, "c2ws": c2ws},
                detail=f"c2ws_ndim={c2ws.ndim}",
            )
            return x
        required_cam_attrs = (
            "cam_injector_layer1",
            "cam_injector_layer2",
            "cam_scale_layer",
            "cam_shift_layer",
        )
        if not all(hasattr(block, attr) for attr in required_cam_attrs):
            _wan_trace("camera_skip_missing_layers", block=block, tensors={"x": x, "c2ws": c2ws})
            return x

        c2ws = c2ws.to(device=x.device, dtype=target_dtype)
        _wan_trace("camera_enter", block=block, tensors={"x": x, "c2ws": c2ws}, sync=True)

        def _run_chunk(x_chunk: torch.Tensor, c2ws_chunk: torch.Tensor, start: int, stop: int) -> torch.Tensor:
            detail = f"chunk={start}:{stop}"
            _wan_trace(
                "camera_chunk_before_injector1",
                block=block,
                tensors={"x": x_chunk, "c2ws": c2ws_chunk},
                detail=detail,
                sync=True,
            )
            hidden = block.cam_injector_layer1(c2ws_chunk)
            _wan_trace(
                "camera_chunk_after_injector1",
                block=block,
                tensors={"hidden": hidden},
                detail=detail,
                sync=True,
            )
            hidden = F.silu(hidden)
            _wan_trace(
                "camera_chunk_after_silu",
                block=block,
                tensors={"hidden": hidden},
                detail=detail,
                sync=True,
            )
            hidden = block.cam_injector_layer2(hidden)
            _wan_trace(
                "camera_chunk_after_injector2",
                block=block,
                tensors={"hidden": hidden},
                detail=detail,
                sync=True,
            )
            hidden = hidden + c2ws_chunk
            _wan_trace(
                "camera_chunk_before_scale",
                block=block,
                tensors={"hidden": hidden},
                detail=detail,
                sync=True,
            )
            cam_scale = block.cam_scale_layer(hidden)
            _wan_trace(
                "camera_chunk_after_scale",
                block=block,
                tensors={"cam_scale": cam_scale},
                detail=detail,
                sync=True,
            )
            _wan_trace(
                "camera_chunk_before_shift",
                block=block,
                tensors={"hidden": hidden},
                detail=detail,
                sync=True,
            )
            cam_shift = block.cam_shift_layer(hidden)
            _wan_trace(
                "camera_chunk_after_shift",
                block=block,
                tensors={"cam_shift": cam_shift},
                detail=detail,
                sync=True,
            )
            output = (1.0 + cam_scale.to(dtype=x_chunk.dtype)) * x_chunk + cam_shift.to(dtype=x_chunk.dtype)
            _wan_trace(
                "camera_chunk_exit",
                block=block,
                tensors={"output": output},
                detail=detail,
                sync=True,
            )
            return output

        def _camera_body(x_arg: torch.Tensor, c2ws_arg: torch.Tensor) -> torch.Tensor:
            return _run_chunk(x_arg, c2ws_arg, 0, int(x_arg.shape[1]))

        # Keep camera injection whole-sequence: chunking repeatedly gathers the
        # same ZeRO-3 offloaded camera weights and can retain them until OOM.
        output = _run_checkpointed_inner(block, _camera_body, x, c2ws)
        _wan_trace("camera_exit", block=block, tensors={"output": output}, sync=True)
        return output

    patched = 0
    block_container = getattr(model, "blocks", None)
    if block_container is None:
        LOGGER.warning("No transformer blocks found for %s", model_name)
        return

    for block_index, block in enumerate(block_container):
        block._pc_wan_block_index = block_index
        if getattr(block, "_pc_memory_efficient_modulation_patched", False):
            continue
        required_attrs = ("modulation", "norm1", "self_attn", "norm2", "ffn", "norm3", "cross_attn")
        if not all(hasattr(block, attr) for attr in required_attrs):
            continue

        def _forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens, dit_cond_dict=None):
            _wan_trace("block_enter", block=self, tensors={"x": x}, sync=True)
            target_dtype = x.dtype
            e = (
                self.modulation.unsqueeze(0).to(device=e.device, dtype=torch.float32)
                + e.to(dtype=torch.float32)
            ).chunk(6, dim=2)
            fp32_sensitive = _use_fp32_sensitive_path(x)

            _wan_trace("before_self_attn_norm", block=self, tensors={"x": x}, sync=True)
            self_attn_input = _modulate(
                self.norm1(x),
                e[0],
                e[1],
                torch.float32 if fp32_sensitive else target_dtype,
            )
            _wan_trace(
                "before_self_attn",
                block=self,
                tensors={"self_attn_input": self_attn_input},
                sync=True,
            )
            y = self.self_attn(self_attn_input, seq_lens, grid_sizes, freqs)
            _wan_trace("after_self_attn", block=self, tensors={"y": y}, sync=True)
            x = _apply_gated_residual(x, y, e[2])
            _wan_trace("after_self_attn_residual", block=self, tensors={"x": x}, sync=True)

            _wan_trace("before_camera_injection", block=self, tensors={"x": x}, sync=True)
            x = _run_camera_injection(self, x, dit_cond_dict, target_dtype)
            _wan_trace("after_camera_injection", block=self, tensors={"x": x}, sync=True)

            _wan_trace("before_cross_attn_norm", block=self, tensors={"x": x}, sync=True)
            cross_input = self.norm3(x).to(target_dtype)
            _wan_trace("before_cross_attn", block=self, tensors={"cross_input": cross_input}, sync=True)
            x = x + self.cross_attn(cross_input, context, context_lens)
            _wan_trace("after_cross_attn", block=self, tensors={"x": x}, sync=True)

            _wan_trace("before_ffn", block=self, tensors={"x": x}, sync=True)
            x = _run_ffn_residual(
                self,
                self.ffn,
                self.norm2,
                x,
                e[3],
                e[4],
                e[5],
                target_dtype,
            )
            _wan_trace("after_ffn", block=self, tensors={"x": x}, sync=True)
            return x

        block.forward = MethodType(_forward, block)
        block._pc_memory_efficient_modulation_patched = True
        patched += 1

    if patched:
        if _should_log_rank_zero():
            LOGGER.info(
                "Memory-efficient modulation patched %s blocks for %s (ffn_chunk_size=%s)",
                patched,
                model_name,
                ffn_chunk_size or "disabled",
            )
            if norm_patched:
                LOGGER.info(
                    "Memory-efficient Wan sequence norms patched %s modules for %s (norm_chunk_size=%s)",
                    norm_patched,
                    model_name,
                    norm_chunk_size,
                )
    else:
        LOGGER.warning("No Wan attention blocks matched modulation patch for %s", model_name)


def _patch_sequence_norm_forward(model: torch.nn.Module, chunk_size: int) -> int:
    """Chunk Wan norm modules that internally cast long sequences to fp32."""

    patched = 0
    target_class_names = {"WanRMSNorm", "WanLayerNorm"}
    for module in model.modules():
        if module.__class__.__name__ not in target_class_names:
            continue
        if getattr(module, "_pc_sequence_norm_chunk_patched", False):
            continue
        original_forward = module.forward

        def _make_forward(fn):
            def _forward(self, x: torch.Tensor):
                if not torch.is_tensor(x) or x.ndim < 3 or x.shape[1] <= chunk_size:
                    return fn(x)
                outputs = []
                for start in range(0, x.shape[1], chunk_size):
                    stop = min(start + chunk_size, x.shape[1])
                    outputs.append(fn(x[:, start:stop]))
                return torch.cat(outputs, dim=1)

            return _forward

        module.forward = MethodType(_make_forward(original_forward), module)
        module._pc_sequence_norm_chunk_patched = True
        patched += 1
    return patched


class LoRALinear(torch.nn.Module):
    """Standard LoRA wrapper for nn.Linear."""

    def __init__(
        self,
        base: torch.nn.Linear,
        *,
        rank: int,
        alpha: int,
        dropout: float,
        chunk_size: int | None = None,
        merge_mode: str = "inplace",
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        merge_mode = str(merge_mode).strip().lower().replace("-", "_")
        if merge_mode not in {"inplace", "out_of_place"}:
            raise ValueError(f"LoRA merge_mode must be one of inplace, out_of_place; got {merge_mode}")
        self.base = base
        self.rank = int(rank)
        self.alpha = int(alpha)
        self.scaling = float(self.alpha) / float(self.rank)
        self.chunk_size = int(chunk_size) if chunk_size is not None and chunk_size > 0 else None
        self.merge_mode = merge_mode
        self.detach_base_out = _env_flag("PC_LORA_DETACH_BASE_OUT")
        self.detach_input = _env_flag("PC_LORA_DETACH_INPUT")
        self.record_local_loss = _env_flag("PC_LORA_LOCAL_LOSS")
        self.clone_input = _env_flag("PC_LORA_INPUT_CONTIGUOUS_CLONE")
        self.clone_hidden = _env_flag("PC_LORA_HIDDEN_CONTIGUOUS_CLONE")
        self.trace_input_meta = _env_flag("PC_LORA_TRACE_INPUT_META")
        self.disable_autocast = _env_flag("PC_LORA_DISABLE_AUTOCAST")
        self.numeric_audit = _numeric_audit_enabled("PC_LORA_NUMERIC_AUDIT")
        self._pc_lora_local_losses: list[torch.Tensor] = []
        self._pc_lora_name = "<unregistered>"
        self._pc_lora_trace_logged = False
        self.dropout = torch.nn.Dropout(float(dropout)) if dropout > 0 else torch.nn.Identity()
        lora_dtype = torch.float32 if _env_flag("PC_FORCE_LORA_FP32") else base.weight.dtype
        self.lora_A = torch.nn.Linear(
            base.in_features,
            self.rank,
            bias=False,
            device=base.weight.device,
            dtype=lora_dtype,
        )
        self.lora_B = torch.nn.Linear(
            self.rank,
            base.out_features,
            bias=False,
            device=base.weight.device,
            dtype=lora_dtype,
        )
        torch.nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_B.weight)
        if self.numeric_audit:
            self.lora_A.weight.register_hook(lambda grad: self._audit_lora_grad("lora_A.weight", grad))
            self.lora_B.weight.register_hook(lambda grad: self._audit_lora_grad("lora_B.weight", grad))
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def _audit_lora_grad(self, suffix: str, grad: torch.Tensor) -> torch.Tensor:
        _audit_tensor_numerics(
            f"lora_grad name={self._pc_lora_name}.{suffix}",
            grad,
            key="lora_grad",
            enabled=True,
        )
        return grad

    def _lora_forward(self, x: torch.Tensor, *, out_dtype: torch.dtype) -> torch.Tensor:
        lora_dtype = self.lora_A.weight.dtype
        raw_x = x
        if self.detach_input:
            x = x.detach()
        lora_input = self.dropout(x.to(dtype=lora_dtype))
        _audit_tensor_numerics(
            f"lora_input name={self._pc_lora_name}",
            lora_input,
            key="lora_forward",
            enabled=self.numeric_audit,
        )
        if self.clone_input:
            lora_input = lora_input.contiguous().clone()
        autocast_context = (
            torch.amp.autocast(lora_input.device.type, enabled=False)
            if self.disable_autocast and lora_input.device.type in {"cuda", "cpu"}
            else contextlib.nullcontext()
        )
        with autocast_context:
            lora_hidden = self.lora_A(lora_input)
            _audit_tensor_numerics(
                f"lora_hidden name={self._pc_lora_name}",
                lora_hidden,
                key="lora_forward",
                enabled=self.numeric_audit,
            )
            if self.clone_hidden:
                lora_hidden = lora_hidden.contiguous().clone()
            lora_out = self.lora_B(lora_hidden)
            _audit_tensor_numerics(
                f"lora_out name={self._pc_lora_name}",
                lora_out,
                key="lora_forward",
                enabled=self.numeric_audit,
            )
        if self.trace_input_meta and not self._pc_lora_trace_logged and _should_log_rank_zero():
            LOGGER.info(
                "PC_LORA_TRACE_INPUT_META name=%s raw_shape=%s raw_dtype=%s raw_stride=%s raw_contig=%s raw_requires_grad=%s input_shape=%s input_dtype=%s input_stride=%s input_contig=%s hidden_shape=%s hidden_dtype=%s hidden_stride=%s hidden_contig=%s out_shape=%s out_dtype=%s out_stride=%s out_contig=%s clone_input=%s clone_hidden=%s disable_autocast=%s",
                self._pc_lora_name,
                tuple(raw_x.shape),
                raw_x.dtype,
                raw_x.stride(),
                raw_x.is_contiguous(),
                raw_x.requires_grad,
                tuple(lora_input.shape),
                lora_input.dtype,
                lora_input.stride(),
                lora_input.is_contiguous(),
                tuple(lora_hidden.shape),
                lora_hidden.dtype,
                lora_hidden.stride(),
                lora_hidden.is_contiguous(),
                tuple(lora_out.shape),
                lora_out.dtype,
                lora_out.stride(),
                lora_out.is_contiguous(),
                self.clone_input,
                self.clone_hidden,
                self.disable_autocast,
            )
            self._pc_lora_trace_logged = True
        if self.record_local_loss:
            self._pc_lora_local_losses.append(lora_hidden.float().square().mean() + lora_out.float().mean())
        if lora_out.dtype != out_dtype:
            lora_out = lora_out.to(dtype=out_dtype)
        return lora_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.record_local_loss:
            self._pc_lora_local_losses = []
        base_out = self.base(x)
        if self.detach_base_out:
            base_out = base_out.detach()
        if self.chunk_size is None or x.ndim < 3 or x.shape[1] <= self.chunk_size:
            lora_out = self._lora_forward(x, out_dtype=base_out.dtype)
            if self.merge_mode == "out_of_place" or self.detach_base_out:
                return base_out + lora_out * self.scaling
            return base_out.add_(lora_out, alpha=self.scaling)

        if self.merge_mode == "out_of_place" or self.detach_base_out:
            lora_chunks = []
            for start in range(0, x.shape[1], self.chunk_size):
                stop = min(start + self.chunk_size, x.shape[1])
                lora_chunks.append(self._lora_forward(x[:, start:stop], out_dtype=base_out.dtype))
            return base_out + torch.cat(lora_chunks, dim=1) * self.scaling

        for start in range(0, x.shape[1], self.chunk_size):
            stop = min(start + self.chunk_size, x.shape[1])
            lora_chunk = self._lora_forward(x[:, start:stop], out_dtype=base_out.dtype)
            base_out[:, start:stop].add_(lora_chunk, alpha=self.scaling)
        return base_out


def _iter_lora_modules(model: torch.nn.Module) -> list[tuple[str, LoRALinear]]:
    return [(name, module) for name, module in model.named_modules() if isinstance(module, LoRALinear)]


def apply_lora_to_wan_model(
    model: torch.nn.Module,
    *,
    model_name: str = "model",
    rank: int,
    alpha: int,
    dropout: float,
    target_prefixes: tuple[str, ...] = ("blocks",),
    block_start: int = 0,
    lora_chunk_size: int | None = None,
    merge_mode: str = "inplace",
) -> None:
    """Replace selected Wan linear layers with standard LoRA adapters."""

    block_start = int(block_start)
    if block_start < 0:
        raise ValueError(f"LoRA block_start must be non-negative, got {block_start}")
    if lora_chunk_size is not None and lora_chunk_size <= 0:
        lora_chunk_size = None
    merge_mode = str(merge_mode).strip().lower().replace("-", "_")
    if merge_mode not in {"inplace", "out_of_place"}:
        raise ValueError(f"LoRA merge_mode must be one of inplace, out_of_place; got {merge_mode}")

    def _block_index(full_name: str) -> int | None:
        parts = full_name.split(".")
        if len(parts) < 2 or parts[0] != "blocks":
            return None
        try:
            return int(parts[1])
        except ValueError:
            return None

    def _is_target_module(full_name: str) -> bool:
        if not any(full_name == prefix or full_name.startswith(f"{prefix}.") for prefix in target_prefixes):
            return False
        index = _block_index(full_name)
        return index is None or index >= block_start

    replaced = 0
    for full_name, module in list(model.named_modules()):
        if not full_name or not _is_target_module(full_name):
            continue
        if ".base" in full_name or ".lora_" in full_name:
            continue
        if isinstance(module, LoRALinear) or not isinstance(module, torch.nn.Linear):
            continue
        parent_name, child_name = full_name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        parent._modules[child_name] = LoRALinear(
            module,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            chunk_size=lora_chunk_size,
            merge_mode=merge_mode,
        )
        parent._modules[child_name]._pc_lora_name = full_name
        replaced += 1

    if replaced == 0:
        raise RuntimeError(f"No linear layers matched LoRA targets for {model_name}")

    for parameter in model.parameters():
        parameter.requires_grad = False
    trainable_params = 0
    total_params = 0
    lora_dtypes: set[torch.dtype] = set()
    detach_base_out = False
    detach_input = False
    local_loss_probe = False
    clone_input = False
    clone_hidden = False
    trace_input_meta = False
    disable_autocast = False
    numeric_audit = False
    for _, module in _iter_lora_modules(model):
        module.lora_A.weight.requires_grad = True
        module.lora_B.weight.requires_grad = True
        lora_dtypes.add(module.lora_A.weight.dtype)
        lora_dtypes.add(module.lora_B.weight.dtype)
        detach_base_out = detach_base_out or module.detach_base_out
        detach_input = detach_input or module.detach_input
        local_loss_probe = local_loss_probe or module.record_local_loss
        clone_input = clone_input or module.clone_input
        clone_hidden = clone_hidden or module.clone_hidden
        trace_input_meta = trace_input_meta or module.trace_input_meta
        disable_autocast = disable_autocast or module.disable_autocast
        numeric_audit = numeric_audit or module.numeric_audit
    for parameter in model.parameters():
        total_params += parameter.numel()
        if parameter.requires_grad:
            trainable_params += parameter.numel()

    lora_dtype_text = ",".join(sorted(str(dtype).replace("torch.", "") for dtype in lora_dtypes))
    model._pc_lora_config = {
        "rank": int(rank),
        "alpha": int(alpha),
        "dropout": float(dropout),
        "target_prefixes": tuple(target_prefixes),
        "block_start": block_start,
        "lora_chunk_size": lora_chunk_size,
        "merge_mode": merge_mode,
        "lora_dtype": lora_dtype_text,
        "force_lora_fp32": _env_flag("PC_FORCE_LORA_FP32"),
        "detach_base_out": detach_base_out,
        "detach_input": detach_input,
        "local_loss_probe": local_loss_probe,
        "clone_input": clone_input,
        "clone_hidden": clone_hidden,
        "trace_input_meta": trace_input_meta,
        "disable_autocast": disable_autocast,
        "numeric_audit": numeric_audit,
    }
    if _should_log_rank_zero():
        LOGGER.info(
            "Applied standard LoRA to %s linear layers for %s (rank=%s, alpha=%s, dropout=%.3f, block_start=%s, lora_chunk_size=%s, merge_mode=%s, lora_dtype=%s, force_lora_fp32=%s, detach_base_out=%s, detach_input=%s, local_loss_probe=%s, clone_input=%s, clone_hidden=%s, trace_input_meta=%s, disable_autocast=%s, numeric_audit=%s, trainable=%s/%s)",
            replaced,
            model_name,
            rank,
            alpha,
            dropout,
            block_start,
            lora_chunk_size or "disabled",
            merge_mode,
            lora_dtype_text,
            _env_flag("PC_FORCE_LORA_FP32"),
            detach_base_out,
            detach_input,
            local_loss_probe,
            clone_input,
            clone_hidden,
            trace_input_meta,
            disable_autocast,
            numeric_audit,
            trainable_params,
            total_params,
        )


def collect_lora_local_loss(model: torch.nn.Module) -> torch.Tensor:
    """Return a direct LoRA-output loss collected during the latest forward."""

    losses: list[torch.Tensor] = []
    for _, module in _iter_lora_modules(model):
        losses.extend(module._pc_lora_local_losses)
    if not losses:
        raise RuntimeError("PC_LORA_LOCAL_LOSS=1 but no LoRA local losses were collected")
    return torch.stack(losses).mean()


def collect_lora_parameter_loss(model: torch.nn.Module) -> torch.Tensor:
    """Return a tiny loss that touches only LoRA parameters, no matmul graph."""

    losses: list[torch.Tensor] = []
    for _, module in _iter_lora_modules(model):
        for parameter in (module.lora_A.weight, module.lora_B.weight):
            p = parameter.float()
            losses.append(p.mean() + 1.0e-6 * p.square().mean())
    if not losses:
        raise RuntimeError("PC_LORA_PARAM_ONLY_LOSS=1 but no LoRA parameters were found")
    return torch.stack(losses).mean()


def collect_lora_synthetic_matmul_loss(model: torch.nn.Module) -> torch.Tensor:
    """Return a LoRA matmul loss using synthetic detached activations."""

    modules = _iter_lora_modules(model)
    if not modules:
        raise RuntimeError("PC_LORA_SYNTHETIC_MATMUL_LOSS=1 but no LoRA modules were found")

    start = max(_env_int("PC_LORA_SYNTHETIC_MODULE_START", 0), 0)
    limit = max(_env_int("PC_LORA_SYNTHETIC_MODULE_LIMIT", 1), 1)
    tokens = max(_env_int("PC_LORA_SYNTHETIC_TOKENS", 3600), 1)
    batch = max(_env_int("PC_LORA_SYNTHETIC_BATCH", 1), 1)
    selected = modules[start : start + limit]
    if not selected:
        raise RuntimeError(
            f"PC_LORA_SYNTHETIC_MATMUL_LOSS=1 selected no modules "
            f"(start={start}, limit={limit}, available={len(modules)})"
        )

    losses: list[torch.Tensor] = []
    descriptions: list[str] = []
    for index, (name, module) in enumerate(selected, start=start):
        dtype = module.lora_A.weight.dtype
        device = module.lora_A.weight.device
        x = torch.randn(
            batch,
            tokens,
            module.lora_A.in_features,
            device=device,
            dtype=dtype,
        )
        hidden = module.lora_A(x)
        out = module.lora_B(hidden)
        losses.append(hidden.float().square().mean() + out.float().mean())
        descriptions.append(
            f"{index}:{name}:in={module.lora_A.in_features}:rank={module.rank}:out={module.lora_B.out_features}"
        )

    if _should_log_rank_zero():
        LOGGER.info(
            "PC_LORA_SYNTHETIC_MATMUL_LOSS=1: modules=%s batch=%s tokens=%s dtype=%s",
            ";".join(descriptions),
            batch,
            tokens,
            ",".join(sorted({str(module.lora_A.weight.dtype).replace("torch.", "") for _, module in selected})),
        )
    return torch.stack(losses).mean()


def extract_lora_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return just the trainable LoRA adapter tensors for one model."""
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if ".lora_A.weight" in key or ".lora_B.weight" in key
    }


def load_lora_state_dict(model: torch.nn.Module, state_dict: dict[str, torch.Tensor], *, model_name: str = "model") -> None:
    """Load adapter-only weights into a LoRA-patched model."""
    if not state_dict:
        LOGGER.warning("No LoRA weights provided for %s", model_name)
        return
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.unexpected_keys:
        raise KeyError(f"Unexpected LoRA keys for {model_name}: {incompatible.unexpected_keys}")
    loaded = len(state_dict)
    LOGGER.info("Loaded %s LoRA tensors for %s", loaded, model_name)


def export_pretrained_state_dict(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    *,
    prefix: str = "",
) -> dict[str, torch.Tensor]:
    """Export a standard checkpoint state_dict, merging LoRA weights when present."""
    exported: dict[str, torch.Tensor] = {}
    lora_modules = dict(_iter_lora_modules(model))
    skip_keys: set[str] = set()

    for module_name, _module in lora_modules.items():
        skip_keys.update(
            {
                f"{module_name}.base.weight",
                f"{module_name}.base.bias",
                f"{module_name}.lora_A.weight",
                f"{module_name}.lora_B.weight",
            }
        )

    for key, value in state_dict.items():
        if not key.startswith(prefix):
            continue
        local_key = key[len(prefix) :]
        if local_key in skip_keys:
            continue
        exported[local_key] = value.detach().cpu()

    for module_name, module in lora_modules.items():
        base_weight_key = f"{prefix}{module_name}.base.weight"
        lora_a_key = f"{prefix}{module_name}.lora_A.weight"
        lora_b_key = f"{prefix}{module_name}.lora_B.weight"
        base_weight = state_dict[base_weight_key]
        lora_a = state_dict[lora_a_key]
        lora_b = state_dict[lora_b_key]
        merged_weight = (
            base_weight.float() + (lora_b.float() @ lora_a.float()) * module.scaling
        ).to(dtype=base_weight.dtype)
        exported[f"{module_name}.weight"] = merged_weight.detach().cpu()

        base_bias_key = f"{prefix}{module_name}.base.bias"
        if base_bias_key in state_dict:
            exported[f"{module_name}.bias"] = state_dict[base_bias_key].detach().cpu()

    return exported


def get_model_subfolder(model_type: str) -> str:
    """Map a logical model type to its checkpoint subfolder."""
    if model_type not in MODEL_TYPE_TO_SUBFOLDER:
        raise ValueError(f"Unsupported model_type: {model_type}")
    return MODEL_TYPE_TO_SUBFOLDER[model_type]


def save_accelerate_checkpoint(
    *,
    accelerator,
    model,
    args,
    tag: str,
    extra_training_state: dict | None = None,
    training_state_filename: str = "resume_state.pt",
) -> Path:
    """Save one or more Stage-1 branches in WanModel.from_pretrained layout."""
    save_dir = Path(args.output_dir) / tag
    if accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    if isinstance(model, dict):
        model_items = list(model.items())
    else:
        subfolder = getattr(args, "subfolder", "")
        if not subfolder:
            raise ValueError("subfolder must be set when saving a single model")
        model_items = [(subfolder, model)]

    for subfolder, branch_model in model_items:
        model_dir = save_dir / subfolder
        if accelerator.is_main_process:
            model_dir.mkdir(parents=True, exist_ok=True)
        accelerator.wait_for_everyone()
        unwrapped = accelerator.unwrap_model(branch_model)
        if hasattr(unwrapped, "save_16bit_model"):
            unwrapped.save_16bit_model(model_dir, "diffusion_pytorch_model.bin")
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped.save_config(model_dir)
        elif accelerator.is_main_process:
            unwrapped.save_pretrained(model_dir, safe_serialization=False)
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        if extra_training_state:
            training_only = save_dir / "training_only"
            training_only.mkdir(parents=True, exist_ok=True)
            torch.save(extra_training_state, training_only / training_state_filename)
    accelerator.wait_for_everyone()
    return save_dir


def prune_checkpoint_dir(path: str | Path) -> None:
    """Delete one checkpoint directory if it exists."""
    root = Path(path)
    if root.exists():
        shutil.rmtree(root)


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device | str) -> None:
    """Move optimizer state tensors between CPU and GPU."""
    target = torch.device(device)
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(target)


def build_dataloader(
    dataset_dir: str,
    *,
    split: str,
    num_frames: int,
    height: int,
    width: int,
    repeat: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    """Build a standard Stage-1 dataloader."""
    dataset = CSGODataset(
        dataset_dir,
        split=split,
        num_frames=num_frames,
        height=height,
        width=width,
        repeat=repeat,
    )
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda items: items[0],
    )


def compute_scheduler_total_steps(dataset_len: int, num_processes: int, grad_accum: int, num_epochs: int) -> int:
    """Mirror the optimizer-step counting logic from Stage-1."""
    iters_per_epoch = math.ceil(dataset_len / num_processes)
    steps_per_epoch = max(math.ceil(iters_per_epoch / grad_accum), 1)
    return max(num_epochs * steps_per_epoch, 1)
