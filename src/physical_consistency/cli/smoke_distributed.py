"""Minimal distributed smoke test for Accelerate/NCCL startup on H20."""

from __future__ import annotations

import logging
import os
from datetime import timedelta

import accelerate
import torch
import torch.distributed as dist
from accelerate.utils import InitProcessGroupKwargs

try:
    from torch.distributed.elastic.multiprocessing.errors import record
except Exception:  # pragma: no cover
    def record(fn):
        return fn


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
LOGGER = logging.getLogger(__name__)


@record
def main() -> None:
    LOGGER.info(
        "Smoke start env rank=%s local_rank=%s world_size=%s master=%s:%s cuda_visible=%s",
        os.environ.get("RANK", ""),
        os.environ.get("LOCAL_RANK", ""),
        os.environ.get("WORLD_SIZE", ""),
        os.environ.get("MASTER_ADDR", ""),
        os.environ.get("MASTER_PORT", ""),
        os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    )
    init_kwargs = InitProcessGroupKwargs(
        backend="nccl",
        timeout=timedelta(minutes=10),
    )
    accelerator = accelerate.Accelerator(
        kwargs_handlers=[init_kwargs],
        log_with=None,
    )
    LOGGER.info(
        "Accelerator ready rank=%s distributed_type=%s num_processes=%s device=%s is_main=%s",
        accelerator.process_index,
        accelerator.distributed_type,
        accelerator.num_processes,
        accelerator.device,
        accelerator.is_main_process,
    )
    accelerator.wait_for_everyone()
    tensor = torch.tensor([accelerator.process_index + 1.0], device=accelerator.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    LOGGER.info("All-reduce result on rank=%s -> %s", accelerator.process_index, float(tensor.item()))
    accelerator.wait_for_everyone()
    LOGGER.info("Smoke success rank=%s", accelerator.process_index)


if __name__ == "__main__":
    main()
