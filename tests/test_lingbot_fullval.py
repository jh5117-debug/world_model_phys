from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

import pytest

from physical_consistency.eval.lingbot_fullval import (
    _prepare_single_gpu_worker_env,
    _build_config,
    build_progress_row,
    shard_rows,
    summarize_physics_iq_outputs_from_rows,
)


def test_shard_rows_round_robins_manifest_rows():
    rows = [{"clip_path": f"val/clips/clip_{idx:04d}", "prompt": f"prompt {idx}"} for idx in range(25)]

    shards = shard_rows(rows, 4)

    assert [len(shard) for shard in shards] == [7, 6, 6, 6]
    assert shards[0][0]["clip_path"] == "val/clips/clip_0000"
    assert shards[0][1]["clip_path"] == "val/clips/clip_0004"
    assert shards[1][0]["clip_path"] == "val/clips/clip_0001"
    assert shards[3][-1]["clip_path"] == "val/clips/clip_0023"


def test_build_progress_row_uses_mean_physics_iq_and_psnr():
    metrics_rows = [
        {"clip_name": "a", "psnr": "10.0"},
        {"clip_name": "b", "psnr": "12.0"},
    ]
    physics_rows = [
        {"sample_id": "a", "physics_iq_style_score": "40.0"},
        {"sample_id": "b", "physics_iq_style_score": "60.0"},
    ]

    row = build_progress_row(
        model_label="LingBot-base",
        processed_count=2,
        total_count=405,
        metrics_rows=metrics_rows,
        physics_rows=physics_rows,
    )

    assert row == {
        "Model": "LingBot-base",
        "Processed": 2,
        "Total": 405,
        "Mean Physics-IQ Score": 50.0,
        "Mean PSNR": 11.0,
    }


def test_summarize_physics_iq_outputs_from_rows_aggregates_numeric_fields():
    rows = [
        {
            "compare_frame_count": "40",
            "mse_mean": "0.10",
            "spatiotemporal_iou_mean": "0.20",
            "spatial_iou": "0.90",
            "weighted_spatial_iou": "0.50",
            "physics_iq_style_score": "45.0",
        },
        {
            "compare_frame_count": "40",
            "mse_mean": "0.06",
            "spatiotemporal_iou_mean": "0.30",
            "spatial_iou": "0.95",
            "weighted_spatial_iou": "0.70",
            "physics_iq_style_score": "55.0",
        },
    ]

    summary = summarize_physics_iq_outputs_from_rows(rows)

    assert summary["count"] == 2
    assert summary["means"]["physics_iq_style_score"]["mean"] == 50.0
    assert summary["means"]["mse_mean"]["mean"] == 0.08


def test_build_config_allows_missing_physics_config(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "physical_consistency.eval.lingbot_fullval.resolve_path_config",
        lambda args, env_file=None: SimpleNamespace(
            dataset_dir="/tmp/dataset",
            output_root="/tmp/output",
            base_model_dir="/tmp/base",
            stage1_ckpt_dir="/tmp/stage1",
        ),
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")

    cfg = _build_config(
        Namespace(
            env_file="",
            dataset_dir="",
            manifest_path="",
            output_root="",
            base_model_dir="",
            stage1_ckpt_dir="",
            val_inf_root="",
            seed=0,
            frame_num=81,
            sample_steps=70,
            guide_scale=5.0,
            height=480,
            width=832,
            control_type="act",
            models="both",
            video_filename="video.mp4",
            video_suffix="_gen.mp4",
            report_every=10,
            poll_seconds=15,
            num_gpus=2,
            ulysses_size=1,
        )
    )

    assert cfg.physics_config == "/tmp/output/configs/physics_iq_dataset_eval.yaml"
    assert cfg.gpu_list == ["0", "1"]


def test_prepare_single_gpu_worker_env_clears_distributed_launch_state():
    prepared = _prepare_single_gpu_worker_env(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29500",
            "WORLD_SIZE": "4",
            "RANK": "2",
            "LOCAL_RANK": "2",
            "GROUP_RANK": "0",
            "ROLE_RANK": "0",
            "NODE_RANK": "0",
            "ACCELERATE_USE_DEEPSPEED": "true",
            "TOKENIZERS_PARALLELISM": "true",
            "UNCHANGED": "ok",
        },
        "7",
    )

    assert prepared["CUDA_VISIBLE_DEVICES"] == "7"
    assert prepared["WORLD_SIZE"] == "1"
    assert prepared["RANK"] == "0"
    assert prepared["LOCAL_RANK"] == "0"
    assert prepared["TOKENIZERS_PARALLELISM"] == "false"
    assert prepared["UNCHANGED"] == "ok"
    assert "MASTER_ADDR" not in prepared
    assert "MASTER_PORT" not in prepared
    assert "ACCELERATE_USE_DEEPSPEED" not in prepared
