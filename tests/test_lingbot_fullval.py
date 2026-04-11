from __future__ import annotations

from physical_consistency.eval.lingbot_fullval import (
    build_progress_row,
    chunk_rows,
    summarize_physics_iq_outputs_from_rows,
)


def test_chunk_rows_splits_manifest_sequentially():
    rows = [{"clip_path": f"val/clips/clip_{idx:04d}", "prompt": f"prompt {idx}"} for idx in range(25)]

    chunks = chunk_rows(rows, 10)

    assert [len(chunk) for chunk in chunks] == [10, 10, 5]
    assert chunks[0][0]["clip_path"] == "val/clips/clip_0000"
    assert chunks[1][0]["clip_path"] == "val/clips/clip_0010"
    assert chunks[2][-1]["clip_path"] == "val/clips/clip_0024"


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
