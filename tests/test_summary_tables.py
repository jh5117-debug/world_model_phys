from __future__ import annotations

from physical_consistency.common.summary_tables import (
    format_csgo_metrics_summary,
    format_physics_iq_summary,
    format_videophy2_summary,
)


def test_format_physics_iq_summary_renders_markdown_tables():
    summary = {
        "seeds": [
            {
                "seed": 0,
                "count": 1,
                "means": {
                    "physics_iq_style_score": {"mean": 43.65, "count": 1},
                    "spatiotemporal_iou_mean": {"mean": 0.155923, "count": 1},
                    "spatial_iou": {"mean": 0.961819, "count": 1},
                    "weighted_spatial_iou": {"mean": 0.504235, "count": 1},
                    "mse_mean": {"mean": 0.104202, "count": 1},
                    "compare_frame_count": {"mean": 40.0, "count": 1},
                },
            }
        ],
        "means": {
            "physics_iq_style_score": {"mean": 43.65, "count": 1},
            "spatiotemporal_iou_mean": {"mean": 0.155923, "count": 1},
            "spatial_iou": {"mean": 0.961819, "count": 1},
            "weighted_spatial_iou": {"mean": 0.504235, "count": 1},
            "mse_mean": {"mean": 0.104202, "count": 1},
            "compare_frame_count": {"mean": 40.0, "count": 1},
        },
    }

    rendered = format_physics_iq_summary(summary, title="Physics-IQ Summary: demo")

    assert "Physics-IQ Summary: demo" in rendered
    assert "| Metric | Mean | Count |" in rendered
    assert "| Score | 43.65 | 1 |" in rendered
    assert "| Seed | Samples | Score |" in rendered
    assert "| 0 | 1 | 43.65 |" in rendered


def test_format_csgo_metrics_summary_omits_missing_metrics():
    summary = {
        "seeds": [
            {
                "seed": 0,
                "count": 1,
                "means": {
                    "psnr": {"mean": 9.6551, "count": 1},
                    "gen_time_s": {"mean": 420.5, "count": 1},
                },
            }
        ],
        "means": {
            "psnr": {"mean": 9.6551, "count": 1},
            "gen_time_s": {"mean": 420.5, "count": 1},
        },
    }

    rendered = format_csgo_metrics_summary(summary, title="CSGO Eval Summary: demo")

    assert "CSGO Eval Summary: demo" in rendered
    assert "PSNR" in rendered
    assert "Gen Time (s)" in rendered
    assert "FID" not in rendered


def test_format_videophy2_summary_renders_joint_score():
    summary = {
        "seeds": [{"seed": 0, "count": 4, "means": {"sa_mean": {"mean": 4.2, "count": 1}, "pc_mean": {"mean": 3.8, "count": 1}, "joint": {"mean": 0.5, "count": 1}}}],
        "means": {
            "sa_mean": {"mean": 4.2, "count": 1},
            "pc_mean": {"mean": 3.8, "count": 1},
            "joint": {"mean": 0.5, "count": 1},
        },
    }

    rendered = format_videophy2_summary(summary, title="VideoPhy-2 Summary: demo")

    assert "VideoPhy-2 Summary: demo" in rendered
    assert "Joint >= 4" in rendered
