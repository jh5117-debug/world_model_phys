from __future__ import annotations

from physical_consistency.common.summary_tables import (
    format_csgo_metrics_summary,
    format_lingbot_generation_summary,
    format_lingbot_progress_summary,
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


def test_format_videophy2_summary_can_hide_per_seed_block():
    summary = {
        "seeds": [
            {
                "seed": 42,
                "count": 8,
                "means": {
                    "sa_mean": {"mean": 4.2, "count": 1},
                    "pc_mean": {"mean": 3.8, "count": 1},
                    "joint": {"mean": 0.5, "count": 1},
                },
            }
        ],
        "means": {
            "sa_mean": {"mean": 4.2, "count": 1},
            "pc_mean": {"mean": 3.8, "count": 1},
            "joint": {"mean": 0.5, "count": 1},
        },
    }

    rendered = format_videophy2_summary(
        summary,
        title="Lingbot_VideoREPA",
        include_per_seed=False,
    )

    assert "Lingbot_VideoREPA" in rendered
    assert "Overall" in rendered
    assert "Per Seed" not in rendered


def test_format_lingbot_progress_summary_renders_combined_table():
    rendered = format_lingbot_progress_summary(
        [
            {
                "Model": "LingBot-base",
                "Processed": 10,
                "Total": 405,
                "Mean Physics-IQ Score": 43.65,
                "Mean PSNR": 9.6551,
            },
            {
                "Model": "LingBot-Stage1",
                "Processed": 10,
                "Total": 405,
                "Mean Physics-IQ Score": 59.25,
                "Mean PSNR": 11.5251,
            },
        ],
        title="LingBot Full-Val Final Summary",
    )

    assert "LingBot Full-Val Final Summary" in rendered
    assert "| Model | Processed | Total | Mean Physics-IQ Score | Mean PSNR |" in rendered
    assert "| LingBot-base | 10 | 405 | 43.65 | 9.6551 |" in rendered
    assert "| LingBot-Stage1 | 10 | 405 | 59.25 | 11.5251 |" in rendered


def test_format_lingbot_generation_summary_renders_basic_table():
    rendered = format_lingbot_generation_summary(
        [
            {"Model": "LingBot-base", "Processed": 10, "Total": 80},
            {"Model": "LingBot-Stage1", "Processed": 4, "Total": 80},
        ],
        title="LingBot Generation Final Summary",
    )

    assert "LingBot Generation Final Summary" in rendered
    assert "| Model | Processed | Total |" in rendered
    assert "| LingBot-base | 10 | 80 |" in rendered
    assert "| LingBot-Stage1 | 4 | 80 |" in rendered
