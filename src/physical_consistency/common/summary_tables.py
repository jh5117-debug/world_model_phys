"""Human-readable summary tables for terminal output."""

from __future__ import annotations

from typing import Any


def _fmt_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    rendered_rows = [[_fmt_value(cell) for cell in row] for row in rows]
    rendered_headers = [_fmt_value(header) for header in headers]
    separator = ["---"] * len(headers)
    table_rows = [rendered_headers, separator, *rendered_rows]
    return "\n".join("| " + " | ".join(row) + " |" for row in table_rows)


def _metric_rows(
    summary: dict[str, Any],
    *,
    metric_order: list[str],
    metric_labels: dict[str, str] | None = None,
) -> list[list[Any]]:
    metric_labels = metric_labels or {}
    rows: list[list[Any]] = []
    means = summary.get("means", {})
    for metric in metric_order:
        payload = means.get(metric)
        if not payload:
            continue
        rows.append(
            [
                metric_labels.get(metric, metric),
                payload.get("mean", ""),
                payload.get("count", ""),
            ]
        )
    return rows


def format_seed_mean_summary(
    summary: dict[str, Any],
    *,
    title: str,
    metric_order: list[str],
    metric_labels: dict[str, str] | None = None,
    sample_count_key: str = "count",
    include_per_seed: bool = True,
) -> str:
    """Render a seed-based summary.json payload as human-readable tables."""
    metric_labels = metric_labels or {}
    sections: list[str] = [title]

    overall_rows = _metric_rows(summary, metric_order=metric_order, metric_labels=metric_labels)
    if overall_rows:
        sections.append("")
        sections.append("Overall")
        sections.append(_markdown_table(["Metric", "Mean", "Count"], overall_rows))

    if include_per_seed:
        seed_rows: list[list[Any]] = []
        seed_entries = summary.get("seeds", [])
        active_seed_metrics = [
            metric
            for metric in metric_order
            if any(item.get("means", {}).get(metric) for item in seed_entries)
        ]
        for item in seed_entries:
            row = [item.get("seed", ""), item.get(sample_count_key, "")]
            has_metric = False
            item_means = item.get("means", {})
            for metric in active_seed_metrics:
                payload = item_means.get(metric)
                value = ""
                if payload:
                    value = payload.get("mean", "")
                    has_metric = True
                row.append(value)
            if has_metric:
                seed_rows.append(row)

        if seed_rows:
            sections.append("")
            sections.append("Per Seed")
            headers = ["Seed", "Samples"] + [metric_labels.get(metric, metric) for metric in active_seed_metrics]
            sections.append(_markdown_table(headers, seed_rows))

    return "\n".join(sections).strip()


def format_physics_iq_summary(summary: dict[str, Any], *, title: str = "Physics-IQ Summary") -> str:
    metric_order = [
        "physics_iq_style_score",
        "spatiotemporal_iou_mean",
        "spatial_iou",
        "weighted_spatial_iou",
        "mse_mean",
        "compare_frame_count",
    ]
    metric_labels = {
        "physics_iq_style_score": "Score",
        "spatiotemporal_iou_mean": "Spatiotemporal IoU",
        "spatial_iou": "Spatial IoU",
        "weighted_spatial_iou": "Weighted Spatial IoU",
        "mse_mean": "MSE",
        "compare_frame_count": "Frames",
    }
    return format_seed_mean_summary(
        summary,
        title=title,
        metric_order=metric_order,
        metric_labels=metric_labels,
    )


def format_videophy2_summary(
    summary: dict[str, Any],
    *,
    title: str = "VideoPhy-2 Summary",
    include_per_seed: bool = True,
) -> str:
    metric_order = ["sa_mean", "pc_mean", "joint"]
    metric_labels = {
        "sa_mean": "SA Mean",
        "pc_mean": "PC Mean",
        "joint": "Joint >= 4",
    }
    return format_seed_mean_summary(
        summary,
        title=title,
        metric_order=metric_order,
        metric_labels=metric_labels,
        include_per_seed=include_per_seed,
    )


def format_csgo_metrics_summary(summary: dict[str, Any], *, title: str = "CSGO Eval Summary") -> str:
    metric_order = [
        "psnr",
        "ssim",
        "lpips",
        "gen_time_s",
        "fid",
        "fvd",
        "flow_direction_accuracy",
        "trajectory_consistency",
        "turn_direction_accuracy",
    ]
    metric_labels = {
        "psnr": "PSNR",
        "ssim": "SSIM",
        "lpips": "LPIPS",
        "gen_time_s": "Gen Time (s)",
        "fid": "FID",
        "fvd": "FVD",
        "flow_direction_accuracy": "Flow Dir Acc",
        "trajectory_consistency": "Trajectory Consistency",
        "turn_direction_accuracy": "Turn Dir Acc",
    }
    return format_seed_mean_summary(
        summary,
        title=title,
        metric_order=metric_order,
        metric_labels=metric_labels,
    )


def format_lingbot_progress_summary(
    rows: list[dict[str, Any]],
    *,
    title: str = "LingBot Eval Progress",
) -> str:
    """Render rolling full-val LingBot progress as a compact terminal table."""
    headers = [
        "Model",
        "Processed",
        "Total",
        "PMF ↑",
        "PSNR ↑",
        "SSIM ↑",
        "LPIPS ↓",
        "FVD ↓",
    ]
    table_rows = [
        [
            row.get("Model", ""),
            row.get("Processed", ""),
            row.get("Total", ""),
            row.get("PMF ↑", ""),
            row.get("PSNR ↑", ""),
            row.get("SSIM ↑", ""),
            row.get("LPIPS ↓", ""),
            row.get("FVD ↓", ""),
        ]
        for row in rows
    ]
    sections = [title]
    if table_rows:
        sections.extend(["", "Overview", _markdown_table(headers, table_rows)])
    return "\n".join(sections).strip()


def format_lingbot_generation_summary(
    rows: list[dict[str, Any]],
    *,
    title: str = "LingBot Generation Progress",
) -> str:
    """Render generation-only LingBot progress as a compact terminal table."""
    headers = ["Model", "Processed", "Total"]
    table_rows = [
        [
            row.get("Model", ""),
            row.get("Processed", ""),
            row.get("Total", ""),
        ]
        for row in rows
    ]
    sections = [title]
    if table_rows:
        sections.extend(["", "Overview", _markdown_table(headers, table_rows)])
    return "\n".join(sections).strip()
