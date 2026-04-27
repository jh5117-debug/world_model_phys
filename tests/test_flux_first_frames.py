from pathlib import Path

from physical_consistency.eval.flux_first_frames import (
    TURBO_SIGMAS,
    build_first_frame_jobs,
    generate_first_frames,
)


class _FakeResult:
    def __init__(self, image):
        self.images = [image]


class _FakePipeline:
    def __init__(self, image):
        self.image = image
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResult(self.image)


def test_build_first_frame_jobs_produces_stable_paths_and_seeds(tmp_path):
    manifest = tmp_path / "videophy2.csv"
    manifest.write_text(
        "sample_id,prompt,source_mode,is_hard\n"
        "videophy2_original_0000,fold a shirt,original,1\n",
        encoding="utf-8",
    )

    jobs = build_first_frame_jobs(
        manifest_csv=manifest,
        output_dir=tmp_path / "frames",
        base_seed=42,
    )

    assert len(jobs) == 1
    assert jobs[0].sample_id == "videophy2_original_0000"
    assert jobs[0].clip_dir == tmp_path / "frames" / "samples" / "videophy2_original_0000"
    assert jobs[0].image_path == jobs[0].clip_dir / "image.jpg"
    assert jobs[0].seed == build_first_frame_jobs(
        manifest_csv=manifest,
        output_dir=tmp_path / "frames",
        base_seed=42,
    )[0].seed


def test_generate_first_frames_writes_images_and_manifest(tmp_path):
    from PIL import Image

    manifest = tmp_path / "videophy2.csv"
    manifest.write_text(
        "sample_id,prompt,source_mode,is_hard\n"
        "videophy2_upsampled_0000,fold blue jeans,upsampled,0\n",
        encoding="utf-8",
    )
    jobs = build_first_frame_jobs(
        manifest_csv=manifest,
        output_dir=tmp_path / "frames",
        base_seed=7,
    )
    fake_pipeline = _FakePipeline(Image.new("RGB", (64, 64), color=(10, 20, 30)))

    rows = generate_first_frames(
        jobs,
        output_manifest_csv=tmp_path / "generated.csv",
        model_id="black-forest-labs/FLUX.2-dev",
        turbo_lora_id="fal/FLUX.2-dev-Turbo",
        height=480,
        width=832,
        pipeline=fake_pipeline,
    )

    assert len(rows) == 1
    assert Path(rows[0]["image_path"]).exists()
    assert rows[0]["model_id"] == "black-forest-labs/FLUX.2-dev"
    assert rows[0]["turbo_lora_id"] == "fal/FLUX.2-dev-Turbo"
    assert fake_pipeline.calls[0]["sigmas"] == TURBO_SIGMAS
    assert fake_pipeline.calls[0]["num_inference_steps"] == 8
