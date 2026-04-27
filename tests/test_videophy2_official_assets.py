from physical_consistency.eval.videophy2_official_assets import (
    build_official_video_manifest_rows,
    build_prompt_manifest_rows,
    download_official_video_subset,
)


def test_build_prompt_manifest_rows_deduplicates_original_prompts():
    rows = [
        {
            "caption": "A person throws a yo-yo against a wall.",
            "upsampled_caption": "A person forcefully throws a red yo-yo against a brick wall.",
            "action": "yoyo",
            "category": "object interaction",
            "is_hard": "0",
        },
        {
            "caption": "A person throws a yo-yo against a wall.",
            "upsampled_caption": "A person forcefully throws a red yo-yo against a brick wall.",
            "action": "yoyo",
            "category": "object interaction",
            "is_hard": "1",
        },
    ]

    output = build_prompt_manifest_rows(rows, prompt_mode="original")

    assert len(output) == 1
    assert output[0]["prompt"] == "A person throws a yo-yo against a wall."
    assert output[0]["is_hard"] == "1"
    assert output[0]["sample_id"] == "videophy2_original_0000"


def test_build_prompt_manifest_rows_keeps_distinct_upsampled_prompts():
    rows = [
        {
            "caption": "A bulldozer moves debris.",
            "upsampled_caption": "A yellow bulldozer clears debris into a dumpster.",
            "action": "bulldozing",
            "category": "construction",
            "is_hard": "0",
        },
        {
            "caption": "A bulldozer moves debris.",
            "upsampled_caption": "A tracked bulldozer pushes broken concrete into a metal dumpster.",
            "action": "bulldozing",
            "category": "construction",
            "is_hard": "0",
        },
    ]

    output = build_prompt_manifest_rows(rows, prompt_mode="upsampled")

    assert len(output) == 2
    assert output[0]["prompt"] != output[1]["prompt"]
    assert output[0]["sample_id"] == "videophy2_upsampled_0000"
    assert output[1]["sample_id"] == "videophy2_upsampled_0001"


def test_build_official_video_manifest_rows_filters_and_uses_requested_prompt_mode():
    rows = [
        {
            "caption": "A person throws a yo-yo against a wall.",
            "upsampled_caption": "A person forcefully throws a red yo-yo against a brick wall.",
            "action": "yoyo",
            "category": "object interaction",
            "is_hard": "1",
            "model_name": "wan",
            "video_url": "https://example.com/wan/yo_yo.mp4",
            "sa": "4",
            "pc": "5",
            "joint": "1",
        },
        {
            "caption": "A bulldozer moves debris.",
            "upsampled_caption": "A tracked bulldozer pushes broken concrete into a metal dumpster.",
            "action": "bulldozing",
            "category": "construction",
            "is_hard": "0",
            "model_name": "hunyuan",
            "video_url": "https://example.com/hunyuan/bulldozer.mp4",
            "sa": "2",
            "pc": "3",
            "joint": "0",
        },
    ]

    output = build_official_video_manifest_rows(
        rows,
        prompt_mode="upsampled",
        hard_only=True,
        model_names=["wan"],
    )

    assert len(output) == 1
    assert output[0]["sample_id"] == "videophy2_official_upsampled_0000"
    assert output[0]["prompt"] == "A person forcefully throws a red yo-yo against a brick wall."
    assert output[0]["videopath"].startswith("wan/")
    assert output[0]["sa_human"] == "4"
    assert output[0]["pc_human"] == "5"


def test_download_official_video_subset_writes_expected_relative_paths(tmp_path):
    rows = [
        {
            "video_url": "https://example.com/model/demo.mp4",
            "videopath": "wan/0000_demo.mp4",
        }
    ]
    calls = []

    def _fake_downloader(url, dest_path, timeout_sec):
        calls.append((url, dest_path, timeout_sec))
        dest_path.write_bytes(b"mp4")

    download_official_video_subset(
        rows,
        video_root=tmp_path / "videos",
        downloader=_fake_downloader,
    )

    assert len(calls) == 1
    assert calls[0][0] == "https://example.com/model/demo.mp4"
    assert calls[0][1] == tmp_path / "videos" / "wan" / "0000_demo.mp4"
    assert (tmp_path / "videos" / "wan" / "0000_demo.mp4").read_bytes() == b"mp4"
