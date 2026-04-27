from physical_consistency.eval.videophy2_official_assets import build_prompt_manifest_rows


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
