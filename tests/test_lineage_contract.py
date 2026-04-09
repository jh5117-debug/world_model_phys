from physical_consistency.lineage.contract import verify_stage1_checkpoint


def test_verify_stage1_checkpoint(tmp_path):
    ckpt = tmp_path / "epoch_2"
    for subdir in ["low_noise_model", "high_noise_model"]:
        target = ckpt / subdir
        target.mkdir(parents=True)
        (target / "config.json").write_text("{}", encoding="utf-8")
        (target / "diffusion_pytorch_model.bin").write_text("x", encoding="utf-8")

    ok, errors = verify_stage1_checkpoint(ckpt)
    assert ok is True
    assert errors == []
