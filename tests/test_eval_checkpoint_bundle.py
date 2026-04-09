from physical_consistency.eval.checkpoint_bundle import materialize_eval_checkpoint_bundle


def _write_branch(root, branch):
    target = root / branch
    target.mkdir(parents=True)
    (target / "config.json").write_text("{}", encoding="utf-8")
    (target / "diffusion_pytorch_model.bin").write_text("x", encoding="utf-8")


def test_materialize_eval_checkpoint_bundle_with_stage1_fallback(tmp_path):
    ft_ckpt = tmp_path / "ft"
    stage1_ckpt = tmp_path / "stage1"
    output_root = tmp_path / "out"

    _write_branch(ft_ckpt, "low_noise_model")
    _write_branch(stage1_ckpt, "low_noise_model")
    _write_branch(stage1_ckpt, "high_noise_model")

    bundle_dir = materialize_eval_checkpoint_bundle(
        ft_ckpt_dir=ft_ckpt,
        stage1_ckpt_dir=stage1_ckpt,
        output_root=output_root,
        experiment_name="exp",
        allow_stage1_fallback=True,
    )
    assert (bundle_dir / "low_noise_model").is_symlink()
    assert (bundle_dir / "high_noise_model").is_symlink()
    assert (bundle_dir / "bundle_manifest.json").exists()
