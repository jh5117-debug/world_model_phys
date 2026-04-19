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


def test_materialize_eval_checkpoint_bundle_supports_relative_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ft_ckpt = tmp_path / "checkpoints" / "exp" / "epoch_1"
    output_root = tmp_path / "relative_out"

    _write_branch(ft_ckpt, "low_noise_model")
    _write_branch(ft_ckpt, "high_noise_model")

    bundle_dir = materialize_eval_checkpoint_bundle(
        ft_ckpt_dir="checkpoints/exp/epoch_1",
        output_root="relative_out",
        experiment_name="exp_epoch_1",
        allow_stage1_fallback=False,
    )

    assert (bundle_dir / "low_noise_model").exists()
    assert (bundle_dir / "high_noise_model").exists()
    assert (bundle_dir / "low_noise_model" / "diffusion_pytorch_model.bin").exists()


def test_materialize_eval_checkpoint_bundle_uses_companion_before_stage1(tmp_path):
    ft_ckpt = tmp_path / "ft_high"
    companion_ckpt = tmp_path / "trained_low"
    stage1_ckpt = tmp_path / "stage1"
    output_root = tmp_path / "out"

    _write_branch(ft_ckpt, "high_noise_model")
    _write_branch(companion_ckpt, "low_noise_model")
    _write_branch(stage1_ckpt, "low_noise_model")
    _write_branch(stage1_ckpt, "high_noise_model")

    bundle_dir = materialize_eval_checkpoint_bundle(
        ft_ckpt_dir=ft_ckpt,
        companion_ckpt_dir=companion_ckpt,
        stage1_ckpt_dir=stage1_ckpt,
        output_root=output_root,
        experiment_name="exp_high",
        allow_stage1_fallback=True,
    )

    assert (bundle_dir / "low_noise_model").resolve() == (companion_ckpt / "low_noise_model").resolve()
    assert (bundle_dir / "high_noise_model").resolve() == (ft_ckpt / "high_noise_model").resolve()
