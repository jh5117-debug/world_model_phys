from __future__ import annotations

from pathlib import Path
import sys

from physical_consistency.eval import csgo_metrics


def test_single_gpu_eval_disables_fsdp_and_passs_control_type(tmp_path, monkeypatch):
    manifest_csv = tmp_path / "metadata_val.csv"
    manifest_csv.write_text("clip_path,prompt\nval/clips/example,prompt\n", encoding="utf-8")

    base_model_dir = tmp_path / "links" / "base_model"
    lingbot_code_dir = tmp_path / "code" / "lingbot-world"
    dataset_dir = tmp_path / "Dataset" / "processed_csgo_v3"
    finetune_code_dir = tmp_path / "code" / "finetune"
    for path in [base_model_dir, lingbot_code_dir, dataset_dir, finetune_code_dir]:
        path.mkdir(parents=True, exist_ok=True)

    captured = {}

    def fake_materialize_dataset_view(_dataset_dir, _manifest_path, view_dir):
        (view_dir / "val" / "clips").mkdir(parents=True, exist_ok=True)

    def fake_run_command(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs

    monkeypatch.setattr(csgo_metrics, "materialize_dataset_view", fake_materialize_dataset_view)
    monkeypatch.setattr(csgo_metrics, "run_command", fake_run_command)
    monkeypatch.setattr(csgo_metrics, "_find_free_port", lambda: 45678)

    cfg = csgo_metrics.CSGOEvalConfig(
        experiment_name="exp_base_smoke_one",
        split="val",
        manifest_path=str(manifest_csv),
        frame_num=81,
        sample_steps=70,
        guide_scale=5.0,
        height=480,
        width=832,
        num_gpus=1,
        ulysses_size=8,
        run_fid_fvd=False,
        run_action_control=False,
        run_videophy2=False,
        base_model_dir=str(base_model_dir),
        ft_ckpt_dir="",
        output_root=str(tmp_path),
        seed_list=[0],
        control_type="act",
    )

    csgo_metrics.run_single_seed_eval(
        cfg,
        finetune_code_dir=str(finetune_code_dir),
        dataset_dir=str(dataset_dir),
        output_root=str(tmp_path),
        seed=0,
        base_model_dir=str(base_model_dir),
        lingbot_code_dir=str(lingbot_code_dir),
        ft_ckpt_dir="",
    )

    cmd = captured["cmd"]
    assert cmd[:7] == [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=1",
        "--master_port",
        "45678",
        "eval_batch.py",
    ]
    assert "--control_type" in cmd
    assert cmd[cmd.index("--control_type") + 1] == "act"
    assert "--ulysses_size" in cmd
    assert cmd[cmd.index("--ulysses_size") + 1] == "1"
    assert "--dit_fsdp" not in cmd
    assert "--t5_fsdp" not in cmd
