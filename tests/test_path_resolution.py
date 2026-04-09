from types import SimpleNamespace

from physical_consistency.common.path_config import resolve_path_config


def test_env_file_override(tmp_path):
    env_file = tmp_path / "paths.env"
    env_file.write_text("BASE_MODEL_DIR=/tmp/base\nDATASET_DIR=/tmp/data\n", encoding="utf-8")
    cfg = resolve_path_config(SimpleNamespace(base_model_dir="", dataset_dir=""), env_file=env_file)
    assert cfg.base_model_dir == "/tmp/base"
    assert cfg.dataset_dir == "/tmp/data"


def test_env_file_expands_previous_keys(tmp_path):
    env_file = tmp_path / "paths.env"
    env_file.write_text(
        "PROJECT_ROOT=/tmp/proj\n"
        "DATASET_DIR=${PROJECT_ROOT}/Dataset/processed_csgo_v3\n"
        "VIDEOPHY2_CKPT_DIR=${PROJECT_ROOT}/models/videophy2\n",
        encoding="utf-8",
    )
    cfg = resolve_path_config(SimpleNamespace(dataset_dir="", videophy2_ckpt_dir=""), env_file=env_file)
    assert cfg.dataset_dir == "/tmp/proj/Dataset/processed_csgo_v3"
    assert cfg.videophy2_ckpt_dir == "/tmp/proj/models/videophy2"
