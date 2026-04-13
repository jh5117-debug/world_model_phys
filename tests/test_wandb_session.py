import sys
from types import SimpleNamespace

import numpy as np

from physical_consistency.wandb_utils.session import _normalize_wandb_target
from physical_consistency.wandb_utils.media import relation_matrix_image


def test_normalize_wandb_target_keeps_explicit_entity():
    entity, project = _normalize_wandb_target(entity="WorldModel_11", project="intro-example")
    assert entity == "WorldModel_11"
    assert project == "intro-example"


def test_normalize_wandb_target_splits_combined_project_path():
    entity, project = _normalize_wandb_target(entity="", project="WorldModel_11/intro-example")
    assert entity == "WorldModel_11"
    assert project == "intro-example"


def test_relation_matrix_image_uses_numpy_image_for_wandb(monkeypatch):
    captured = {}

    class _FakeImage:
        def __init__(self, data, caption=""):
            captured["data"] = data
            captured["caption"] = caption

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Image=_FakeImage))

    image = relation_matrix_image(np.ones((4, 4), dtype=np.float32), "student_spatial")

    assert isinstance(image, _FakeImage)
    assert isinstance(captured["data"], np.ndarray)
    assert captured["data"].ndim == 3
    assert captured["caption"] == "student_spatial"
