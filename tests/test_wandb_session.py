from physical_consistency.wandb_utils.session import _normalize_wandb_target


def test_normalize_wandb_target_keeps_explicit_entity():
    entity, project = _normalize_wandb_target(entity="WorldModel_11", project="intro-example")
    assert entity == "WorldModel_11"
    assert project == "intro-example"


def test_normalize_wandb_target_splits_combined_project_path():
    entity, project = _normalize_wandb_target(entity="", project="WorldModel_11/intro-example")
    assert entity == "WorldModel_11"
    assert project == "intro-example"
