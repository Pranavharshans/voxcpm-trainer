from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG_MODULE = ROOT / "src" / "voxcpm" / "training" / "config.py"


@pytest.fixture
def config_module(monkeypatch):
    argbind_stub = type(sys)("argbind")
    argbind_stub.parse_args = lambda *args, **kwargs: {}
    argbind_stub.scope = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "argbind", argbind_stub)

    spec = importlib.util.spec_from_file_location("voxcpm_training_config", CONFIG_MODULE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_load_yaml_config_expands_environment(config_module, monkeypatch, tmp_path):
    monkeypatch.setenv("VOXCPM_RUN_DIR", "/work/run-01")
    config = tmp_path / "train.yaml"
    config.write_text("save_path: ${VOXCPM_RUN_DIR}/checkpoints\n", encoding="utf-8")

    assert config_module.load_yaml_config(config)["save_path"] == "/work/run-01/checkpoints"


def test_load_yaml_config_rejects_missing_environment(config_module, monkeypatch, tmp_path):
    monkeypatch.delenv("MALAYALAM_TRAIN_MANIFEST", raising=False)
    config = tmp_path / "train.yaml"
    config.write_text("train_manifest: ${MALAYALAM_TRAIN_MANIFEST}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="MALAYALAM_TRAIN_MANIFEST"):
        config_module.load_yaml_config(config)
