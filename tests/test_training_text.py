from __future__ import annotations

import importlib.util
import sys
import unicodedata
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TEXT_MODULE = ROOT / "src" / "voxcpm" / "training" / "text.py"
spec = importlib.util.spec_from_file_location("voxcpm_training_text", TEXT_MODULE)
text_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = text_module
assert spec.loader is not None
spec.loader.exec_module(text_module)

normalize_training_text = text_module.normalize_training_text


def test_normalizes_canonically_equivalent_text_to_nfc():
    decomposed = unicodedata.normalize("NFD", "മലയാളം")

    assert normalize_training_text(decomposed) == unicodedata.normalize("NFC", decomposed)


def test_none_preserves_original_codepoints():
    decomposed = unicodedata.normalize("NFD", "മലയാളം")

    assert normalize_training_text(decomposed, "none") == decomposed


def test_rejects_unknown_normalization():
    with pytest.raises(ValueError, match="Unsupported text normalization"):
        normalize_training_text("മലയാളം", "malayalam")
