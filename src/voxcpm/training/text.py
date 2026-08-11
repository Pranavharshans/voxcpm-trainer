"""Text preparation shared by VoxCPM training utilities."""

from __future__ import annotations

import unicodedata


SUPPORTED_UNICODE_NORMALIZATIONS = {"NFC", "NFD", "NFKC", "NFKD"}


def normalize_training_text(text: str, normalization: str = "NFC") -> str:
    """Normalize a transcript before text tokenization.

    NFC is the default because Malayalam vowel signs and combining marks can be
    represented by canonically equivalent Unicode sequences. Keeping one
    canonical representation prevents equivalent transcripts from producing
    different token sequences.
    """

    if not isinstance(text, str):
        raise TypeError(f"Training text must be a string, got {type(text).__name__}")

    form = normalization.upper()
    if form in {"", "NONE"}:
        return text
    if form not in SUPPORTED_UNICODE_NORMALIZATIONS:
        choices = ", ".join(sorted(SUPPORTED_UNICODE_NORMALIZATIONS))
        raise ValueError(f"Unsupported text normalization {normalization!r}; use one of {choices}, or 'none'")
    return unicodedata.normalize(form, text)
