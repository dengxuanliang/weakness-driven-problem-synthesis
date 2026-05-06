"""Prompt and reference file loaders."""

from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REFERENCES_DIR = PACKAGE_ROOT / "references"
PROMPTS_DIR = REFERENCES_DIR / "prompts"


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text()


def load_reference(name: str) -> str:
    return (REFERENCES_DIR / name).read_text()
