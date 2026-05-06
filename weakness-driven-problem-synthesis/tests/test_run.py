from pathlib import Path


def test_skill_package_files_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "pyproject.toml").exists()
    assert (root / "SKILL.md").exists()
    assert (root / "scripts").is_dir()
