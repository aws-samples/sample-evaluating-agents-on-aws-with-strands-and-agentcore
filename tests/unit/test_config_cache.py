"""Tests for the process-wide config cache in run_experiment."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from agentic_evaluation.run_experiment import get_config, reset_config_cache


@pytest.fixture(autouse=True)
def _clean_config_cache() -> Iterator[None]:
    """Isolate each test from a config cached by an earlier import or test."""
    reset_config_cache()
    yield
    reset_config_cache()


def _write_config(directory: Path, project_name: str) -> Path:
    config_path = directory / "eval_config.yaml"
    config_path.write_text(f'project:\n  name: "{project_name}"\n', encoding="utf-8")
    return config_path


def test_get_config_reads_the_yaml_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A rewritten file is not picked up while the config is still cached."""
    monkeypatch.setenv("EVAL_CONFIG_PATH", str(_write_config(tmp_path, "first")))

    assert get_config().project_name == "first"

    _write_config(tmp_path, "second")
    assert get_config().project_name == "first"


def test_reset_config_cache_forces_a_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a reset, the next call re-resolves EVAL_CONFIG_PATH and re-reads."""
    monkeypatch.setenv("EVAL_CONFIG_PATH", str(_write_config(tmp_path, "first")))
    assert get_config().project_name == "first"

    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setenv("EVAL_CONFIG_PATH", str(_write_config(other, "second")))
    reset_config_cache()

    assert get_config().project_name == "second"
