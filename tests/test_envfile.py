import os

from macreduce.envfile import load_project_env


def test_env_file_fills_missing_token_and_does_not_override(tmp_path, monkeypatch):
    monkeypatch.delenv("MACREDUCE_TOKEN", raising=False)
    (tmp_path / ".env").write_text('MACREDUCE_TOKEN="abcdefghijklmnopqrstuvwx"\n')
    assert load_project_env(tmp_path) == tmp_path / ".env"
    assert os.environ["MACREDUCE_TOKEN"] == "abcdefghijklmnopqrstuvwx"
    monkeypatch.setenv("MACREDUCE_TOKEN", "shell-token-that-is-long-enough")
    (tmp_path / ".env").write_text("MACREDUCE_TOKEN=from-file-should-lose\n")
    load_project_env(tmp_path)
    assert os.environ["MACREDUCE_TOKEN"] == "shell-token-that-is-long-enough"


def test_missing_env_file_is_silent(tmp_path):
    assert load_project_env(tmp_path) is None
