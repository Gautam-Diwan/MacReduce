from pathlib import Path

import pytest

from macreduce.lora_client import BackendUnavailable, require_ane, write_client_shard


def test_client_shard_keeps_text_and_leaves_eval_empty(tmp_path: Path):
    write_client_shard(tmp_path, [{"text": "alpha"}, {"text": "beta"}])
    assert (tmp_path / "train.jsonl").read_text().count("\n") == 2
    assert (tmp_path / "valid.jsonl").read_text() == ""
    assert (tmp_path / "test.jsonl").read_text() == ""


def test_ane_stack_does_not_fall_back_to_the_gpu():
    with pytest.raises(BackendUnavailable) as caught:
        require_ane("aneforge")
    message = str(caught.value)
    assert "aneforge" in message
    assert "gpu" not in message.lower()
