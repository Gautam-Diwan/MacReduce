import json
import math
import sys
from types import ModuleType

import pytest

from macreduce.finetune import (
    MetricLog,
    epoch_steps,
    perplexity,
    prefix_records,
    records_from_texts,
)


def test_perplexity_is_the_exp_of_mean_loss():
    assert perplexity(0) == 1
    assert perplexity(math.log(10)) == pytest.approx(10)
    with pytest.raises(ValueError):
        perplexity(-0.1)


def test_one_epoch_is_one_pass_over_complete_batches():
    assert epoch_steps(23767, 4, 1) == 5941
    assert epoch_steps(4096, 8, 3) == 1536
    assert epoch_steps(10, 4, 2) == 4
    with pytest.raises(ValueError):
        epoch_steps(3, 4, 1)


def test_curve_prefix_keeps_file_order():
    records = [{"text": str(i)} for i in range(10)]
    assert prefix_records(records, 3) == records[:3]
    with pytest.raises(ValueError):
        prefix_records(records, 11)


def test_records_drop_blank_wikitext_lines():
    records = records_from_texts([" Hello ", "", "   ", " = Title = "])
    assert records == [{"text": "Hello"}, {"text": "= Title ="}]


def test_metric_log_adds_validation_perplexity():
    log = MetricLog()
    log.on_train_loss_report({"iteration": 10, "train_loss": 2.0})
    log.on_val_loss_report({"iteration": 10, "val_loss": 0.0})
    assert log.train == [{"iteration": 10, "train_loss": 2.0}]
    assert log.validation[0]["val_perplexity"] == 1


def test_dataset_fetch_pins_revision_and_rejects_wrong_bytes(monkeypatch, tmp_path):
    from macreduce import finetune

    path = tmp_path / "bad.parquet"
    path.write_bytes(b"untrusted wrong revision")
    calls = []
    hub = ModuleType("huggingface_hub")

    def download(*args, **kwargs):
        calls.append((args, kwargs))
        return str(path)

    hub.hf_hub_download = download
    arrow = ModuleType("pyarrow")
    parquet = ModuleType("pyarrow.parquet")
    arrow.parquet = parquet
    monkeypatch.setitem(sys.modules, "pyarrow", arrow)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", parquet)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    with pytest.raises(ValueError, match="checksum mismatch"):
        finetune._fetch_split("train")
    assert calls[0][1]["revision"] == finetune.DATASET_REVISION
    assert calls[0][1]["repo_type"] == "dataset"
    assert calls[0][1]["filename"] == "wikitext-2-raw-v1/train-00000-of-00001.parquet"


def test_old_unverified_dataset_cache_is_not_reused(tmp_path):
    from macreduce.finetune import DATASET_REVISION, SPLITS, _manifest_matches

    for name in SPLITS.values():
        (tmp_path / name).write_text("")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"revision": DATASET_REVISION}))
    assert not _manifest_matches(path)
