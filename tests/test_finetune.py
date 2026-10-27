import math

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
