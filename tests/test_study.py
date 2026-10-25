from macreduce.study import (
    acceptance_from_counts,
    annotate_exact,
    assign_passes,
    draft_seconds,
    execute,
    hindsight_depth,
    render_report,
    trial_plan,
)


def test_acceptance_inverts_the_geometric_model():
    assert acceptance_from_counts(5, 15, 1) == pytest_approx(0.5)
    assert acceptance_from_counts(8, 10, 4) == 1
    assert acceptance_from_counts(0, 4, 0) is None
    assert acceptance_from_counts(4, 4, 2) is None


def pytest_approx(value):
    import pytest

    return pytest.approx(value, abs=1e-3)


def test_negative_draft_fit_is_unknown():
    assert draft_seconds(100, 10, 0, 4, 0.2) is None
    fitted = draft_seconds(10, 10, 4, 2, 0.05)
    assert fitted is not None and fitted > 0


def test_hindsight_and_calibration_do_not_treat_a_mismatch_as_fast():
    records = []
    for index, depth, decode, drafted in (
        (0, 0, 10, 0),
        (1, 2, 30, 8),
        (2, 0, 10, 0),
        (3, 2, 12, 4),
    ):
        records.append(
            {
                "segment": "short",
                "engine": "none" if depth == 0 else "gpu",
                "depth": depth,
                "tokens": [1, 2] if depth == 0 or index == 1 else [1, 9],
                "ttft_seconds": 0.2,
                "decode_tokens_per_second": decode,
                "generated_tokens": 10,
                "from_draft_tokens": drafted,
                "status": "ok",
            }
        )
    annotated = assign_passes(annotate_exact(records))
    assert annotated[1]["exact_match"]
    assert not annotated[3]["exact_match"]
    assert annotated[0]["pass"] == "calibrate"
    assert annotated[2]["pass"] == "evaluate"
    report = render_report(records, None, None)
    short = report["segments"][0]
    assert short["hindsight_decode_depth"] == 2
    assert hindsight_depth(short["depths"], "decode") == 2
    assert report["peer"]["status"] == "not_run"
    assert report["ane_gate"]["placement_verified"] is False


def test_plan_covers_each_depth_twice_and_rejects_one_cycle():
    import pytest

    plan = trial_plan(cycles=2)
    assert sum(item["depth"] == 4 and item["segment"] == "code" for item in plan) == 2
    with pytest.raises(ValueError):
        trial_plan(cycles=1)


def test_execute_records_a_failed_child(tmp_path):
    report = execute(tmp_path, 2, 4, "false")
    assert report["trials"][0]["status"] == "error"
    assert report["protocol"]["models"]["target"] == "mlx-community/Qwen3-1.7B-4bit"
    assert (tmp_path / "report.json").is_file()
