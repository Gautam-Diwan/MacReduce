import json

from macreduce.observe import Events, normalize_mactop, parse_mactop, report


def test_missing_power_is_not_zero_and_zero_is_preserved():
    absent = normalize_mactop({"soc_metrics": {"cpu_power": 1.2}})
    assert absent["ane_power_w"] is None
    zero = normalize_mactop({"soc_metrics": {"ane_power": 0}})
    assert zero["ane_power_w"] == 0
    assert zero["scope"] == "system"
    assert "not attributable" in zero["attribution"]


def test_mactop_stream_variants():
    values = [{"timestamp": "a"}, {"timestamp": "b"}]
    for text in (
        json.dumps(values),
        "\n".join(json.dumps(x) for x in values),
        "".join(json.dumps([x], indent=2) for x in values),
    ):
        assert list(parse_mactop(text)) == values


def test_report_produces_separate_node_traces(tmp_path):
    for n in ("a", "b"):
        e = Events(tmp_path / "events", n)
        with e.span("train_step", tokens=16):
            pass
    path = report(tmp_path)
    assert path.exists()
    for n in ("a", "b"):
        trace = json.loads((tmp_path / f"{n}.trace.json").read_text())
        assert trace["traceEvents"][0]["ts"] == 0
