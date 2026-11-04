from macreduce.telemetry import headless_series, integrate_joules, parse_prometheus


def test_prometheus_power_gauges():
    text = """
# HELP mactop_power_watts Current power usage in watts
mactop_power_watts{component="ane"} 0.5
mactop_power_watts{component="gpu"} 2.25
mactop_cpu_usage_percent 10
"""
    assert parse_prometheus(text) == {"ane": 0.5, "gpu": 2.25}


def test_trapezoid_joules():
    samples = [
        {"t": 0.0, "gpu": 1.0, "ane": 0.0},
        {"t": 2.0, "gpu": 3.0, "ane": 2.0},
    ]
    assert integrate_joules(samples, "gpu") == 4.0
    assert integrate_joules(samples, "ane") == 2.0
    assert integrate_joules([{"t": 0.0, "gpu": 1.0}], "gpu") is None


def test_headless_series_keeps_memory_and_stops_on_partial_json():
    text = (
        '{"timestamp":"2026-10-28T19:00:00-04:00","soc_metrics":{"gpu_power":1.5,'
        '"ane_power":0.2},"memory":{"used":1000000000,"total":8000000000}}'
        '{"timestamp":"2026-10-28T19:00:01-04:00","soc'
    )
    rows = headless_series(text)
    assert len(rows) == 1
    assert rows[0]["gpu"] == 1.5
    assert rows[0]["memory_used_bytes"] == 1000000000
