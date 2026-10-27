import pytest

from macreduce.placement import CostTable, EngineCost, choose, choose_device, expected_tokens, fixed
from macreduce.serving import (
    Prompt,
    Trial,
    abba,
    beats,
    builtin_trace,
    noise_floor,
    run_abba,
    sweep_order,
    token_diff,
)


def table(**overrides) -> CostTable:
    base = dict(
        target_seconds_per_token=1.0,
        verify_seconds=0.2,
        verify_seconds_per_extra_token=0.0,
        acceptance=0.8,
        engines={
            "gpu": EngineCost(0.05, available=True, placement_verified=True),
        },
        prefix_tokens=1000,
        kv_bytes=1_000_000,
        prefill_tokens_per_second=100.0,
        disk_bytes_per_second=None,
        link_bytes_per_second=None,
    )
    base.update(overrides)
    return CostTable(**base)


def test_expected_tokens_matches_geometric_sum():
    assert expected_tokens(0, 0.3) == 1
    assert expected_tokens(4, 1) == 5
    assert expected_tokens(2, 0) == 1
    assert expected_tokens(1, 0.5) == pytest.approx(1.5)


def test_cheap_accepted_draft_uses_the_longest_depth():
    decision = choose(table(acceptance=1, engines={"gpu": EngineCost(1e-4, True, True)}))
    assert decision.engine == "gpu"
    assert decision.depth == 8


def test_rejected_drafts_stay_on_the_target():
    decision = choose(
        table(acceptance=0, engines={"gpu": EngineCost(1.0, True, True)}, verify_seconds=1.0)
    )
    assert decision.engine == "none"
    assert decision.depth == 0


def test_unverified_ane_is_not_selected_even_when_its_cost_is_lowest():
    decision = choose(
        table(
            acceptance=1,
            engines={
                "gpu": EngineCost(0.5, True, True),
                "ane": EngineCost(1e-6, True, False),
            },
        )
    )
    assert decision.engine != "ane"
    assert any("ANE" in reason for reason in decision.reasons)


def test_verified_ane_can_win_on_cost():
    decision = choose(
        table(
            acceptance=1,
            engines={"ane": EngineCost(1e-4, True, True)},
            verify_seconds=0.05,
        )
    )
    assert decision.engine == "ane"
    assert decision.depth == 8


def test_unknown_acceptance_disables_speculation():
    decision = choose(table(acceptance=None))
    assert (decision.engine, decision.depth) == ("none", 0)


def test_missing_link_is_not_a_free_peer_transfer():
    decision = choose(table(link_bytes_per_second=None, disk_bytes_per_second=1e9))
    assert decision.prefix == "disk"
    assert "link rate unavailable" in decision.reasons


def test_recompute_wins_when_the_cache_is_slower_than_prefill():
    decision = choose(
        table(prefill_tokens_per_second=1_000_000, disk_bytes_per_second=1, kv_bytes=1_000_000)
    )
    assert decision.prefix == "recompute"
    assert decision.prefix_seconds == pytest.approx(0.001)


def test_no_prefix_measurement_stays_unknown():
    decision = choose(
        table(
            prefix_tokens=10,
            prefill_tokens_per_second=None,
            kv_bytes=None,
            disk_bytes_per_second=None,
            link_bytes_per_second=None,
        )
    )
    assert decision.prefix is None
    assert decision.prefix_seconds is None


def test_static_configuration_does_not_consult_the_cost_table():
    choice = fixed("gpu", 4, "recompute")
    assert choice.seconds_per_token is None
    with pytest.raises(ValueError):
        fixed("gpu", 0, "recompute")


def test_abba_and_sweep_orders():
    assert abba(2) == ["A", "B", "B", "A", "A", "B", "B", "A"]
    assert sweep_order((0, 1, 2), 2) == [0, 1, 2, 2, 1, 0]
    assert [prompt.segment for prompt in builtin_trace()] == ["short", "long", "code"]


def test_noise_floor_and_both_metrics_required():
    assert noise_floor([1.0]) is None
    assert noise_floor([1.0, 1.0, 1.0, 3.0]) == 0
    adaptive = [Trial("A", "short", 1.0, 10.0, True, 8)]
    static = [Trial("B", "short", 1.5, 8.0, True, 8)]
    assert beats(adaptive, static, ttft_floor=0.1, decode_floor=0.5)["beats"]
    assert not beats(adaptive, static, ttft_floor=1.0, decode_floor=0.5)["beats"]
    mismatch = [Trial("A", "short", 1.0, 10.0, False, 8)]
    assert (
        beats(mismatch, static, ttft_floor=0.1, decode_floor=0.5)["reason"]
        == "greedy outputs differ"
    )


def test_token_diff_reports_the_first_disagreement():
    assert token_diff((1, 2), (1, 2))["exact"]
    diff = token_diff((1, 2, 3), (1, 9))
    assert diff["first_mismatch"] == 1
    assert token_diff((1, 2), (1, 2, 3))["first_mismatch"] == 2


def test_choose_device_uses_only_positive_measurements():
    choice = choose_device({"cpu": 0.02, "gpu": 0.01, "ane": None})
    assert choice["engine"] == "gpu"
    assert choice["measured"] == {"cpu": 0.02, "gpu": 0.01}
    assert choose_device({"cpu": None, "gpu": None})["engine"] is None
    with pytest.raises(ValueError):
        choose_device({"tpu": 1.0})


def test_abba_runner_rejects_a_mislabeled_trial():
    prompt = Prompt("short", "hi")

    def wrong(arm, item):
        return Trial("other", item.segment, 1, 1, True, 1)

    with pytest.raises(ValueError):
        run_abba(wrong, [prompt], 1)
