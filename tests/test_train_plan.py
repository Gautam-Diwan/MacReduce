import pytest

from macreduce.train_plan import (
    PEER,
    Seat,
    TrainPlan,
    comparison_plans,
    probe_backends,
    report,
    validate,
)


def test_comparison_covers_explicit_ane_overlap_and_topologies():
    plans = comparison_plans()
    questions = {plan.question for plan in plans}
    assert questions == {"baseline", "explicit_ane_trainer", "implicit_coreml", "overlap", "topology"}
    topologies = {plan.topology for plan in plans}
    assert {
        "ring_allreduce",
        "heterogeneous_allreduce",
        "hierarchical_allreduce",
        "tree_allreduce",
        "alongside",
    } <= topologies
    named = {plan.name: plan for plan in plans}
    assert not named["heterogeneous_one_mac"].needs_second_mac()
    assert named["heterogeneous_two_mac"].needs_second_mac()
    assert all(not plan.needs_second_mac() or PEER in plan.hosts() for plan in plans)
    body = report()
    assert body["curve"]["epochs"] == 3
    assert body["curve"]["train_examples"] == 4096
    assert body["status"] == "planned"
    assert body["measured"] is False
    assert body["baseline"] == "gpu_one_mac"
    assert all(plan["measured"] is False for plan in body["plans"])


def test_ring_needs_two_trainers_and_explicit_plans_name_a_private_stack():
    lone = TrainPlan(
        "lone",
        "topology",
        "ring_allreduce",
        "train_only",
        (Seat("a", "local", "train", "mlx_gpu"),),
    )
    with pytest.raises(ValueError, match="two trainers"):
        validate(lone)
    mixed_ring = TrainPlan(
        "mixed",
        "topology",
        "ring_allreduce",
        "train_only",
        (
            Seat("gpu", "local", "train", "mlx_gpu"),
            Seat("ane", "peer", "train", "aneforge"),
        ),
    )
    with pytest.raises(ValueError, match="equal trainers"):
        validate(mixed_ring)
    named = {plan.name: plan for plan in comparison_plans()}
    assert named["aneforge_one_mac"].seats[0].train_backend == "aneforge"
    assert named["ring_two_mac"].needs_second_mac()
    assert not named["gpu_one_mac"].needs_second_mac()


def test_probe_does_not_treat_a_missing_stack_as_the_gpu():
    rows = {row["backend"]: row for row in probe_backends()}
    assert rows["orion"]["status"] == "unpinned"
    assert rows["maderix"]["status"] == "unpinned"
    assert rows["aneforge"]["placement_verified"] is False
    assert rows["mlx_gpu"]["role"] == "baseline"
    if rows["aneforge"]["status"] == "missing":
        assert "mlx" not in rows["aneforge"]["error"]
