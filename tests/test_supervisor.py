import runpy

import pytest

SUPERVISOR = runpy.run_path("tools/bounded_smoke.py")
MIB = 1024**2


def test_swap_limit_can_be_raised_without_disabling_it():
    default = SUPERVISOR["swap_growth_limit"]({})
    custom = SUPERVISOR["swap_growth_limit"]({"safety": {"max_swap_growth_bytes": 1024 * MIB}})
    assert default == 512 * MIB
    assert SUPERVISOR["swap_stop_reason"](1000 * MIB, 1700 * MIB, default) is not None
    assert SUPERVISOR["swap_stop_reason"](1000 * MIB, 1700 * MIB, custom) is None
    assert SUPERVISOR["swap_stop_reason"](1000 * MIB, 2100 * MIB, custom) is not None


def test_swap_guard_measures_growth_not_preexisting_usage():
    check = SUPERVISOR["swap_stop_reason"]
    assert check(4096 * MIB, 4196 * MIB, 512 * MIB) is None
    assert check(4096 * MIB, 4608 * MIB, 512 * MIB) is None
    assert check(4096 * MIB, 4608 * MIB + 1, 512 * MIB) is not None


def test_explicit_null_disables_swap_guard_but_omitting_field_keeps_default():
    assert SUPERVISOR["swap_growth_limit"]({"safety": {"max_swap_growth_bytes": None}}) is None
    assert SUPERVISOR["swap_growth_limit"]({}) == 512 * MIB
    assert SUPERVISOR["swap_stop_reason"](0, 100000 * MIB, None) is None


@pytest.mark.parametrize("value", [0, -1, True, "1073741824"])
def test_invalid_swap_policies_fail_before_launch(value):
    with pytest.raises(ValueError, match="positive integer"):
        SUPERVISOR["swap_growth_limit"]({"safety": {"max_swap_growth_bytes": value}})
