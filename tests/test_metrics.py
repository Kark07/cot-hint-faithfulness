"""Tests for Wilson CI and the rejection rule."""

import math

from src.metrics import passes_rejection_rule, wilson_ci


def test_wilson_known_values():
    # k=0, n=10: Wilson 95% upper bound ~ 0.278; lower = 0.
    p = wilson_ci(0, 10)
    assert p.p == 0.0
    assert p.lo == 0.0
    assert 0.25 < p.hi < 0.32

    # k=10, n=10: upper ~= 1 (clamped), lower ~ 0.722.
    p = wilson_ci(10, 10)
    assert p.p == 1.0
    assert p.hi >= 0.999
    assert 0.68 < p.lo < 0.76


def test_wilson_zero_n():
    p = wilson_ci(0, 0)
    assert math.isnan(p.p) and math.isnan(p.lo) and math.isnan(p.hi)


def test_rejection_rule_fires_only_on_lower_ci():
    # High point estimate but n small -> lower CI below 0.80 -> do not reject.
    p = wilson_ci(8, 10)   # p=0.8 but lower ~ 0.49
    assert passes_rejection_rule(p, threshold=0.80) is False

    # Large n with p=0.9 -> lower CI > 0.80 -> reject H1.
    p = wilson_ci(90, 100)
    assert passes_rejection_rule(p, threshold=0.80) is True


def test_decide_three_zones():
    from src.metrics import decide
    assert decide(wilson_ci(29, 29)) == "INSUFFICIENT_EVIDENCE"   # n < 30
    assert decide(wilson_ci(30, 30)) == "H1_REJECTED"
    assert decide(wilson_ci(28, 30)) == "INCONCLUSIVE"            # lo 0.787 < 0.80
    assert decide(wilson_ci(29, 30)) == "H1_REJECTED"             # lo 0.833
    assert decide(wilson_ci(8, 30)) == "H1_SUPPORTED"             # hi 0.444 < 0.50
    assert decide(wilson_ci(18, 30)) == "INCONCLUSIVE"            # 0.6, mid zone
    assert decide(wilson_ci(0, 0)) == "INSUFFICIENT_EVIDENCE"


def test_rejection_rule_defined_zero_sample():
    p = wilson_ci(0, 0)
    assert passes_rejection_rule(p) is False
