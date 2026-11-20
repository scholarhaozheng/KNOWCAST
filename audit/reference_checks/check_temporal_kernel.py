"""Independent Gaussian interval-weight examples for the operational specification."""
import argparse
import json
import math
from pathlib import Path


def cdf(x, mean, spread):
    return 0.5 * (1 + math.erf((x - mean) / (spread * math.sqrt(2))))


def interval_mass(low, high, mean, spread, policy):
    assert high >= low and mean >= 0 and spread >= 0
    if spread == 0:
        return float(low <= mean < high)
    if policy == "full_gaussian":
        return cdf(high, mean, spread) - cdf(low, mean, spread)
    assert policy in {"causal_discard", "causal_renormalized"}
    if high <= 0:
        return 0.0
    value = cdf(high, mean, spread) - cdf(max(low, 0), mean, spread)
    if policy == "causal_renormalized":
        value /= 1 - cdf(0, mean, spread)
    return value


def weights(nodes, intervals, mean, spread, policy):
    result = [sum(w * interval_mass(a - t, b - t, mean, spread, policy)
                  for t, w in nodes) for a, b in intervals]
    removed = cdf(0, mean, spread) if policy == "causal_discard" and spread else 0
    outside = 1 - sum(result) - removed
    assert outside >= -1e-12
    return result, removed, max(outside, 0)


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-11), (actual, expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    checks = []

    bins = [(0, 1), (1, 2), (2, 3)]
    for policy in ("full_gaussian", "causal_discard", "causal_renormalized"):
        actual, removed, outside = weights([(0, 1)], bins, 1, 0, policy)
        assert actual == [0, 1, 0] and removed == outside == 0
    checks.append({"name": "deterministic_half_open_bin", "status": "passed"})

    close(interval_mass(-1, 1, 0, 1, "full_gaussian"), 0.6826894921370859)
    close(interval_mass(-math.inf, math.inf, 2, 3, "full_gaussian"), 1)
    checks.append({"name": "gaussian_cdf_interval_mass", "status": "passed"})

    actual, removed, outside = weights([(0, 1)], [(0, math.inf)], 0, 1, "causal_discard")
    close(actual[0], 0.5)
    close(removed, 0.5)
    close(outside, 0)
    checks.append({"name": "discarded_negative_tail_is_accounted", "status": "passed"})

    actual, removed, outside = weights([(0, 1)], [(0, 1)], 0, 1, "causal_renormalized")
    close(actual[0], 0.6826894921370859)
    close(outside, 1 - actual[0])
    close(removed, 0)
    checks.append({"name": "conditional_kernel_finite_grid_tail", "status": "passed"})

    actual, removed, outside = weights([(0, 0.25), (1, 0.75)], bins, 1, 0, "causal_discard")
    assert actual == [0, 0.25, 0.75]
    for policy in ("full_gaussian", "causal_discard", "causal_renormalized"):
        values, removed, outside = weights([(0, 0.25), (1, 0.75)], bins, 0.5, 0.8, policy)
        close(17 * sum(values) + 17 * removed + 17 * outside, 17)
        assert all(x >= 0 for x in values)
    checks.append({"name": "entry_quadrature_and_mass_ledger", "status": "passed"})

    report = {"artifact_kind": "independent_formula_reference_checks", "executed_pseudocode": False,
              "numerical_oracles_passed": len(checks), "checks": checks}
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
