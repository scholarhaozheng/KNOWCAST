"""Independent standard-library mathematical oracles for the model specification.

This script does not execute the .pseudo files, load original models, import Torch,
train KNOWCAST, or reproduce paper results. Expected values use small hand examples.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def assert_close(actual, expected, tolerance=1e-9):
    if isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), (actual, expected)
        for a, e in zip(actual, expected):
            assert_close(a, e, tolerance)
    else:
        assert math.isfinite(actual), actual
        assert abs(actual - expected) <= tolerance, (actual, expected, tolerance)


def gate_update(previous, candidate, update_gate):
    return [(1 - z) * old + z * new
            for old, new, z in zip(previous, candidate, update_gate)]


def masked_softmax(utilities, feasible):
    if len(utilities) != len(feasible):
        raise ValueError("Utility and mask lengths differ.")
    valid = [i for i, keep in enumerate(feasible) if keep]
    if not valid:
        raise ValueError("Empty feasible set: this oracle explicitly rejects it.")
    maximum = max(utilities[i] for i in valid)
    weights = {i: math.exp(utilities[i] - maximum) for i in valid}
    total = sum(weights.values())
    return [weights.get(i, 0.0) / total for i in range(len(utilities))]


def stable_teacher_probability(step, k):
    if step < 0 or k <= 0 or not math.isfinite(step) or not math.isfinite(k):
        raise ValueError("step must be finite/nonnegative and k finite/positive.")
    a = step / k - math.log(k)
    if a >= 0:
        small = math.exp(-a)
        return small / (1 + small)
    return 1 / (1 + math.exp(a))


def ipf(production, attraction, cost, zeta, tolerance=1e-12, maximum_iterations=10000):
    if zeta <= 0 or any(value <= 0 for value in production + attraction):
        raise ValueError("This small oracle handles positive feasible margins only.")
    assert_close(sum(production), sum(attraction))
    kernel = [[math.exp(-zeta * value) for value in row] for row in cost]
    origin = [1.0] * len(production)
    destination = [1.0] * len(attraction)
    for iteration in range(1, maximum_iterations + 1):
        origin = [1 / sum(destination[n] * attraction[n] * kernel[m][n]
                          for n in range(len(attraction)))
                  for m in range(len(production))]
        destination = [1 / sum(origin[m] * production[m] * kernel[m][n]
                               for m in range(len(production)))
                       for n in range(len(attraction))]
        flow = [[origin[m] * production[m] * destination[n] * attraction[n] * kernel[m][n]
                 for n in range(len(attraction))]
                for m in range(len(production))]
        row_totals = [sum(row) for row in flow]
        column_totals = [sum(row[n] for row in flow) for n in range(len(attraction))]
        error = max([abs(a - b) for a, b in zip(row_totals, production)]
                    + [abs(a - b) for a, b in zip(column_totals, attraction)])
        if error <= tolerance:
            return {"flow": flow, "kernel": kernel, "origin": origin,
                    "destination": destination, "iterations": iteration,
                    "maximum_margin_error": error}
    raise ArithmeticError("IPF did not converge within this oracle's explicit limit.")


def check_gru_equation():
    previous = [2.0, -3.0]
    candidate = [-4.0, 5.0]
    cases = [([0.0, 0.0], [2.0, -3.0]),
             ([1.0, 1.0], [-4.0, 5.0]),
             ([0.25, 0.75], [0.5, 3.0])]
    results = []
    for gate, expected in cases:
        actual = gate_update(previous, candidate, gate)
        assert_close(actual, expected)
        results.append({"z": gate, "expected": expected, "actual": actual})
    hidden = [2, 3]
    reset = [0, 1]
    matrix = [[1, 2], [3, 4]]
    reset_before = [sum(w * r * h for w, r, h in zip(row, reset, hidden))
                    for row in matrix]
    transformed = [sum(w * h for w, h in zip(row, hidden)) for row in matrix]
    reset_after = [r * value for r, value in zip(reset, transformed)]
    assert_close(reset_before, [6, 12])
    assert_close(reset_after, [0, 18])
    assert reset_before != reset_after
    return {"name": "Eq32_update_gate_and_reset_placement", "status": "passed",
            "cases": results, "reset_before_transform_expected": [6, 12],
            "reset_before_transform_actual": reset_before,
            "different_reset_after_transform": reset_after}


def check_softmax():
    logits = [1000.0, 999.0, 2000.0]
    mask = [True, True, False]
    expected = [0.7310585786300049, 0.2689414213699951, 0.0]
    actual = masked_softmax(logits, mask)
    assert_close(actual, expected)
    assert_close(sum(actual), 1.0)
    assert_close(masked_softmax([-1000, -1001], [True, True]), expected[:2])
    assert_close(masked_softmax([4, -2], [False, True]), [0.0, 1.0])
    empty_rejected = False
    try:
        masked_softmax([1, 2], [False, False])
    except ValueError:
        empty_rejected = True
    assert empty_rejected
    return {"name": "masked_stable_path_softmax", "status": "passed",
            "utilities": logits, "feasible_mask": mask,
            "expected": expected, "actual": actual,
            "empty_set_policy": "explicit oracle rejection, not a recovered paper policy",
            "empty_set_rejected": empty_rejected}


def check_assignment():
    # One retained real destination per origin and one residual column.
    compressed = [[10.0, 90.0], [6.0, 4.0], [8.0, 2.0]]
    retained_destinations = [2, 0, 1]
    paths = {
        (0, 2): [(0.25, [(0, 2)]), (0.75, [(0, 1), (1, 2)])],
        (1, 0): [(1.0, [(1, 0)])],
        (2, 1): [(1.0, [(2, 1)])],
    }
    per_od = {}
    section = {}
    for origin, destination in enumerate(retained_destinations):
        demand = compressed[origin][0]
        path_flows = []
        for probability, directed_sections in paths[(origin, destination)]:
            flow = demand * probability
            path_flows.append(flow)
            for edge in set(directed_sections):  # Binary incidence, not traversal multiplicity.
                section[edge] = section.get(edge, 0.0) + flow
        assert_close(sum(path_flows), demand)
        per_od[f"{origin}->{destination}"] = path_flows
    expected_paths = {"0->2": [2.5, 7.5], "1->0": [6.0], "2->1": [8.0]}
    expected_sections = {"0->2": 2.5, "0->1": 7.5, "1->2": 7.5,
                         "1->0": 6.0, "2->1": 8.0}
    actual_sections = {f"{a}->{b}": value for (a, b), value in section.items()}
    for key in expected_paths:
        assert_close(per_od[key], expected_paths[key])
    for key in expected_sections:
        assert_close(actual_sections[key], expected_sections[key])
    assert set(actual_sections) == set(expected_sections)
    total_assigned = sum(sum(values) for values in per_od.values())
    assert_close(total_assigned, 24.0)
    assert_close(sum(row[1] for row in compressed), 96.0)
    assert (2, 0) not in section  # A reverse edge is not manufactured.
    return {"name": "retained_OD_path_and_directed_section_conservation", "status": "passed",
            "expected_path_flows": expected_paths, "actual_path_flows": per_od,
            "expected_section_flows": expected_sections, "actual_section_flows": actual_sections,
            "expected_assigned_demand": 24.0, "actual_assigned_demand": total_assigned,
            "excluded_residual_demand": 96.0,
            "section_total": sum(section.values()),
            "section_total_note": "31.5 exceeds 24 because one path uses two sections"}


def check_gravity():
    cost = [[0.0, math.log(2)], [math.log(2), 0.0]]
    first = ipf([3.0, 3.0], [3.0, 3.0], cost, 1.0)
    second = ipf([3.0, 6.0], [4.0, 5.0], cost, 1.0)
    expected_kernel = [[1.0, 0.5], [0.5, 1.0]]
    expected_first = [[2.0, 1.0], [1.0, 2.0]]
    expected_second = [[2.0, 1.0], [2.0, 4.0]]
    assert_close(first["kernel"], expected_kernel)
    assert_close(first["flow"], expected_first, 1e-10)
    assert_close(second["flow"], expected_second, 1e-10)
    old_coefficients_on_new_margins = [
        [first["origin"][m] * [3.0, 6.0][m] * first["destination"][n]
         * [4.0, 5.0][n] * expected_kernel[m][n] for n in range(2)]
        for m in range(2)]
    stale_row_totals = [sum(row) for row in old_coefficients_on_new_margins]
    assert max(abs(x - y) for x, y in zip(stale_row_totals, [3, 6])) > 1.0
    return {"name": "exponential_deterrence_and_time_specific_IPF", "status": "passed",
            "cost": cost, "zeta": 1.0, "expected_kernel": expected_kernel,
            "actual_kernel": first["kernel"],
            "case_1": {"production": [3, 3], "attraction": [3, 3],
                       "expected_flow": expected_first, **first},
            "case_2": {"production": [3, 6], "attraction": [4, 5],
                       "expected_flow": expected_second, **second},
            "stale_first_case_coefficients_on_case_2": old_coefficients_on_new_margins,
            "stale_row_totals": stale_row_totals,
            "proof_scope": "positive feasible 2x2 examples, not all degenerate/IPF support cases"}


def check_schedule():
    cases = []
    for step in [0.0, 10.0, 100.0]:
        actual = stable_teacher_probability(step, 10.0)
        expected = 10.0 / (10.0 + math.exp(step / 10.0))
        assert_close(actual, expected)
        cases.append({"step": step, "k": 10.0, "expected": expected, "actual": actual})
    large = stable_teacher_probability(1_000_000.0, 10.0)
    assert_close(large, 0.0)
    assert cases[0]["actual"] > cases[1]["actual"] > cases[2]["actual"] > large
    return {"name": "legacy_inverse_sigmoid_schedule_stability", "status": "passed",
            "profile": "legacy alternative; paper training uses teacher forcing",
            "finite_cases": cases, "large_step": 1_000_000,
            "large_step_expected": 0.0, "large_step_actual": large}


def check_utility_gradient():
    attributes = [[1.0, 0.0], [0.0, 1.0]]
    downstream_weights = [1.0, 3.0]

    def downstream_objective(beta):
        utilities = [sum(a * b for a, b in zip(row, beta)) for row in attributes]
        probabilities = masked_softmax(utilities, [True, True])
        return sum(p * w for p, w in zip(probabilities, downstream_weights))

    beta = [0.0, 0.0]
    step = 1e-5
    measured = []
    for axis in range(2):
        plus, minus = list(beta), list(beta)
        plus[axis] += step
        minus[axis] -= step
        measured.append((downstream_objective(plus) - downstream_objective(minus)) / (2 * step))
    expected_gradient = [-0.5, 0.5]
    assert_close(downstream_objective(beta), 2.0)
    assert_close(measured, expected_gradient, 1e-9)
    prior = [-0.30, -0.573]
    prior_attributes = [[4, 0], [3, 1], [6, 0]]
    prior_utilities = [sum(a * b for a, b in zip(row, prior)) for row in prior_attributes]
    assert_close(prior_utilities, [-1.2, -1.473, -1.8])
    return {"name": "utility_to_softmax_finite_difference", "status": "passed",
            "objective": "weighted probability surrogate, not a trained forecasting network",
            "beta_at_derivative": beta, "attributes": attributes,
            "downstream_weights": downstream_weights, "expected_objective": 2.0,
            "expected_gradient": expected_gradient, "finite_difference_gradient": measured,
            "finite_difference_step": step, "reported_prior": prior,
            "prior_example_expected_utilities": [-1.2, -1.473, -1.8],
            "prior_example_actual_utilities": prior_utilities}


def static_registration_evidence(workspace):
    import re
    path = workspace / "models/Net_0207.pseudo"
    text = path.read_text(encoding="utf-8")
    contracts = {
        "SimpleAutoencoder.__init__": [r"REGISTER", r"INITIALIZE"],
        "ConfiguredMLPEncoder.__init__": [r"REGISTER.*weight\[layer\].*bias\[layer\]"],
        "Net_0207.__init__": [r"registry\s*<-", r"generic_encoder\s*<-\s*REGISTER",
                              r"encoder parameters trainable", r"REGISTER distribution_encoder"],
        "Net_0207.prepare_autoencoders": [r"DIFFERENTIATE.*encoder and decoder", r"optimizer"],
        "Net_0207.checkpoint_contract": [r"registry\.state_dict\(\)", r"RESTORE registry"],
    }
    evidence = []
    for name, patterns in contracts.items():
        match = re.search(r"(?ms)^PROCEDURE " + re.escape(name) + r"\(.*?(?=^PROCEDURE |^CLASS |\Z)", text)
        assert match, name
        for pattern in patterns:
            assert re.search(pattern, match.group(), re.IGNORECASE), (name, pattern)
        evidence.append({"procedure": name, "line": text[:match.start()].count("\n") + 1,
                         "contracts": patterns})
    training = (workspace / "train_save_history.pseudo").read_text(encoding="utf-8")
    assert re.search(r"BUILD_OPTIMIZER\(model,", training)
    return {"name": "autoencoder_parameter_registration_contract", "status": "static_contract_present",
            "file": str(path.relative_to(workspace)).replace("\\", "/"), "evidence": evidence,
            "scope": "Declaration-scoped registration, pretraining, checkpoint and optimizer contracts"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[2]
    checks = [check_gru_equation(), check_softmax(), check_assignment(), check_gravity(),
              check_schedule(), check_utility_gradient(), static_registration_evidence(workspace)]
    result = {"artifact_kind": "independent_formula_reference_checks",
              "executed_pseudocode": False,
              "executed_original_model": False,
              "trained_model": False,
              "reproduced_paper_results": False,
              "dependencies": ["Python standard library"],
              "numerical_oracles_passed": sum(check["status"] == "passed" for check in checks),
              "static_contract_checks": 1,
              "checks": checks}
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
