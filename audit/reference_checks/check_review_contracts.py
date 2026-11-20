"""Independent hand examples for the November 20 review; not a pseudocode interpreter."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), (actual, expected)


def run_checks():
    checks = []
    # Stable path IDs are neither array offsets nor a nested OD mapping.
    od = ('A', 'C')
    feasible = ['via-B', 'direct']
    probabilities = {(od, 'direct'): 0.25, (od, 'via-B'): 0.75}
    rho = [probabilities[(od, path_id)] for path_id in feasible]
    flows = [8 * value for value in rho]
    assert flows == [6, 2]
    close(sum(flows), 8)
    checks.append({'name':'composite_path_key_and_order', 'status':'passed',
                   'feasible_order':feasible, 'gathered_probabilities':rho,
                   'actual_path_flow':flows, 'expected_path_flow':[6,2]})

    # A known empty feasible set records unassigned demand rather than inventing a route.
    demand, candidates = 7.0, []
    assigned = 0.0
    unassigned = demand if not candidates else 0.0
    close(assigned + unassigned, demand)
    assert unassigned > 0
    checks.append({'name':'empty_path_set_keeps_unassigned_mass', 'status':'passed',
                   'demand':demand, 'assigned':assigned, 'unassigned':unassigned,
                   'coverage':'incomplete; not verified zero section flow'})

    # Shared physical infrastructure must retain line identity for train-load conversion.
    interval_hours = 0.25
    volumes = {'line_1':120.0, 'line_2':60.0}
    frequencies = {'line_1':12.0, 'line_2':6.0}
    loads = {line:volumes[line]/(frequencies[line]*interval_hours) for line in volumes}
    assert loads == {'line_1':40.0, 'line_2':40.0}
    incorrectly_pooled_line_1 = sum(volumes.values())/(frequencies['line_1']*interval_hours)
    assert incorrectly_pooled_line_1 == 60.0
    checks.append({'name':'line_specific_uniform_arrival_conversion', 'status':'passed',
                   'expected_loads':{'line_1':40.0,'line_2':40.0}, 'actual_loads':loads,
                   'incorrect_pooled_line_1_load':incorrectly_pooled_line_1,
                   'scope':'declared uniform-arrival reconstruction example; paper conversion unspecified'})

    # No division is reached when service is zero, for either zero or positive demand.
    rows = []
    for volume, service in [(0.0,0.0),(5.0,0.0),(5.0,2.0)]:
        if service == 0:
            status = 'no_train_in_interval' if volume == 0 else 'incompatible_support'
            load = None
        else:
            status, load = 'defined', volume/service
        rows.append({'volume':volume,'service':service,'status':status,'load':load})
    assert [r['load'] for r in rows] == [None,None,2.5]
    checks.append({'name':'zero_service_guard_before_division', 'status':'passed','cases':rows})
    return {'artifact_kind':'independent_contract_hand_examples', 'audit_date':'2026-11-20',
            'checks_passed':len(checks), 'executed_pseudocode':False,
            'executed_original_model':False, 'trained_model':False,
            'reproduced_paper_results':False, 'dependencies':['Python standard library'],
            'checks':checks}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args=parser.parse_args()
    rendered=json.dumps(run_checks(), ensure_ascii=False, indent=2)
    if args.output: args.output.write_text(rendered+'\n', encoding='utf-8')
    print(rendered)
