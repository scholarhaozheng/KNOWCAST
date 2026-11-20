"""Numerical examples for flow allocation, training losses, metrics, and data timing."""
from __future__ import annotations
import argparse
import itertools
import json
import math
from datetime import datetime
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path)
args = parser.parse_args()
checks = []

def close(actual, expected, tol=1e-7):
    if isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), (actual, expected)
        for a, e in zip(actual, expected): close(a,e,tol)
    else:
        assert math.isclose(actual,expected,rel_tol=tol,abs_tol=tol),(actual,expected)

def check(name, **details):
    checks.append(dict(name=name,status='passed',**details))

def allocate(u, reference):
    assert u>=0 and all(x>=0 for x in reference)
    total=sum(reference)
    share=[x/total for x in reference] if total else [1/len(reference)]*len(reference)
    return [u*x for x in share]

normal=allocate(12,[1,2,3]); zero=allocate(12,[0,0,0]); none_unfinished=allocate(0,[1,2,3])
close(normal,[2,4,6]); close(zero,[4,4,4]); close(none_unfinished,[0,0,0])
assert all(math.isclose(sum(a),u) for a,u in [(normal,12),(zero,12),(none_unfinished,0)])
check('unfinished_allocation_and_observed_zero', expected_nonzero=[2,4,6],actual_nonzero=normal,
      expected_zero=[4,4,4],actual_zero=zero,missing_history_policy='separate missing-data state')

# Each row's retained column names a different destination; residual weights are fixed.
rds=[1,2,0]
weights=[[.25,0,.75],[.5,.5,0],[0,.5,.5]]
pred=[[10.,6.],[20.,8.],[30.,4.]]
def totals(x):
    p=[sum(row) for row in x]; a=[0.,0.,0.]
    for o,row in enumerate(x):
        a[rds[o]] += row[0]
        for d in range(3): a[d] += row[1]*weights[o][d]
    return p,a
production,attraction=totals(pred)
close(production,[16,28,34]); close(attraction,[35.5,16,26.5]); close(sum(production),sum(attraction))
check('Eq15_residual_destination_totals',expected_production=[16,28,34],actual_production=production,
      expected_attraction=[35.5,16,26.5],actual_attraction=attraction,total_flow=78,
      origin_specific_destinations=rds,zero_history_uniform_row=weights[2])

target_p=[17,27,35]; target_a=[34.5,18,25.5]
def generation_loss(x):
    p,a=totals(x)
    return sum((u-v)**2 for u,v in zip(p,target_p))+sum((u-v)**2 for u,v in zip(a,target_a))
close(generation_loss(pred),9)
fd=[]; step=1e-5
for o in range(3):
    row=[]
    for k in range(2):
        xp=[r[:] for r in pred]; xm=[r[:] for r in pred]
        xp[o][k]+=step; xm[o][k]-=step
        row.append((generation_loss(xp)-generation_loss(xm))/(2*step))
    fd.append(row)
expected_grad=[[-6,0],[4,1],[0,-3]]
close(fd,expected_grad)
assert sum(x*x for x in [1,-1]) == 2 and abs(sum([1,-1])/2)==0
check('Eq16_squared_loss_and_OD_finite_difference',expected_loss=9,actual_loss=generation_loss(pred),
      expected_gradient=expected_grad,finite_difference_gradient=fd,
      gradient_method='central finite difference',squared_cancellation_counterexample=2)

# M=4 complete target matrices, each with N*K=6 cells; choose every m=2 subset.
E=[6,12,18,24]; G=[1,4,9,16]; M=4; cells=6; m=2
full_od=sum(E)/(M*cells); full_g=sum(G)
est=[(sum(E[j] for j in s)/(m*cells),(M/m)*sum(G[j] for j in s))
     for s in itertools.combinations(range(M),m)]
mean_od=sum(x[0] for x in est)/len(est); mean_g=sum(x[1] for x in est)/len(est)
close([mean_od,mean_g],[full_od,full_g])
micro=[range(0,2),range(2,4)]
acc_od=sum(sum(E[j] for j in s)/(M*cells) for s in micro)
acc_g=sum(sum(G[j] for j in s) for s in micro)
close([acc_od,acc_g],[full_od,full_g])
check('declared_minibatch_estimators',full_objectives=[full_od,full_g],
      expected_uniform_subset_estimators=[mean_od,mean_g],full_partition_accumulation=[acc_od,acc_g],
      subset_count=len(est),policy='uniform sampling of complete target matrices; full-partition accumulation')

decay=[.0264*max(1-.1*e,0) for e in [0,9,10,11]]
close(decay,[.0264,.00264,0,0])
check('Eq17_formula_indices',epsilon=[0,9,10,11],weights=decay,
      epoch_index_mapping='configuration parameter')

def metrics(pred,truth):
    err=[abs(p-y) for p,y in zip(pred,truth)]
    return [sum(err)/len(err),math.sqrt(sum(x*x for x in err)/len(err)),100*sum(err)/sum(truth)]
h1=metrics([1,2],[1,1]); h2=metrics([4,7],[1,3])
means=[(a+b)/2 for a,b in zip(h1,h2)]
expected_means=[2.,(math.sqrt(.5)+math.sqrt(12.5))/2,112.5]
close(means,expected_means)
pooled=metrics([1,2,4,7],[1,1,1,3])
assert not math.isclose(means[1],pooled[1]) and not math.isclose(means[2],pooled[2])
check('horizon_first_metrics',per_horizon=[h1,h2],mean_horizons=means,
      pooled_counterexample=pooled,wmape_unit='percent')

truth_by_id={'a':1.,'b':3.,'c':2.}
shuffled=[('c',4.),('a',2.),('b',4.)]
aligned=[(pid,p,truth_by_id[pid]) for pid,p in shuffled]
close(metrics([x[1] for x in aligned],[x[2] for x in aligned]),[4/3,math.sqrt(2),100*4/6])
assert len(set(x[0] for x in aligned))==len(truth_by_id)
batches=[['a','b'],['c']]
assert sum(map(len,batches))==3
expected_set=set(truth_by_id)
assert set(x[0] for x in shuffled[:-1])!=expected_set
check('sample_ID_join_and_final_partial_batch',shuffled_ids=[x[0] for x in shuffled],
      paired_truth=[x[2] for x in aligned],batch_sizes=list(map(len,batches)),
      missing_sample_detected=True,join_key='sample_id')

cutoff=datetime.fromisoformat('2023-05-01T10:20:00')
records=[{'origin':0,'dest':1,'entry':'2023-05-01T10:02:00','exit':'2023-05-01T10:10:00'},
         {'origin':0,'dest':2,'entry':'2023-05-01T10:05:00','exit':'2023-05-01T10:25:00'},
         {'origin':0,'dest':2,'entry':'2023-05-01T10:08:00','exit':'2023-05-01T10:30:00'}]
finished=[0,0,0]; unfinished=0
for row in records:
    assert datetime.fromisoformat(row['entry'])<cutoff
    if datetime.fromisoformat(row['exit'])<=cutoff: finished[row['dest']]+=1
    else: unfinished+=1
assert sum(finished)+unfinished==3 and finished==[0,1,0] and unfinished==2
check('cutoff_finished_unfinished_conservation',actual_cohort=3,finished=finished,unfinished=unfinished,
      delayed_exit_destinations='hidden until exit time',
      fixture_assumption='event available at its timestamp')

freq_hour=60/5; interval_count=freq_hour*15/60
close([freq_hour,interval_count],[12,3])
check('service_frequency_unit_conversion',headway_minutes=5,frequency_per_hour=freq_hour,
      interval_minutes=15,departures_per_interval=interval_count,
      storage_unit='trains_per_hour')

report={'artifact_kind':'data_training_formula_examples',
        'dependencies':['Python standard library'],
        'checks_passed':len(checks),'checks':checks}
rendered = json.dumps(report, indent=2, ensure_ascii=False)
if args.output is not None:
    args.output.write_text(rendered + "\n", "utf-8")
print(rendered)
