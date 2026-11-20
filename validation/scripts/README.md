# Formula examples

Run from the repository root with Python 3:

```text
python validation/scripts/check_model_math.py
python validation/scripts/check_data_training_math.py
python validation/scripts/check_assignment_math.py
python validation/scripts/check_temporal_kernel.py
python validation/scripts/check_structure.py
```

| Script | Examples |
| --- | --- |
| `check_model_math.py` | Six examples: recurrent gates, masked softmax, path assignment, IPF, scheduling and utility derivatives; one parameter-registration check |
| `check_data_training_math.py` | Nine examples: unfinished-trip allocation, residual scatter, losses, minibatch estimators, metrics, sample keys, forecast cutoffs and service units |
| `check_assignment_math.py` | Four examples: path keys, unassigned demand, line-specific load conversion and zero service |
| `check_temporal_kernel.py` | Five examples: deterministic bins, Gaussian interval integrals, causal tails, finite grids and mass conservation |
| `check_structure.py` | Module paths and class/procedure declarations against [modules.json](../modules.json) |

Each script prints JSON and accepts `--output PATH` to save the result. Assertions compare calculated and expected values; run with assertions enabled.
