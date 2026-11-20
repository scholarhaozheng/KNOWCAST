# Reference checks

These standard-library Python scripts check repository structure and independent numerical examples.

From the repository root:

```text
python audit/reference_checks/check_model_math.py
python audit/reference_checks/check_data_training_math.py
python audit/reference_checks/check_review_contracts.py
python audit/reference_checks/check_temporal_kernel.py
python audit/reference_checks/check_structure.py
```

| Script | Coverage |
| --- | --- |
| Model math | Six examples: GRU, softmax, assignment, IPF, scheduling and utility derivatives; one declaration-scoped parameter contract |
| Data/training math | Nine examples: allocation, residual scatter, losses, batch estimators, metrics, sample identity, cutoffs and service units |
| Review contracts | Four examples: composite path keys, unassigned demand, line-specific conversion and zero service |
| Temporal kernel | Five examples: deterministic bins, Gaussian integrals, negative tails, finite grids and quadrature mass conservation |
| Structure | Original declaration coverage, current file hashes and removed source markers |

Each script prints JSON and accepts `--output PATH`. Assertions identify mismatched expectations; run with Python assertions enabled. A structural hash mismatch means the coverage inventory needs refreshing after an edit.

The 24 numerical examples evaluate the formulas independently. They do not execute the pseudocode or train the forecasting model.
