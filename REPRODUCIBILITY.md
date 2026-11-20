# Implementation inputs and checks

The [pseudocode guide](PSEUDOCODE_GUIDE.md) describes the full workflow. This page collects its required data and verification steps.

## Inputs

| Input | Use |
| --- | --- |
| Raw AFC records, topology and timetables | Cutoff-aware demand, routes and service |
| Demographics, temporal context and event records | Generation covariates |
| Training-fitted destination map and residual shares | Stable compressed targets and station totals |
| Auxiliary checkpoints, scalers and pretrained encoders | Generation guidance and knowledge embeddings |
| Configuration, seeds, split identities and environment | Training, selection and comparison |
| Operational annotations and calibration data | Carriage expansion, compensation and load alignment |

The repository includes algorithms, schemas, reported settings and seven reference CSV tables. Full data and fitted artifacts must be supplied. Configuration choices are grouped in [implementation parameters](specs/implementation_parameters.md).

## Verification sequence

1. Check observation cutoffs and split identities before preparing samples.
2. Verify conservation for unfinished allocation, destination compression and residual redistribution.
3. Check IPF marginals, feasible-path probabilities and retained-OD assignment on small examples.
4. Confirm that the forecasting loss updates active utility parameters and knowledge encoders.
5. Verify batch scaling, target masks and per-horizon metrics.
6. Select configurations and checkpoints by validation MAE, then export held-out predictions.
7. Match operational estimates by section, line, direction, train and observation time.

The repository's [reference scripts](audit/reference_checks/README.md) check structure and independent formula examples. Model training and numerical reproduction require an executable implementation and the inputs above. Keep the configuration, selected checkpoint, predictions and measured results together.
