# Data and configuration

Use the [pseudocode workflow](PSEUDOCODE_GUIDE.md) with the inputs below to implement and evaluate KNOWCAST.

## Inputs

| Input | Use |
| --- | --- |
| AFC records, station topology and timetables | Cutoff-aware demand, routes and service |
| Population and census data, calendar and events | Generation covariates; temporal-only variants use calendar and events |
| Training-fitted destination map and residual shares | Compressed targets and station totals |
| Auxiliary checkpoints, scalers and pretrained encoders | Generation guidance and knowledge embeddings |
| Configuration, seeds, split identities and environment | Training, selection and comparison |
| Operational annotations and calibration data | Carriage expansion, compensation and load alignment |

Seven reference CSV tables accompany the algorithms. Supply the AFC data, operational records and fitted artifacts for the chosen experiment. [Implementation parameters](specs/implementation_parameters.md) lists configuration fields; [paper settings](specs/paper_settings.yaml) records values reported in the article.

## Evaluation workflow

1. Set the observation cutoff, split dates and sample identities.
2. Check flow conservation in unfinished allocation, destination compression and residual redistribution.
3. Check IPF marginals, feasible-path probabilities and retained-OD assignment on small examples.
4. Confirm gradients reach the active utility parameters and knowledge encoders.
5. Align batch transforms, target masks and per-horizon metrics.
6. Select configurations and checkpoints by validation MAE, then evaluate the held-out split.
7. Match operational estimates by section, line, direction, train and observation time.

The [validation scripts](validation/scripts/README.md) provide small numerical examples and module checks. For model experiments, save the executable implementation, configuration, selected checkpoint, prediction records and measured results together.
