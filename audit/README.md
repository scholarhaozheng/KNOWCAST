# Coverage and reference checks

| Record | Contents |
| --- | --- |
| [File coverage](SOURCE_COVERAGE.md) | 42 original module counterparts and 200 original declarations |
| [Source inventory](source_inventory.json) | Original signatures, locations and hashes |
| [Coverage JSON](source_coverage.json) / [CSV](source_coverage.csv) | File correspondence and pseudocode hashes |
| [Declaration index](declarations.json) | Current algorithm declarations |
| [Structure results](structure-check-results.json) | Actual declaration coverage and file-hash checks |
| [Reference scripts](reference_checks/README.md) | Reusable structural and numerical checks |
| [Mathematical check summary](mathematical_validation.json) | 24 numerical examples and one static parameter contract |
| [Model examples](model-math-results.json) | Recurrent gates, softmax, assignment, IPF, scheduling and derivatives |
| [Data and training examples](data-training-math-results.json) | Allocation, losses, metrics, sample identity, cutoffs and service units |
| [Assignment examples](review-contract-results.json) | Path keys, unassigned demand, line-specific conversion and zero service |
| [Temporal kernel examples](temporal-kernel-results.json) | Interval integration, negative tails and mass conservation |

The source inventory identifies the module structure used for the pseudocode companion. Structural coverage and independent formula examples have separate scopes; neither runs the forecasting model. See [edition status](../REPOSITORY_STATUS.md) and the [reading guide](../PSEUDOCODE_GUIDE.md).
