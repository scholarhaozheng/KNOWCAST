# KNOWCAST pseudocode guide

The repository contains 42 file-by-file counterparts of the original Python modules, preserving all 200 original declarations. The algorithms follow the paper; the original source supplies the module and workflow inventory. See [edition status](REPOSITORY_STATUS.md) for provenance.

For the research overview, figures and reported results, return to the [README](README.md) or [project website](https://scholarhaozheng.github.io/KNOWCAST/).

## Reading route

1. [Prepare data and knowledge](generating_domain_knowledge_no_DO_clean.pseudo): topology, schedules, AFC, destination maps, historical references, generation models, gravity and encoder pretraining.
2. [Load and batch samples](lib/utils_CUROP.pseudo): transforms, unfinished allocations, memory mapping and encoder/decoder alignment.
3. [Build KNOWCAST](models/Net_0207.pseudo): registered knowledge modules, dynamic IPF, path utility, pretrained encoders and recurrent forecasting.
4. [Train](train_save_history.pseudo), [search configurations](bayes_opt.pseudo), and [evaluate checkpoints](evaluate_button.pseudo) through the [shared evaluation procedures](train_evaluate_functions.pseudo).
5. [Compare variants](experiments/ablation_and_sensitivity.pseudo) and [assess operational loads](experiments/operational_load_assessment.pseudo).

## Module coverage

| Scope | Original files | Responsibilities |
| --- | ---: | --- |
| Root entry points | 5 | Artifact generation, training, search, evaluation and dispatch |
| `lib/` | 3 | Loaders, scalers, batching, graphs, device transfer and metrics |
| `models/` | 7 | Graph/recurrent cells, fusion, encoder/decoder, utility, autoencoders and gravity |
| `dmn_knw_gnrtr/` | 11 | AFC, compression, windows, paths, incidence, demographics and auxiliary fitting |
| `metro_components/` | 10 | Stations, lines, sections, routes, managers and workbook loading |
| `metro_data_convertor/` | 6 | Serialization, timetables, frequency arrays and probability diagnostics |

The seven original CSV tables remain in `dmn_knw_gnrtr/`. [File coverage](audit/SOURCE_COVERAGE.md) and the [declaration index](audit/declarations.json) give the complete inventory.

Four added modules complete the workflow: [configuration](config.pseudo), [plotting](visualization_tools/plot_graph.pseudo), ablation/sensitivity, and operational assessment. Baseline adapters expose common training, prediction and selection interfaces.

## Notation and parameters

Each module begins with its purpose and paper references. Procedures state inputs and outputs, then show the calculation and return value. The [syntax guide](specs/pseudocode_conventions.md) defines assignments, shapes and status records; the [shared interfaces](specs/interface_contracts.md) define fields passed between modules.

Use the [parameter table](specs/implementation_parameters.md) for choices supplied by configuration, the [reported settings](specs/paper_settings.yaml) for numerical values from the paper, and the [equation crosswalk](specs/paper_to_pseudocode.md) to locate the corresponding procedures.

The five [conceptual chapters](pseudocode/README.md) offer a shorter reading route. The [coverage and check index](audit/README.md) and [reference checks](audit/reference_checks/README.md) document this edition's structural and mathematical checks.
