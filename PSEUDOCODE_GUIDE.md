# KNOWCAST pseudocode guide

The 46 `.pseudo` modules describe data preparation, knowledge integration, OD forecasting, training and evaluation. The [README](README.md) and [project website](https://scholarhaozheng.github.io/KNOWCAST/) present the paper's research overview and results.

## Reading route

1. [Prepare data and knowledge](generating_domain_knowledge_no_DO_clean.pseudo): topology, schedules, AFC, destination maps, historical references, generation models, gravity and encoder pretraining.
2. [Load and batch samples](lib/utils_CUROP.pseudo): transforms, unfinished allocations and encoder/decoder alignment.
3. [Build KNOWCAST](models/Net_0207.pseudo): knowledge encoders, dynamic IPF, path utility and recurrent forecasting.
4. [Train](train_save_history.pseudo), [search configurations](bayes_opt.pseudo) and [evaluate checkpoints](evaluate_button.pseudo) through the [shared evaluation procedures](train_evaluate_functions.pseudo).
5. [Compare variants](experiments/ablation_and_sensitivity.pseudo) and [assess operational loads](experiments/operational_load_assessment.pseudo).

## Modules

| Directory | Modules | Responsibilities |
| --- | ---: | --- |
| Root | 6 | Configuration, data preparation, training, search, evaluation and dispatch |
| `lib/` | 3 | Loading, scaling, batching, graphs, device transfer and metrics |
| `models/` | 7 | Graph/recurrent cells, fusion, encoder/decoder, utility, autoencoders and gravity |
| `dmn_knw_gnrtr/` | 11 | AFC, compression, windows, paths, incidence, demographics and auxiliary fitting |
| `metro_components/` | 10 | Stations, lines, sections, routes, managers and workbook loading |
| `metro_data_convertor/` | 6 | Serialization, timetables, frequency arrays and probability diagnostics |
| `experiments/` | 2 | Ablations, sensitivity, runtime comparison and operational loads |
| `visualization_tools/` | 1 | Training curves, metrics and OD prediction plots |

The [module index](validation/MODULES.md) lists all algorithms. Seven reference CSV tables are in `dmn_knw_gnrtr/`.

## Notation and parameters

Each module states its purpose, paper references, inputs and outputs. Follow the [syntax guide](specs/pseudocode_conventions.md) for notation and the [shared interfaces](specs/interface_contracts.md) for fields passed between modules.

The [parameter table](specs/implementation_parameters.md) lists configuration choices; [paper settings](specs/paper_settings.yaml) records reported values; the [equation crosswalk](specs/paper_to_pseudocode.md) locates the relevant procedures.

The five [method chapters](pseudocode/README.md) offer a shorter reading route. [Numerical examples](validation/scripts/README.md) exercise the main equations, and [data and configuration](REPRODUCIBILITY.md) describes implementation inputs.
