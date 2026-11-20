# Paper to pseudocode crosswalk

Page numbers refer to the printed 1–37 pagination of the [published article](../docs/assets/knowcast-paper.pdf).

| Paper location | Method requirement | Pseudocode chapter |
| --- | --- | --- |
| §3.2, pp.7–9, Fig.3 and Eq.1 | Encoder and decoder; future inputs available at the forecast origin | [04 · KITP](../pseudocode/04_kitp_and_forecasting.md) |
| §4.2.1, p.11, Eq.2 | Actual, finished and unfinished demand; conservation at the cutoff | [01 · Data](../pseudocode/01_data_and_observability.md) |
| §4.2.2, p.11, Eq.3–4 | Training-only destination map fixed across splits; residual aggregation | [01 · Data](../pseudocode/01_data_and_observability.md) |
| §4.2.3–4.2.4, pp.11–13, Eq.5–13 | Day/week/month references, unfinished allocation and fusion | [01 · Data](../pseudocode/01_data_and_observability.md) and [04 · KITP](../pseudocode/04_kitp_and_forecasting.md) |
| §4.3.1–4.3.2, pp.13–14, Eq.14–16 | Station context, residual-aware destination totals and squared generation loss | [02 · Generation](../pseudocode/02_generation_and_distribution.md) |
| §4.3.3, pp.14–15, Eq.17–19 | Progressive loss weighting and original-scale OD MAE | [02 · Generation](../pseudocode/02_generation_and_distribution.md) and [05 · Training](../pseudocode/05_training_and_evaluation.md) |
| §4.4, pp.15–16, Eq.20 | Fixed calibrated deterrence coefficient, exponential impedance, time-specific IPF and autoencoder | [02 · Distribution](../pseudocode/02_generation_and_distribution.md) |
| §4.5, pp.16–18, Eq.21–24 | Trainable path utility, OD-wise Logit and section traversal probabilities | [03 · Path choice](../pseudocode/03_path_choice_and_service.md) |
| §4.6, pp.18–19, Eq.25–27 | Section-probability and service-frequency representations | [03 · Service](../pseudocode/03_path_choice_and_service.md) |
| §4.7, p.19, Eq.28–30 | Path flows, directed-section flows and horizon-specific averages; residual exclusion | [03 · Assignment](../pseudocode/03_path_choice_and_service.md) |
| §4.8, pp.19–20, Eq.31–33 | Graph attention, gated recurrence and station hidden-state collection | [04 · KITP](../pseudocode/04_kitp_and_forecasting.md) |
| §5.1, pp.20–22, Table 2 | Data partition, metrics and hyperparameter search | [05 · Training](../pseudocode/05_training_and_evaluation.md) and [settings](paper_settings.yaml) |
| §5.2–5.3, pp.23–29, Tables 3–6 | Baselines and ablation results reported in the paper | [05 · Evaluation](../pseudocode/05_training_and_evaluation.md) |
| §5.4–5.6, pp.29–33, Tables 7–8 | Runtime, indirect load assessment and task-setting sensitivity | [05 · Evaluation](../pseudocode/05_training_and_evaluation.md) |

## Configuration

[Paper settings](paper_settings.yaml) lists reported numerical values. [Implementation parameters](implementation_parameters.md) covers window and batch sizes, layer layouts, calibration, IPF stopping rules, historical-reference handling, normalization and optimization choices supplied by configuration.

## Algorithm modules

The chapters introduce the method; the following modules give the detailed procedures:

| Equations / responsibility | Primary standalone files |
| --- | --- |
| Eq. (1), encoder/decoder with future knowledge | [Net_0207](../models/Net_0207.pseudo), [ODNet](../models/OD_Net_att.pseudo) |
| Eq. (2), cutoff finished/unfinished observations | [OD generation](../dmn_knw_gnrtr/generating_array_OD.pseudo), [timestamp alignment](../dmn_knw_gnrtr/generating_domain_knowledge_with_timestamp.pseudo) |
| Eqs. (3)-(4), fixed retained destinations and residual | [OD generation](../dmn_knw_gnrtr/generating_array_OD.pseudo), [loader](../lib/utils_CUROP.pseudo) |
| Eqs. (5)-(8), three references and unfinished allocations | [OD generation](../dmn_knw_gnrtr/generating_array_OD.pseudo), [loader](../lib/utils_CUROP.pseudo) |
| Eqs. (9)-(13), three recurrent branches and gated fusion | [ODNet](../models/OD_Net_att.pseudo), [GATRU](../models/GATRUCell.pseudo) |
| Eq. (14), demographic/temporal generation features | [Census/raster processing](../dmn_knw_gnrtr/trip_generation_info_generating.pseudo), [signal generation](../dmn_knw_gnrtr/PYGT_signal_generation_one_hot.pseudo) |
| Independent generation pretraining | [DCRNN wrapper](../dmn_knw_gnrtr/RecurrentGCN_for_trip_generation.pseudo), [training](../dmn_knw_gnrtr/run_PYGT_0917.pseudo), [evaluation](../dmn_knw_gnrtr/test_PYGT_0917.pseudo) |
| Eqs. (15)-(19), residual-aware station loss and progressive weighting | [Training](../train_save_history.pseudo), [model generation outputs](../models/Net_0207.pseudo) |
| Eq. (20), exponential deterrence and dynamic IPF | [Offline calibration](../dmn_knw_gnrtr/fit_trip_distribution_model.pseudo), [model gravity component](../models/Net_0207.pseudo) |
| Eqs. (21)-(24), feasible paths, utility and probability | [Route generation](../dmn_knw_gnrtr/Generating_Metro_Related_data.pseudo), [incidence](../dmn_knw_gnrtr/generating_OD_section_pssblty_sparse_array_0209.pseudo), [model utility/logit](../models/Net_0207.pseudo) |
| Eqs. (25)-(27), section and line-frequency embeddings | [Timetables](../metro_data_convertor/Get_Time_DepartFreDic.pseudo), [frequency arrays](../metro_data_convertor/Process_Time_DepartFreDic.pseudo), [knowledge encoders](../models/Net_0207.pseudo) |
| Eqs. (28)-(30), retained-OD path/section flow outputs | [Model assignment](../models/Net_0207.pseudo), [assignment summaries](../train_evaluate_functions.pseudo) |
| Eqs. (31)-(33), attention and gated recurrence | [GAT](../models/gat.pseudo), [GATRU](../models/GATRUCell.pseudo) |
| Eqs. (34)-(37), metrics and horizon aggregation | [Metrics](../lib/metrics.pseudo), [evaluation](../train_evaluate_functions.pseudo) |
| Table 2, search and selected settings | [Search](../bayes_opt.pseudo), [reported settings](paper_settings.yaml) |
| Tables 3-7 and §5.6, baselines, ablations, efficiency and task sensitivity | [Experiment protocol](../experiments/ablation_and_sensitivity.pseudo) |
| §5.5 and Table 8, indirect dynamic train-load assessment | [Operational assessment](../experiments/operational_load_assessment.pseudo) |
| End-to-end artifact preparation and lifecycle | [Pipeline](../generating_domain_knowledge_no_DO_clean.pseudo), [shared contracts](interface_contracts.md) |

The [module index](../validation/MODULES.md) also lists supporting data, network and plotting utilities.
