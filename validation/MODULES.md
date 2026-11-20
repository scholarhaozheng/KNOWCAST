# Module index

46 pseudocode modules contain 345 class and procedure declarations. The [reading guide](../PSEUDOCODE_GUIDE.md) groups these modules by stage of the forecasting method.

| Module | Purpose | Declarations |
| --- | --- | ---: |
| [bayes_opt.pseudo](../bayes_opt.pseudo) | resumable hyperparameter search, validation-only selection, progress figures, | 5 |
| [config.pseudo](../config.pseudo) | Define runtime arguments and entrypoint modes. | 6 |
| [dmn_knw_gnrtr/Generating_Metro_Related_data.pseudo](../dmn_knw_gnrtr/Generating_Metro_Related_data.pseudo) | Build metro topology, enumerate candidate routes, and index OD/path/section incidences. | 9 |
| [dmn_knw_gnrtr/PYGT_signal_generation_one_hot.pseudo](../dmn_knw_gnrtr/PYGT_signal_generation_one_hot.pseudo) | Build aligned graph-temporal signals for production and attraction estimation. | 2 |
| [dmn_knw_gnrtr/RecurrentGCN_for_trip_generation.pseudo](../dmn_knw_gnrtr/RecurrentGCN_for_trip_generation.pseudo) | Recurrent generation estimator used independently of the main OD learner. | 4 |
| [dmn_knw_gnrtr/fit_trip_distribution_model.pseudo](../dmn_knw_gnrtr/fit_trip_distribution_model.pseudo) | Calibrate the fixed deterrence coefficient and balance gravity flows at each time. | 4 |
| [dmn_knw_gnrtr/generating_OD_section_pssblty_sparse_array_0209.pseudo](../dmn_knw_gnrtr/generating_OD_section_pssblty_sparse_array_0209.pseudo) | Produce time-indexed candidate attributes, exact binary incidence and optional CP storage. | 3 |
| [dmn_knw_gnrtr/generating_array_OD.pseudo](../dmn_knw_gnrtr/generating_array_OD.pseudo) | AFC extraction, station reconciliation, OD construction, compression, windows, and historical references. | 12 |
| [dmn_knw_gnrtr/generating_domain_knowledge_with_timestamp.pseudo](../dmn_knw_gnrtr/generating_domain_knowledge_with_timestamp.pseudo) | Join sample-conditioned service, paths, impedance and covariates on exact timestamps. | 3 |
| [dmn_knw_gnrtr/processing_sql.pseudo](../dmn_knw_gnrtr/processing_sql.pseudo) | Inspect AFC table schemas, combine approved source tables, and prepare date-range extracts. | 2 |
| [dmn_knw_gnrtr/run_PYGT_0917.pseudo](../dmn_knw_gnrtr/run_PYGT_0917.pseudo) | Independent pretraining of production or attraction estimation. | 1 |
| [dmn_knw_gnrtr/test_PYGT_0917.pseudo](../dmn_knw_gnrtr/test_PYGT_0917.pseudo) | Reload and evaluate the independent generation estimator. | 1 |
| [dmn_knw_gnrtr/trip_generation_info_generating.pseudo](../dmn_knw_gnrtr/trip_generation_info_generating.pseudo) | Station-district reconciliation, census cleaning, raster population extraction, feature fusion. | 6 |
| [evaluate_button.pseudo](../evaluate_button.pseudo) | restore a declared checkpoint, evaluate it once, and export aligned predictions | 8 |
| [experiments/ablation_and_sensitivity.pseudo](../experiments/ablation_and_sensitivity.pseudo) | Compare KNOWCAST components, learners, convergence, runtime and task settings. | 11 |
| [experiments/operational_load_assessment.pseudo](../experiments/operational_load_assessment.pseudo) | Convert retained-OD forecasts to section flows and compare aligned train loads. | 18 |
| [generating_domain_knowledge_no_DO_clean.pseudo](../generating_domain_knowledge_no_DO_clean.pseudo) | Prepare network, demand, pretrained knowledge and sample-aligned artifacts. | 4 |
| [lib/__init__.pseudo](../lib/__init__.pseudo) | Empty package initializer for metrics and data utilities. | 0 |
| [lib/metrics.pseudo](../lib/metrics.pseudo) | Calculate OD forecasting metrics. | 6 |
| [lib/utils_CUROP.pseudo](../lib/utils_CUROP.pseudo) | Load, align, normalize, batch and transfer all forecasting inputs. | 30 |
| [metro_components/Line.pseudo](../metro_components/Line.pseudo) | Store stations in their physical sequence along one line. | 3 |
| [metro_components/LineManager.pseudo](../metro_components/LineManager.pseudo) | Manage line sequences, adjacency state and line diagnostics. | 6 |
| [metro_components/MetroRequester_SuZhou.pseudo](../metro_components/MetroRequester_SuZhou.pseudo) | Load a station registry, line sequences, directed adjacency and geographic distances. | 3 |
| [metro_components/MetroUtil.pseudo](../metro_components/MetroUtil.pseudo) | Resolve canonical and legacy line identifiers. | 1 |
| [metro_components/OD.pseudo](../metro_components/OD.pseudo) | Store a directed real-station OD identity and conditional section annotations. | 4 |
| [metro_components/Path.pseudo](../metro_components/Path.pseudo) | Extend candidate paths without mutating accepted paths. | 4 |
| [metro_components/Route.pseudo](../metro_components/Route.pseudo) | Construct the shortest eligible contiguous same-line segment between endpoints. | 4 |
| [metro_components/Section.pseudo](../metro_components/Section.pseudo) | Represent one directed physical section on one serving line. | 4 |
| [metro_components/Station.pseudo](../metro_components/Station.pseudo) | Represent one physical station shared by all serving lines. | 3 |
| [metro_components/StationManager.pseudo](../metro_components/StationManager.pseudo) | Maintain a canonical station registry with bijective matrix-axis indices. | 5 |
| [metro_data_convertor/Find_project_root.pseudo](../metro_data_convertor/Find_project_root.pseudo) | Resolve the project directory from configuration or an upward marker search. | 1 |
| [metro_data_convertor/Generating_logit_probabilities.pseudo](../metro_data_convertor/Generating_logit_probabilities.pseudo) | Compute offline path and directed section probability artifacts. | 2 |
| [metro_data_convertor/Get_Time_DepartFreDic.pseudo](../metro_data_convertor/Get_Time_DepartFreDic.pseudo) | Expand timetable rows into dated, directed, line-specific service records. | 4 |
| [metro_data_convertor/Process_Time_DepartFreDic.pseudo](../metro_data_convertor/Process_Time_DepartFreDic.pseudo) | Tensorize line-specific service records on the shared station and line axes. | 2 |
| [metro_data_convertor/Reprocessing_OD_visiting_prob.pseudo](../metro_data_convertor/Reprocessing_OD_visiting_prob.pseudo) | Tensorize OD-conditioned directed section probabilities. | 1 |
| [metro_data_convertor/convert_objects_to_dict.pseudo](../metro_data_convertor/convert_objects_to_dict.pseudo) | Encode and decode network records while preserving shared identities and typed values. | 3 |
| [models/GATRUCell.pseudo](../models/GATRUCell.pseudo) | Graph attention followed by a recurrent update. | 9 |
| [models/GGRUCell.pseudo](../models/GGRUCell.pseudo) | Legacy relation-mixture recurrent alternative. | 9 |
| [models/Net_0207.pseudo](../models/Net_0207.pseudo) | Build knowledge features and forecast compressed OD flows. | 66 |
| [models/OD_Net_att.pseudo](../models/OD_Net_att.pseudo) | Recurrent encoder branches, unfinished-demand fusion and OD decoder. | 19 |
| [models/__init__.pseudo](../models/__init__.pseudo) | Package containing the OD forecaster, graph recurrent cells and graph operators. | 0 |
| [models/gat.pseudo](../models/gat.pseudo) | Multihead graph attention with optional edge features. | 13 |
| [models/rgcn.pseudo](../models/rgcn.pseudo) | Basis-decomposed relation-mixture graph convolution. | 8 |
| [train_evaluate_functions.pseudo](../train_evaluate_functions.pseudo) | inference, original-scale metrics, run directories, and learning-rate scheduling | 9 |
| [train_save_history.pseudo](../train_save_history.pseudo) | main training, generation consistency, checkpoint selection, history and diagnostics | 21 |
| [visualization_tools/plot_graph.pseudo](../visualization_tools/plot_graph.pseudo) | Plot training curves, utility trajectories, metrics and OD predictions. | 6 |
