# Original-file coverage

This inventory compares the preserved source snapshot with each standalone pseudocode replacement. Declaration coverage and line counts do not prove behavioral equivalence or experiment reproduction. Coverage is derived from actual CLASS/PROCEDURE headers, without embedded source annotations. Duplicate overloaded declarations are counted individually; added procedures are excluded from the original-symbol count.

| Original file | Replacement | Source / pseudo lines | Original / covered declarations |
| --- | --- | ---: | ---: |
| `bayes_opt.py` | [bayes_opt.pseudo](../bayes_opt.pseudo) | 173 / 275 | 2 / 2 |
| `dmn_knw_gnrtr/fit_trip_distribution_model.py` | [dmn_knw_gnrtr/fit_trip_distribution_model.pseudo](../dmn_knw_gnrtr/fit_trip_distribution_model.pseudo) | 106 / 114 | 4 / 4 |
| `dmn_knw_gnrtr/generating_array_OD.py` | [dmn_knw_gnrtr/generating_array_OD.pseudo](../dmn_knw_gnrtr/generating_array_OD.pseudo) | 465 / 536 | 9 / 9 |
| `dmn_knw_gnrtr/generating_domain_knowledge_with_timestamp.py` | [dmn_knw_gnrtr/generating_domain_knowledge_with_timestamp.pseudo](../dmn_knw_gnrtr/generating_domain_knowledge_with_timestamp.pseudo) | 84 / 183 | 1 / 1 |
| `dmn_knw_gnrtr/Generating_Metro_Related_data.py` | [dmn_knw_gnrtr/Generating_Metro_Related_data.pseudo](../dmn_knw_gnrtr/Generating_Metro_Related_data.pseudo) | 285 / 308 | 9 / 9 |
| `dmn_knw_gnrtr/generating_OD_section_pssblty_sparse_array_0209.py` | [dmn_knw_gnrtr/generating_OD_section_pssblty_sparse_array_0209.pseudo](../dmn_knw_gnrtr/generating_OD_section_pssblty_sparse_array_0209.pseudo) | 246 / 176 | 3 / 3 |
| `dmn_knw_gnrtr/processing_sql.py` | [dmn_knw_gnrtr/processing_sql.pseudo](../dmn_knw_gnrtr/processing_sql.pseudo) | 135 / 112 | 1 / 1 |
| `dmn_knw_gnrtr/PYGT_signal_generation_one_hot.py` | [dmn_knw_gnrtr/PYGT_signal_generation_one_hot.pseudo](../dmn_knw_gnrtr/PYGT_signal_generation_one_hot.pseudo) | 142 / 154 | 1 / 1 |
| `dmn_knw_gnrtr/RecurrentGCN_for_trip_generation.py` | [dmn_knw_gnrtr/RecurrentGCN_for_trip_generation.pseudo](../dmn_knw_gnrtr/RecurrentGCN_for_trip_generation.pseudo) | 19 / 84 | 3 / 3 |
| `dmn_knw_gnrtr/run_PYGT_0917.py` | [dmn_knw_gnrtr/run_PYGT_0917.pseudo](../dmn_knw_gnrtr/run_PYGT_0917.pseudo) | 67 / 96 | 1 / 1 |
| `dmn_knw_gnrtr/test_PYGT_0917.py` | [dmn_knw_gnrtr/test_PYGT_0917.pseudo](../dmn_knw_gnrtr/test_PYGT_0917.pseudo) | 43 / 70 | 1 / 1 |
| `dmn_knw_gnrtr/trip_generation_info_generating.py` | [dmn_knw_gnrtr/trip_generation_info_generating.pseudo](../dmn_knw_gnrtr/trip_generation_info_generating.pseudo) | 409 / 342 | 5 / 5 |
| `evaluate_button.py` | [evaluate_button.pseudo](../evaluate_button.pseudo) | 186 / 197 | 7 / 7 |
| `generating_domain_knowledge_no_DO_clean.py` | [generating_domain_knowledge_no_DO_clean.pseudo](../generating_domain_knowledge_no_DO_clean.pseudo) | 226 / 316 | 3 / 3 |
| `lib/__init__.py` | [lib/__init__.pseudo](../lib/__init__.pseudo) | 0 / 12 | 0 / 0 |
| `lib/metrics.py` | [lib/metrics.pseudo](../lib/metrics.pseudo) | 23 / 94 | 4 / 4 |
| `lib/utils_CUROP.py` | [lib/utils_CUROP.pseudo](../lib/utils_CUROP.pseudo) | 546 / 919 | 27 / 27 |
| `metro_components/Line.py` | [metro_components/Line.pseudo](../metro_components/Line.pseudo) | 12 / 28 | 3 / 3 |
| `metro_components/LineManager.py` | [metro_components/LineManager.pseudo](../metro_components/LineManager.pseudo) | 68 / 64 | 5 / 5 |
| `metro_components/MetroRequester_SuZhou.py` | [metro_components/MetroRequester_SuZhou.pseudo](../metro_components/MetroRequester_SuZhou.pseudo) | 123 / 115 | 2 / 2 |
| `metro_components/MetroUtil.py` | [metro_components/MetroUtil.pseudo](../metro_components/MetroUtil.pseudo) | 4 / 24 | 1 / 1 |
| `metro_components/OD.py` | [metro_components/OD.pseudo](../metro_components/OD.pseudo) | 13 / 38 | 4 / 4 |
| `metro_components/Path.py` | [metro_components/Path.pseudo](../metro_components/Path.pseudo) | 61 / 62 | 4 / 4 |
| `metro_components/Route.py` | [metro_components/Route.pseudo](../metro_components/Route.pseudo) | 58 / 71 | 4 / 4 |
| `metro_components/Section.py` | [metro_components/Section.pseudo](../metro_components/Section.pseudo) | 13 / 38 | 4 / 4 |
| `metro_components/Station.py` | [metro_components/Station.pseudo](../metro_components/Station.pseudo) | 18 / 31 | 3 / 3 |
| `metro_components/StationManager.py` | [metro_components/StationManager.pseudo](../metro_components/StationManager.pseudo) | 35 / 68 | 4 / 4 |
| `metro_data_convertor/convert_objects_to_dict.py` | [metro_data_convertor/convert_objects_to_dict.pseudo](../metro_data_convertor/convert_objects_to_dict.pseudo) | 12 / 135 | 1 / 1 |
| `metro_data_convertor/Find_project_root.py` | [metro_data_convertor/Find_project_root.pseudo](../metro_data_convertor/Find_project_root.pseudo) | 13 / 32 | 1 / 1 |
| `metro_data_convertor/Generating_logit_probabilities.py` | [metro_data_convertor/Generating_logit_probabilities.pseudo](../metro_data_convertor/Generating_logit_probabilities.pseudo) | 65 / 90 | 2 / 2 |
| `metro_data_convertor/Get_Time_DepartFreDic.py` | [metro_data_convertor/Get_Time_DepartFreDic.pseudo](../metro_data_convertor/Get_Time_DepartFreDic.pseudo) | 130 / 194 | 3 / 3 |
| `metro_data_convertor/Process_Time_DepartFreDic.py` | [metro_data_convertor/Process_Time_DepartFreDic.pseudo](../metro_data_convertor/Process_Time_DepartFreDic.pseudo) | 56 / 85 | 2 / 2 |
| `metro_data_convertor/Reprocessing_OD_visiting_prob.py` | [metro_data_convertor/Reprocessing_OD_visiting_prob.pseudo](../metro_data_convertor/Reprocessing_OD_visiting_prob.pseudo) | 32 / 53 | 1 / 1 |
| `models/__init__.py` | [models/__init__.pseudo](../models/__init__.pseudo) | 0 / 6 | 0 / 0 |
| `models/gat.py` | [models/gat.pseudo](../models/gat.pseudo) | 402 / 211 | 10 / 10 |
| `models/GATRUCell.py` | [models/GATRUCell.pseudo](../models/GATRUCell.pseudo) | 180 / 139 | 8 / 8 |
| `models/GGRUCell.py` | [models/GGRUCell.pseudo](../models/GGRUCell.pseudo) | 181 / 101 | 8 / 8 |
| `models/Net_0207.py` | [models/Net_0207.pseudo](../models/Net_0207.pseudo) | 509 / 1249 | 23 / 23 |
| `models/OD_Net_att.py` | [models/OD_Net_att.pseudo](../models/OD_Net_att.pseudo) | 302 / 274 | 9 / 9 |
| `models/rgcn.py` | [models/rgcn.pseudo](../models/rgcn.pseudo) | 104 / 74 | 7 / 7 |
| `train_evaluate_functions.py` | [train_evaluate_functions.pseudo](../train_evaluate_functions.pseudo) | 192 / 437 | 6 / 6 |
| `train_save_history.py` | [train_save_history.pseudo](../train_save_history.pseudo) | 603 / 831 | 4 / 4 |

Total: **42 files, 200 original declarations**. Original: **6341 lines** (5431 nonblank); replacements: **8438 lines** (8063 nonblank).

Exact hashes, missing/additional declaration lists and source signatures are available in [source_coverage.json](source_coverage.json) and [source_inventory.json](source_inventory.json).

The seven original CSV reference tables remain at their original paths. The [reference checks](reference_checks/README.md) provide structural and independent mathematical checks for this edition.
