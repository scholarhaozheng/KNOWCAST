# Shared interfaces and units

These interfaces define fields, units and identities passed between modules.

## Dimensions and identities

| Name | Meaning |
| --- | --- |
| `Q`, `B` | Dataset samples and current tensor batch rows; padding rows carry a false validity mask |
| `N` | Stations in the fixed station registry |
| `R`, `K=R+1` | Retained destinations per origin and compressed width including the residual |
| `lh`, `lf` | Independent encoder and forecast lengths |
| `L_lines` | Service-line count |
| `station_order_id` | Station IDs and tensor-axis order |
| `RDS_id` | Training-fitted retained-destination mapping |
| `residual_shares_id` | Training-fitted allocation weights for nonretained destinations |
| `transform_id` | Feature layout, fitted parameters and training-fit identity |
| `sample_id` | Forecast-window identity; batched form is `sample_ids` |

Station tensors align by `station_order_id`. For origin `o`, retained column `k` maps to `RDS[o,k]`; residual column `R` aggregates the other destinations. Array indices are zero-based: retained slots `0..R-1`, residual slot `R`, horizons `0..lf-1`. A mathematical horizon label `lambda=1..lf` maps to array index `lambda-1`.

Training removes padded rows before defining `actual_B` and computing losses. Inference exports and metrics select rows with true `valid_mask`. The field `sample_count` counts valid sample identities.

Timestamps retain their timezone and civil date. `service_date` groups overnight operations. Eq. (14)'s `TOD` is elapsed hours since local calendar midnight in `[0,24)`.

`cfg.domain_knowledge.generation_covariates` selects `full` or `temporal_only`. The full profile includes demographics; the temporal profile contains `Wknd`, `TOD`, `Season` and `Event`. Feature schemas, transforms and auxiliary checkpoints carry the same profile ID.

## Stored samples

| Field | Shape / semantics |
| --- | --- |
| `finished` | `[Q,lh,N,K]`, finished OD observed by the forecast cutoff, in passenger counts |
| `unfinished_raw` | `[Q,lh,N,1]`, observed entries with journeys unfinished at the forecast origin |
| `history_day_raw`, `history_week_raw`, `history_month_raw` | `[Q,lh,N,K]`, completed reference OD with tags `reference_od` and `passenger_count` |
| `y` | `[Q,lf,N,K]`, future supervised OD labels |
| `xtime`, `ytime` | `[Q,lh]`, `[Q,lf]`, per-step timestamps |
| `forecast_origin` | `[Q]`, information cutoff |
| `sample_id` | `[Q]`, stable sample IDs |
| `valid_mask` | `[Q]`, true for valid samples and false for padding |
| `target_observation_mask` | `[Q,lf,N,K]`, target-cell observation validity; a complete-observation objective uses true cells throughout |
| `encoder_knowledge`, `decoder_knowledge` | Sequences of lengths `lh`, `lf`, with availability metadata |
| `auxiliary_context_by_task` | Production/attraction input windows or cutoff-safe references, with auxiliary lengths, target offset, graph/scaler/checkpoint IDs and availability |
| `dynamic_impedance` | `[N,N]` per active distribution step, with units, target time, availability and source version |
| Mapping/transform IDs | Shared manifest for all arrays |

The variant manifest lists active optional fields: references, allocations, auxiliary contexts, generic inputs, paths and service records. Loaders, shufflers and collators process those fields; a missing active field is an error. Diagnostics can retain additional fields.

Disk aliases such as `unfinished`, `history_day/week/month`, `history` and `yesterday` are mapped by the manifest. Raw historical references and runtime allocations have separate names and representation tags. The compatibility adapter maps runtime `history` to allocated week input and `yesterday` to allocated day input.

An observed zero historical row produces uniform `1/K` destination shares. Missing rows follow the configured missing-reference policy. Month references use the configured calendar-month rule. The history-availability policy also determines whether April observations supply context for May forecasts.

## Loader representation

The loader computes each allocation in passenger counts:

`allocation_r_count[o,k] = unfinished_raw[o] × normalized_nonnegative_reference_r[o,k]`.

Each allocated row sums to `unfinished_raw[o]`. The training-fitted input transform then produces `allocation_day/week/month`, tagged `representation_kind=unfinished_allocation` and a transform ID. The model consumes these transformed allocations directly.

`DatasetBundle` exposes `frozen_train_scaler` (alias `scaler`), `expected_counts[split]`, `targets_by_id`, `split_sample_order`, transforms, loaders and `component_context`. Training-fit mode estimates transforms; evaluation restores them.

Each `targets_by_id[split][sample_id]` stores `raw_target_count`, normalized targets, target times and observation masks. Evaluation checks the inverse transform against the raw count within numerical tolerance and uses the raw count as truth. Zero-flow diagnostics consequently use exact observed zeros.

## Model input and output

`BatchBundle` contains:

- `encoder_steps[lh]` and `decoder_steps[lf]`, including graph layout, features and availability;
- `target_od[B,lf,N,K]` for training and evaluation;
- `sample_ids`, `forecast_origin`, `xtime`, `ytime`, `valid_mask` and mapping/transform IDs;
- raw unfinished counts and normalized unfinished allocations;
- target-specific auxiliary generation contexts and future knowledge.

The inference input view contains the features and metadata. Supervised targets remain with the caller. Selecting a shorter forecast applies identical indices to targets, `ytime`, decoder knowledge, masks and ragged path metadata, yielding exactly `lf` steps. Truth joins use `sample_id` and target timestamps.

Auxiliary windows have their own lengths and target offsets. Slicing a main-model step preserves the complete nested auxiliary window. Each context is matched to that step's timestamp.

`ForecastBundle` contains:

- `od_normalized[B,lf,N,K]`;
- `assignment_context_by_step[h][sample_id]` when assignment is active, with the same forward pass's real-OD probabilities, feasible sets, target-time incidence, ordered line-aware routes, timestamps and version IDs;
- optional probability/feasible-set aliases, diagnostics and original-scale path/section flows;
- `generation_targets` when guidance is active: separate `production` and `attraction` arrays `[B,lf,N]`, timestamps, transform IDs, availability and parameter policy;
- mapping/transform IDs for prediction exports.

`diagnostics.generation_targets`, when present, aliases the top-level record. OD-only adapters serve consumers that require only OD outputs.

## Transport knowledge

| Object | Shape / semantics |
| --- | --- |
| Section | Directed `(source_station,destination_station,line_id)`, with a physical `(source,destination)` projection |
| Feasible paths | Ordered real-OD alternatives with availability masks and stable path IDs |
| Incidence | Sparse mapping from path IDs to distinct directed physical sections; excluded paths contribute zero |
| CP factors | Optional approximate representation, identified separately from exact incidence |
| Path probabilities | Sum to one within each nonempty feasible OD set; excluded alternatives have zero probability |
| `K7` frequency | `[N,N,L_lines]`, stored in trains/hour |
| No service | Known raw frequency zero, distinct from missing timetable data |
| Impedance `C_t` | `[N,N]`, generalized cost with units and target time; the manifest identifies static or dynamic sources |
| Gravity marginals | Production/attraction vectors in passenger counts with equal total demand |
| Gravity output | Time-specific IPF solution with a fixed calibrated positive `zeta` |

Trains/hour is the storage convention used here. Convert headway in minutes using `60/headway`, or seconds using `3600/headway`. Interval train count is frequency times `interval_minutes/60`. Feasibility uses raw service support.

`ServiceRecord` contains `raw_frequency`, `encoder_frequency`, `structural_support`, `known_status`, `raw_units=trains_per_hour`, `encoder_scale`, `frequency_transform_id`, `line_order_id` and availability. The training-fitted transform normalizes the encoder view once. Missingness remains in `known_status`.

The assignment record stores `path_probabilities[(o,d),path_id]`, `feasible_sets[(o,d)]`, `incidence[(o,d),path_id]` and `line_aware_routes[(o,d),path_id]`. Gather each OD's probability vector in feasible-set order. Incidence supplies the binary section indicator in Eq. (24); the ordered line-aware route supplies traversal order and travel timing for operational load assessment.

Inputs follow active dependencies. Generation guidance and distribution features require independent generation artifacts. Service features and path feasibility require raw service. Each structured encoder uses its own pretraining artifact. Generic inputs follow their configured demographic, calendar and service dependencies. Preparation reports missing dependencies before constructing the variant.

For Case 3b, `build_generic_exogenous_adapter` supplies the schema, encoder specification and factory. The producer creates `generic_exogenous_inputs`; `Net` constructs and registers one `ConfiguredMLPEncoder` through `build_generic_encoder`. Its dense layers, activations and dropout produce `knowledge.generic_exogenous_by_origin[N,F_mlp]`. Feature order, station alignment, availability and the fitted transform accompany the inputs.

Distribution reads `dynamic_impedance` from each selected knowledge step. The producer uses `impedance_store`; batching preserves each matrix and its metadata while slicing the outer sample and main-model time axes.

## Objective and evaluation

`N_train` counts predicted matrices under the configured overlap/multiplicity rule. Eq. (18) is OD MAE in passenger-flow units; Eq. (16) sums squared station residuals. The minibatch estimator defines sampling, reduction and accumulation for both terms.

Residual destination flow contributes to attractions through origin-specific training shares. Differentiable scatter and sum operations connect these totals to OD predictions. Frozen auxiliary targets can be detached while the prediction-side loss retains gradients.

Inference feeds preceding predictions into the decoder. The configured nonnegativity policy applies to metrics, exports and assignment. Compute MAE, RMSE and WMAPE for each horizon, then average each metric over horizons. `MAPE_net_od` is the compatibility name for the WMAPE ratio; `WMAPE_percent_od` is percent.

`EvaluationResult` contains `sample_count`, per-horizon metrics, aggregates, ordered sample metadata, output-policy/transform IDs and optional prediction/truth/mask arrays. These arrays are the ones used for the metrics. Missing or duplicate sample identities raise an alignment error.

## Parameter ownership

Construct and register the main network, active utility layers and pretrained encoders once. Save their parameters in the selected checkpoint. Gravity `zeta`, station mappings, residual shares and fitted transforms remain fixed. The generation-model configuration selects frozen or fine-tuned parameters during main training. Evaluation freezes all parameters.

Forward passes reuse registered modules and fitted transforms. Checkpoint restore matches the architecture, tensor keys, identities and availability metadata.

## Experiment interfaces

[Ablation and sensitivity](../experiments/ablation_and_sensitivity.pseudo) defines component flags, learner adapters, baseline interfaces, validation-based selection, runtime measurement and task changes. Each baseline and the alternative GConvLSTM learner supplies an implementation of its forecast adapter.

[Operational load assessment](../experiments/operational_load_assessment.pseudo) consumes original-scale OD and the same forward pass's assignment context. Its `ForecastView` carries prediction validity, sample IDs, target times, forecast origins and a transform receipt. Prediction validity and target-observation masks are separate. Entry intervals begin at the target time and have the configured duration.

Contributions retain line, direction, hops and entry interval. A trajectory selects one forecast per entry cohort. Gaussian interval weights distribute flow to receiving intervals; the mass ledger tracks received, discarded and outside-grid mass. Required entry support determines coverage. Train-load rows retain source forecast IDs and match observations by section, line, direction, date, train and time. [Implementation parameters](implementation_parameters.md) lists calibration, tail and matching choices.
