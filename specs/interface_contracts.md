# Shared interfaces and units

These contracts define the fields, units and identities passed between modules. Alternative physical layouts use adapters that preserve these meanings.

## Dimensions and identities

| Name | Meaning |
| --- | --- |
| `Q`, `B` | Genuine dataset samples and current tensor batch rows; `B` may include explicitly masked storage padding, which never counts as an observation |
| `N` | Number of real stations in the immutable station registry |
| `R`, `K=R+1` | Retained real destinations per origin and compressed width including one residual |
| `lh`, `lf` | Independent encoder and forecast lengths |
| `L_lines` | Number of service lines; never confuse with recurrent layer count |
| `station_order_id` | Content identity of station IDs and their tensor-axis ordering |
| `RDS_id` | Identity of training-fitted per-origin retained destination mappings |
| `residual_shares_id` | Identity of training-fitted nonretained destination allocation weights |
| `transform_id` | Identity of feature layout, fitted transform parameters and fit provenance |
| `sample_id` | Unique forecast-window identity; batched form is `sample_ids` |

Station axes with equal sizes are not interchangeable if their ordering IDs differ. Compressed column `k` maps to `RDS[o,k]` separately for each origin `o`. Column `R` is never interpreted as a real destination.

The training adapter removes padded rows before defining its genuine `actual_B` and computing losses. Inference may carry padded rows through a forward pass, but exports and metrics retain only true `valid_mask` rows. `sample_count` always counts genuine sample identities, not the physical tensor length.

Software array indices in the shared contracts are zero-based: retained columns `0..R-1`, residual column `R`, horizons `0..lf-1`. A paper equation written with one-based mathematical indices needs an explicit conversion before indexing these records.

Absolute timestamps retain their timezone and civil date. `service_date` is a separate operating-day label for overnight timetable grouping; changing that boundary does not redefine Eq. (14)'s `TOD`, which is elapsed hours since local calendar midnight in `[0,24)`. Covariate profiles are selected with `cfg.domain_knowledge.generation_covariates`: `full` includes the demographic block; `temporal_only` supplies `Wknd`, `TOD`, `Season` and `Event` without loading the demographic block. Feature schemas, transforms and auxiliary checkpoints must carry the same profile identity.

## On-disk sample record

| Field | Shape / semantics |
| --- | --- |
| `finished` | `[Q,lh,N,K]`, original-scale cutoff-observable finished OD |
| `unfinished_raw` | `[Q,lh,N,1]`, observed entries whose destination is unavailable at forecast origin |
| `history_day_raw`, `history_week_raw`, `history_month_raw` | `[Q,lh,N,K]`, raw completed historical reference OD, tagged `reference_od` and `passenger_count` |
| `y` | `[Q,lf,N,K]`, future OD labels, kept separate from inference inputs |
| `xtime`, `ytime` | `[Q,lh]`, `[Q,lf]`, full per-step timestamps |
| `forecast_origin` | `[Q]`, information cutoff for that forecast |
| `sample_id` | `[Q]`, stable sample IDs; plural alias must be explicit |
| `valid_mask` | `[Q]` sample validity; on-disk genuine records normally true, loader-created padding is false |
| `target_observation_mask` | `[Q,lf,N,K]` target-cell observation validity, separate from padding; complete-observation objectives require true cells throughout |
| `encoder_knowledge`, `decoder_knowledge` | Separate sequences of lengths `lh`, `lf`, with available-at metadata |
| `auxiliary_context_by_task` inside knowledge records | Complete production/attraction DCRNN input windows or verified cutoff-safe references, with independent auxiliary lengths, target offset, graph/scaler/checkpoint IDs and availability |
| `dynamic_impedance` inside each knowledge step | `[N,N]` when distribution features are active; units, target time, available-at time and source version accompany the matrix |
| Mapping/transform IDs | One coherent manifest referenced by every array |

The table describes the complete model's schema. Variant manifests explicitly identify active optional fields: historical references/allocations, auxiliary contexts, generic inputs, path records and service records follow their component dependencies. Load, shuffle and collate operations skip absent disabled fields while rejecting missing required fields. An explicit archival or diagnostic request may retain extra fields; disabled components do not require fabricated zero arrays.

Legacy disk names `unfinished`, `history_day/week/month`, `history` and `yesterday` are accepted only through a manifest that states their exact semantics. A misleading legacy field name is never used to guess which calendar reference it contains.

Runtime loaders use the explicit `allocation_*` fields. They do not reuse ambiguous runtime `history_day/week/month` names for normalized allocations; those names can be disk aliases of raw references only. The original `history`/`yesterday` argument compatibility adapter must map to documented allocated week/day inputs and retain representation/scale tags.

Historical references distinguish **observed zero** from **missing**. An observed zero row uses uniform `1/K` allocation; missing data follows an explicitly supplied policy. Month means the declared calendar-month reference, not an assumed 30-day offset. Whether April context may be used for May forecasts is an unresolved experiment choice.

## Loader representation

The loader computes each unfinished allocation once, in passenger-count units:

`allocation_r_count[o,k] = unfinished_raw[o] × normalized_nonnegative_reference_r[o,k]`.

It verifies that each row sums to `unfinished_raw[o]`, then applies the declared training-fitted model-input transform to produce `allocation_day/week/month`. These model fields carry `representation_kind=unfinished_allocation` and their transform ID. The model consumes them directly. It cannot renormalize an already transformed allocation or multiply by unfinished demand again.

`DatasetBundle` exposes `frozen_train_scaler` (legacy alias `scaler`), `expected_counts[split]`, `targets_by_id`, `split_sample_order`, explicit transforms, loaders and `component_context`. Fitting is allowed only in a declared training-fit mode. Evaluation restores compatible frozen transforms before loading data.

Each `targets_by_id[split][sample_id]` record preserves `raw_target_count` in original passenger-count units alongside normalized targets, target times and observation masks. Normalization never overwrites that raw record. Evaluation checks the inverse-transformed target against the raw count within the declared numerical tolerance, then uses the raw count as canonical truth. This keeps a genuine zero exactly zero for the paper's zero-flow diagnostics instead of testing a floating-point round trip for exact equality.

## Model input and output

`BatchBundle` contains:

- `encoder_steps[lh]` and `decoder_steps[lf]` with graph layout, features and availability metadata;
- `target_od[B,lf,N,K]` for training/evaluation callers, excluded from the model's inference input view;
- `sample_ids`, `forecast_origin`, `xtime`, `ytime`, `valid_mask` and all mapping/transform IDs;
- raw unfinished counts for conservation checks and normalized unfinished allocations for recurrent inputs;
- target-specific auxiliary generation features and independently aligned future knowledge.

If a longer stored forecast is selected, the collator uses the same target indices to slice `target_od`, `ytime`, decoder knowledge, masks and ragged path metadata. It must return exactly `lf` steps. Truth joining uses both `sample_id` and the selected target timestamps.

Auxiliary DCRNN window lengths are not assumed equal to the main model's `lh`/`lf`. Selecting a main-model knowledge step preserves its complete nested auxiliary context and target offset; it does not truncate that inner window or replace a Seq2Seq forecast with a stateless current-feature projection. Reusing a service snapshot does not authorize reusing generation context at another timestamp.

`ForecastBundle` contains:

- `od_normalized[B,lf,N,K]`;
- path probabilities and feasible-set identities by forecast step when assignment is active;
- `assignment_context_by_step[h][sample_id]`, the canonical same-forward-pass record containing real-OD path probabilities, feasible sets, exact target-time incidence, ordered line-aware routes, timestamp and version IDs; standalone probability/feasible-set outputs are aliases of these fields;
- diagnostics and optional original-scale path/section flow outputs;
- `generation_targets` when generation guidance is active: separate `production` and `attraction` arrays `[B,lf,N]`, target timestamps, transform IDs, availability and parameter policy;
- input mapping/transform identities, preserved in prediction exports.

`generation_targets` is canonical at the top level; `diagnostics.generation_targets` may be an alias of the same record, never a second independently generated copy. A legacy OD-only adapter is allowed only when consumers do not require the omitted fields.

## Transport knowledge

| Object | Contract |
| --- | --- |
| Section | Directed `(source_station,destination_station,line_id)` identity; physical `(source,destination)` projection is explicit |
| Feasible paths | Ordered real-OD path alternatives, per-time availability mask and unique path IDs |
| Incidence | Exact sparse mapping from feasible path IDs to directed sections; padded/infeasible paths have zero incidence |
| CP factors | Optional approximate representation; cannot silently substitute for exact incidence |
| Path probabilities | Sum to one within each nonempty feasible real-OD set; masked alternatives have zero mass |
| `K7` frequency | `[N,N,L_lines]`; this reconstruction standardizes on **trains/hour**, which is an explicit unit convention rather than a paper-recovered storage unit |
| No service | Known raw frequency zero; distinct from missing timetable information |
| Impedance `C_t` | `[N,N]`, time-specific admissible generalized cost with declared units; static reuse only when its manifest says static |
| Gravity marginals | Separate original-scale production/attraction vectors with equal total demand; any reconciliation rule is recorded |
| Gravity output | Dynamic IPF solution with frozen calibrated positive zeta and per-time balancing factors |

Headway in minutes converts to frequency using `60/headway`; seconds use `3600/headway`. Interval count is frequency times `interval_minutes/60`. Path feasibility uses raw physical service support, never normalized feature values, which can be negative or shifted.

`ServiceRecord` retains `raw_frequency`, `encoder_frequency`, `structural_support` and `known_status` separately. It carries `raw_units=trains_per_hour`, `encoder_scale`, `frequency_transform_id`, `line_order_id` and available-at metadata. Only the encoder view is normalized, once, using a training-fitted transform compatible with encoder pretraining. An unknown value cannot become observed zero by scaling; disabling the service encoder does not remove raw service needed by path feasibility.

The canonical assignment record uses `path_probabilities[(o,d),path_id]`, `feasible_sets[(o,d)]`, `incidence[(o,d),path_id]` and `line_aware_routes[(o,d),path_id]`. The probability mapping uses a composite key containing the real OD tuple and the stable path ID. To obtain one OD's probability vector, gather these values in the exact order of `feasible_sets[(o,d)]`; a nested lookup by OD alone is not the storage contract. Incidence is a set of distinct directed physical sections for the paper's binary section indicator; a route is an ordered, line-aware sequence. The operational load procedure uses the ordered route to locate contributions in travel time and the physical projection to identify an observed section. These representations cannot silently replace each other.

Component dependencies are conditional. Generation guidance or distribution features require the independent generation artifacts; service features or path feasibility require raw service information; a structured encoder requires only its own pretraining artifacts. Generic exogenous inputs may require raw demographic, calendar or service features even when the structured generation and path components are disabled. Missing dependencies are reported before constructing that variant, rather than represented by fabricated zero features.

For Case 3b, `build_generic_exogenous_adapter` returns an input schema, encoder specification and factory. The producer prepares `generic_exogenous_inputs`; `Net` calls `build_generic_encoder` once and registers the resulting `ConfiguredMLPEncoder`. Its dense layers, activations and dropout produce `knowledge.generic_exogenous_by_origin[N,F_mlp]`. The configured feature order, station alignment, availability and frozen input transform accompany the inputs. See [implementation parameters](implementation_parameters.md) for the layer specification.

Distribution features read `dynamic_impedance` from the selected encoder or decoder knowledge step. The producer takes an explicit `impedance_store`; loaders and collators preserve each matrix and its metadata while slicing only the outer sample and main-model time axes.

## Objective and evaluation

`N_train` counts predicted matrices with a declared overlap/multiplicity convention. Eq. (18) is full original-scale OD MAE; Eq. (16) is a sum of station residual squares. A minibatch estimator must state both scalings. The training specification gives explicit alternatives and marks them as reconstruction choices.

Residual destination flow contributes to station attractions using origin-specific training shares. The differentiable scatter/sum remains attached to model predictions. Frozen auxiliary targets may be detached; the generation residual loss may not be placed inside a no-gradient block.

Inference is autoregressive without future OD labels. A declared output nonnegativity policy applies consistently to evaluation, exports and path assignment. Per-horizon MAE, RMSE and WMAPE are computed before arithmetic averaging over horizons. The compatibility metric name `MAPE_net_od` denotes a ratio; `WMAPE_percent_od` denotes percent.

`EvaluationResult` includes `sample_count`, per-horizon metrics, aggregates, ordered sample metadata, output-policy/transform IDs and, when requested, the same prediction/truth/mask arrays used to compute those metrics. Missing or duplicated samples are errors, not an instruction to truncate to the smaller array length.

## Parameter ownership

Main-network, enabled utility and pretrained-then-fine-tuned encoder parameters are registered once and saved in the selected checkpoint. Gravity zeta, station mappings, residual shares and fitted scalers are fixed artifacts. Independent generation-model behavior during main training requires an explicit freeze/finetune declaration; all evaluation parameters are frozen.

A model forward pass cannot construct fresh trainable encoders, reload a checkpoint for every sample or fit new transforms. Checkpoint restore validates architecture, tensor keys, identities and availability manifests before evaluating.

## Added experiment interfaces

[Ablation and sensitivity](../experiments/ablation_and_sensitivity.pseudo) describes case flags, compatible learner adapters, external baseline requirements, validation-only comparisons, timing and task changes. The original snapshot did not contain all ten benchmark implementations or a complete GConvLSTM forecasting adapter; those dependencies remain explicit requirements.

[Operational load assessment](../experiments/operational_load_assessment.pseudo) consumes original-scale OD and the same forward pass's assignment context. Its `ForecastView` includes prediction validity, sample IDs, target times, forecast origins and a transform receipt. Prediction validity is independent of target-observation masks. An entry interval begins at its target time and has the configured duration.

Path-to-section contributions preserve line, direction, hops and entry interval. A trajectory selects one forecast per entry cohort. Gaussian interval weights map those contributions to receiving intervals; a mass ledger records received, discarded and outside-grid mass. Required entry support determines receiving coverage. Train-level rows retain their source forecast IDs and are matched by section, line, direction, date, train and time. [Implementation parameters](implementation_parameters.md) collects calibration, tail and matching choices.
