# Training and evaluation specification

Source: [the 37-page paper](../docs/assets/knowcast-paper.pdf), especially Sections 4.3-4.6 (pp. 13-18), 5.1-5.2 (pp. 20-24), and 5.5-5.6 (pp. 32-33). Page numbers below are one-based PDF page numbers.

## Inputs and unresolved choices

The reported City S network has 154 stations and 17 transfer stations, with 15-minute intervals. The raw OD matrix contains 154 x 154 entries, including diagonal entries. The main comparisons use a fixed 0.23 matrix-dimension retention ratio. Each origin has its own retained-destination mapping; one additional residual column preserves flow to all non-retained destinations. The ratio does not mean that only 23% of passengers are retained.

Training covers March 1-31, 2023; validation covers May 1-10; testing covers May 11-20. April is the reported temporal gap. A future implementation must explicitly resolve which historical reference observations are available for a May prediction and how windows crossing split boundaries are handled. The gap alone does not answer these questions.

The exact main-run lookback and forecast lengths, batch size, full training budget, seed, optimizer settings, and several preprocessing details are not unambiguously specified by the paper text. They remain `null` in the settings file. Section 5.6's examples `(lookback=2, forecast=4)` and `(lookback=4, forecast=12)` are sensitivity settings, not evidence of the main-run configuration. Likewise, do not recover an exact integer destination dimension by rounding `154 * 0.23`.

## Preparation and pretraining

The paper specifies independent trip-generation estimation, offline gravity calibration, and reconstruction pretraining followed by joint fine-tuning of the feature encoders. This reconstruction prescribes fitting all generation estimators, the gravity coefficient, autoencoders, scalers, destination mappings, and residual shares using the training split only. Validation and test use frozen parameters and forward computations; validation may select a configuration or checkpoint but does not fit these components. Time-specific IPF and feature computation may still use admissible inputs during evaluation without updating fitted parameters. These are requirements for a future implementation, not evidence that a particular historical run passed them.

```text
REQUIRE resolved task dimensions, data manifests, and preprocessing policies
PARTITION target examples by the reported training / validation / test dates

FIT each origin's retained-destination mapping using training-period OD totals
FREEZE that mapping for every split
COMPUTE residual destination-allocation shares using training-period OD totals
USE uniform shares over that origin's non-retained destinations if their total is zero
FIT any data-dependent normalization using training data only

PRETRAIN independent station production and attraction estimators (DCRNN)
  fit using training examples only
  use station-context and temporal covariates available at the prediction origin
  record the training data, architecture, normalization, and checkpoint

CALIBRATE the gravity deterrence coefficient zeta offline from training data only
FREEZE zeta during the main model's training
AT each applicable time step, balance production / attraction and solve IPF

INITIALIZE path-utility coefficients to the reported -0.30 and -0.573
RECORD the exact coefficients used to generate the training section-probability matrices
  and the path features, feasible-path masks, and preprocessing provenance

PRETRAIN reconstruction autoencoders separately for:
  compressed gravity-distribution matrices
  retained-OD section-probability matrices
  line-specific departure-frequency matrices
USE mean squared reconstruction error, as in Section 4.4.3
FIT these autoencoders using training-split matrices only
SAVE the pretrained encoder parameters

INITIALIZE the forecasting model with these pretrained encoders
REGISTER the encoders once as persistent trainable model parameters
USE the recorded utility initialization for the forecasting model
FINE-TUNE the feature encoders and full-model utility coefficients
  jointly with the forecasting model through the training-split forecasting objective
FOR every validation / test pass, freeze all parameters and use forward computation only
```

The paper does not supply all auxiliary-model architectures, pretraining schedules, calibration constants, or data artifacts needed to execute these steps. Changing the latent dimension during hyperparameter search also requires a compatible pretrained encoder; how these checkpoints are produced and isolated must be recorded rather than guessed.

## Loss and one training epoch

Let `I_hat[b, h, m, k]` and `I_true[b, h, m, k]` be predicted and observed compressed OD flows in **original passenger-flow units**. Here `b` is an example, `h` a forecast step, `m` a real origin, and `k` an origin-specific retained slot or the residual slot. `destination(m,k)` maps a retained slot to a real destination. Its value can differ between origins at the same slot index.

Equation (15) reconstructs destination totals for the attraction loss. Equation (16) defines a sum of squared station residuals, while Equation (18) defines the mean absolute OD error over the full training objective. A batch sum combined with a batch mean does not by itself specify an estimator of those full-data objectives. The sampling, reduction, accumulation, and relative scaling below are explicit implementation decisions and must be declared before training.

```text
DEFINE full L_OD according to Eq. (18), including its N_train denominator
DEFINE full L_generation according to Eq. (16), retaining its squared-error sum
REQUIRE a declared minibatch estimator for these full-data objectives
  record the sampling scheme, example multiplicities, scaling, and accumulation rules
REQUIRE a declared mapping from training-loop epochs to formula index epsilon

FOR each training epoch with its declared epsilon:
  auxiliary_weight = initial_weight * max(1 - alpha_decay * epsilon, 0)

  FOR each training batch:
    predict compressed future OD using the knowledge-integrated encoder-decoder
    convert predictions and targets to the original flow scale
    collect the unreduced absolute OD errors abs(I_hat - I_true)

    obtain station production P and attraction A for the matching target times
      from the pretrained generation estimators under a declared main-training policy
      if that policy freezes the estimators, their inference may disable gradient tracking

    KEEP gradient tracking enabled for every operation on I_hat below:
      production_hat[b,h,m] = sum over k of I_hat[b,h,m,k]

      attraction_hat[b,h,n] = 0 for every real destination n
      FOR every origin m and every retained slot k:
        n = destination(m,k)
        attraction_hat[b,h,n] += I_hat[b,h,m,k]
      FOR every origin m and every non-retained destination n:
        attraction_hat[b,h,n] += residual_share[m,n] * I_hat[b,h,m,residual]

      collect unreduced squared production residuals (P - production_hat)^2
      collect unreduced squared attraction residuals (A - attraction_hat)^2

      L_step = DECLARED_MINIBATCH_ESTIMATOR(
        full Eq. (18) L_OD + auxiliary_weight * full Eq. (16) L_generation,
        batch errors, declared sampling / scaling / accumulation rules)

    clear parameter gradients only at the start of the declared accumulation group
    backpropagate L_step according to the declared accumulation rule
    update registered parameters only when that accumulation group is complete
  finish any final partial group using its explicitly declared scaling rule
```

With the reported selected `alpha_decay=0.1`, evaluating the formula at `epsilon=0`, `9`, and `10` gives `w0`, `0.1*w0`, and zero; it remains zero for `epsilon>=10`. These are illustrative formula evaluations, not a recovered convention for numbering training-loop epochs. That convention remains unresolved. The selected `w0=0.0264` is a rounded reported value. This document prescribes no fixed minibatch rescaling. Replacing the squared-error sum with an absolute mean residual is not equivalent: positive and negative station errors can cancel.

The attraction term cannot sum compressed columns as if each column were the same destination for every origin. The residual slot is a mixture of destinations, not an additional station. The generation estimator's freezing policy during main training is not fully specified by the paper and must be declared. If frozen, this does not justify detaching the OD prediction or placing its loss calculation inside a gradient-disabled scope.

## Hyperparameter search and model selection

Table 2 and Section 5.1.4 (p. 22) specify an Optuna TPE search with SQLite storage and a MedianPruner: 80 trials, at most 40 training epochs per trial, minimizing validation MAE. Trial 78 is the reported best trial. The search domains and selected values are transcribed in the settings file; selected continuous values are marked `reported_rounded`.

```text
CREATE a study with TPE sampler, MedianPruner, and SQLite storage
REQUIRE explicit sampler/pruner parameters and randomization policy

FOR each of the 80 search trials:
  sample all eight Table 2 hyperparameters from their stated domains
  initialize a fresh compatible model and compatible pretrained encoders
  train for no more than 40 epochs
  after each scheduled validation evaluation:
    compute original-scale validation MAE using the horizon protocol below
    report the intermediate validation MAE and step to the trial
    query the pruner; stop the trial if it requests pruning
    retain the checkpoint with the lowest validation MAE for this trial
  return the trial's best validation MAE if the trial completes

SELECT the completed trial with the lowest validation objective
RECORD its exact configuration and study history
RUN full-length training with that configuration and the resolved training budget
SELECT the final checkpoint by validation MAE only
FREEZE the model, preprocessing, and evaluation rules before final test evaluation
```

Merely constructing a pruner does not implement pruning: intermediate values must reach it and its decision must stop the trial. The pseudocode's checkpoint and logging steps make a future implementation reviewable; they do not recover undocumented tie-breaking or original study internals. A new search is not expected to reproduce the original trial index or rounded optimum without the original study and environment.

No test score may control hyperparameter selection, early stopping, or checkpoint selection. The paper describes its plotted test curves as post-hoc diagnostics (p. 24). Any reconstruction of those curves must retain that separation.

## OD evaluation

During evaluation, the decoder uses its previous predictions, not future observed OD labels. Only exogenous information that was known, announced, or forecastable at the prediction origin can be supplied for future steps (Section 3.2, p. 8).

```text
FOR each forecast horizon h:
  collect the predictions and observations for all eligible test examples
  convert both to original flow units using the frozen transformation
  evaluate the same compressed OD target for all compared models

  MAE[h]   = mean(abs(prediction - observation))
  RMSE[h]  = sqrt(mean((prediction - observation)^2))
  WMAPE[h] = 100 * sum(abs(prediction - observation)) / sum(observation)

REPORT arithmetic mean over horizons separately for MAE, RMSE, and WMAPE
```

This order matters: pooling all horizons before taking the RMSE or WMAPE is generally different. If a WMAPE denominator is zero, the implementation needs a declared policy; none is recovered here. Missing-value handling and output clipping also require explicit decisions, not assumptions imported from historical code. For segmented demand analysis, Equation (37) defines `SMAPE = 100 * mean(abs(prediction - observation) / ((abs(observation) + abs(prediction))/2 + 1e-6))`.

Table 3 reports full-model MAE 1.02, RMSE 3.04, and WMAPE 54.25%, alongside ten baselines. The ablation and alternative-learner results in Section 5.3 are also **paper-reported results**, not outputs reproduced by this repository. A changed compression ratio changes the prediction target and must be presented as task-setting sensitivity, not a like-for-like improvement.

## Indirect path and section-load assessment

There are no observed path labels for direct path-flow validation. Section 5.5 (pp. 32-33) compares aligned estimated train loads with sampled observations. Path assignment operates only on explicitly retained real OD pairs; the residual slot has no unique destination or candidate-path set.

```text
FOR each retained real OD pair and forecast step:
  path_flow = predicted_OD_flow * conditional_path_probability
AGGREGATE path flows over the directed sections traversed by each path
EXCLUDE the residual destination slot from path and section assignment

OBSERVATION SIDE:
  count passengers in one designated carriage of a selected train
  expand that count to a train-level estimate using a factor
    calibrated for the same corridor and service period

MODEL SIDE:
  apply dynamic scale compensation for non-retained destinations
  disperse the compensated flow with a Gaussian temporal kernel
    whose delay and spread depend on topological hop distance
  convert section volumes to train loads using dynamic headway ratios

ALIGN modeled and observed train loads at the sampled locations and times
REPORT WMAPE and Pearson correlation with calibration assumptions and uncertainty
```

The observation protocol has 14 daily time slices over 10 days, including weekends; exact times are in the settings file. The [operational procedures](../experiments/operational_load_assessment.pseudo) expand this flow into contribution records, Gaussian interval weights, a mass ledger and train-load matching. Calibration factors and kernel choices are listed in [implementation parameters](../specs/implementation_parameters.md).

This indirect operational assessment retains calibration identities, separates calibration from evaluation observations, and reports the coverage and units of each matched estimate.

## Implementation checks

Use these checks when implementing the algorithms:

- Verify split membership, prediction-origin information availability, training-only fitting of every generation estimator, gravity coefficient, autoencoder, mapping, scaler, and residual share; verify frozen validation/test forward passes and explicit treatment of the April gap.
- Verify destination restoration with a hand example where the same compressed column represents different stations for different origins; verify conservation of origin and reconstructed destination totals.
- Verify squared generation loss numerically and verify a nonzero gradient from that loss to OD-predictor parameters when the residual is nonzero and its weight is active.
- Verify the declared minibatch estimator and its sampling/scaling/accumulation rules against the full objectives; verify formula values at `epsilon=0`, `9`, and `10` separately from the declared epoch-index convention.
- Verify recorded utility coefficients were available before generation of the section-probability pretraining matrices; verify persistent pretrained encoder parameters receive forecasting gradients and are included in optimization/checkpoints.
- Verify the trial reports reach the pruner, pruning can occur, all Table 2 domains are used, and neither model nor trial selection can access test scores.
- Verify metric formulas and per-horizon aggregation with hand examples, including declared handling of zero denominators, missing values, and negative predictions.
- Verify conditional path probabilities sum to one over feasible paths, retained-OD path flow is conserved, and no residual-column flow receives an invented route.
- Verify section-load alignment units, calibration provenance, temporal boundaries, and uncertainty before interpreting Table 8-like comparisons.
- Archive exact configs, software versions, seeds, data/split identifiers, pretrained and final checkpoints, predictions, and search records. Compare newly generated results with the reported values only after these artifacts exist; do not label the reported values as regenerated outputs.
