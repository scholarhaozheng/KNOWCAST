# Training and evaluation

This chapter follows Sections 4.3–4.6 (pp. 13–18), 5.1–5.2 (pp. 20–24) and 5.5–5.6 (pp. 32–33) of the [paper](../docs/assets/knowcast-paper.pdf).

## Data and task settings

The paper's City S network has 154 stations, 17 transfer stations and 15-minute intervals. Raw OD matrices contain 154 × 154 entries, including the diagonal. The main comparison uses a 0.23 matrix-dimension retention ratio. Each origin retains its own destination map; the residual column preserves flow to all other destinations.

Training covers March 1–31, 2023; validation covers May 1–10; testing covers May 11–20. April separates training and evaluation. Configuration determines which available historical observations supply context for May forecasts and how windows cross split boundaries.

Supply lookback and forecast lengths, batch size, training budget, seed, optimizer settings and preprocessing in configuration. The settings file uses `null` for values the paper leaves unspecified. Section 5.6 reports sensitivity windows `(lookback=2, forecast=4)` and `(lookback=4, forecast=12)`. Set the integer retained count `R` separately from the reported dimension ratio.

## Preparation and pretraining

Pretrain the generation estimators and feature encoders, and calibrate gravity before fitting the main predictor. Use training data for these stages and for destination maps, residual shares and transforms. Validation selects configurations and checkpoints. Evaluation freezes fitted parameters while computing time-specific knowledge and IPF from available inputs.

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

Configuration supplies auxiliary architectures, pretraining schedules, calibration parameters and artifact locations. Each searched latent dimension selects or trains a compatible encoder checkpoint.

## Loss and one training epoch

Let `I_hat[b, h, m, k]` and `I_true[b, h, m, k]` be predicted and observed compressed OD flows in **original passenger-flow units**. Here `b` is an example, `h` a forecast step, `m` a real origin, and `k` an origin-specific retained slot or the residual slot. `destination(m,k)` maps a retained slot to a real destination. Its value can differ between origins at the same slot index.

Eq. (15) constructs destination totals for attraction loss. Eq. (16) sums squared station residuals; Eq. (18) averages absolute OD errors. The minibatch estimator specifies sampling, multiplicity, reduction, accumulation and relative scaling.

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

With `alpha_decay=0.1`, formula indices `epsilon=0`, `9` and `10` give weights `w0`, `0.1*w0` and zero. The weight stays zero for `epsilon>=10`. Configuration maps displayed epochs to `epsilon`. The paper reports the rounded selected value `w0=0.0264`.

Attractions gather retained columns by real destination and distribute the residual using training shares. Prediction-side totals retain gradients. Configuration selects whether the independently pretrained generation estimator stays frozen or is fine-tuned.

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

Report intermediate validation scores to the pruner and stop trials when requested. Configuration defines checkpoint tie-breaking, sampler/pruner settings and random seeds.

Validation MAE controls hyperparameter selection, early stopping and checkpoint selection. Test curves are post-hoc diagnostics (p. 24).

## OD evaluation

Evaluation feeds preceding predictions into the decoder. Future exogenous inputs are plans, announced conditions or forecasts available at the prediction origin (Section 3.2, p. 8).

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

Compute metrics for each horizon before averaging. Configuration supplies zero-denominator, missing-value and output-clipping policies. For demand segments, Eq. (37) defines `SMAPE = 100 * mean(abs(prediction - observation) / ((abs(observation) + abs(prediction))/2 + 1e-6))`.

Table 3 reports full-model MAE 1.02, RMSE 3.04 and WMAPE 54.25%, with ten baselines. Section 5.3 reports ablations and an alternative learner. Compression-ratio experiments change the prediction target and are evaluated as task-setting sensitivity.

## Indirect path and section-load assessment

Section 5.5 (pp. 32–33) assesses path-derived section flows indirectly through sampled train loads. Assignment covers retained real OD pairs; residual demand enters through the subsequent compensation step.

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

## Numerical checks

The [validation scripts](../validation/scripts/README.md) provide small examples for the following calculations:

- Flow conservation through compression, unfinished allocation and residual redistribution.
- Squared generation loss, original-scale OD MAE and their prediction gradients.
- Minibatch scaling, accumulation and formula weights at `epsilon=0`, `9` and `10`.
- Per-horizon metrics, conditional path probabilities and retained-OD assignment.
- Temporal-kernel mass accounting and train-load matching by section, line, direction and time.
