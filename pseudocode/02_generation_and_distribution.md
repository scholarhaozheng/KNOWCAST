# Trip generation, progressive loss, and gravity features

Sections 4.3–4.4 of the [paper](../docs/assets/knowcast-paper.pdf), pp. 13–16, define generation guidance, progressive loss and gravity features in Eqs. (14)–(20). Section 5.1.2, p. 21, describes the feature sources. Symbols `N`, `R`, `K=R+1`, `RDS` and `w_res` follow the data chapter.

## Inputs and outputs

The paper uses `A` both for actual OD data and for attraction in different contexts. Here, `Y` denotes actual compressed OD targets and `A_est` denotes estimated station attraction, to keep those quantities distinct.

| Object | Shape | Meaning |
| --- | --- | --- |
| `K_TG[t]` | `[N,F_TG]` | Station demographic and temporal covariates |
| `G_TAZ` | `N` nodes with directed weighted edges | Station-centered spatial graph |
| `P_est[t]`, `A_est[t]` | `[N]` each | Independently estimated production and attraction |
| `Y_hat`, `Y` | `[Q,N,K]` | Predicted and actual OD matrices for `Q` forecast targets |
| `c[t]` | `[N,N]` | Generalized origin-destination impedance |
| `B_origin[t]`, `B_destination[t]` | `[N]` each | Time-specific gravity balancing coefficients |
| `M_TD[t]` | `[N,N]` | Doubly constrained gravity distribution |
| `M_compressed[t]` | `[N,K]` | Gravity matrix after the fixed OD compression |
| `Z_TD[t]` | A declared latent shape | Autoencoder output supplied to the predictor |

The feature schema determines `F_TG`, categorical encodings and station alignment. Encoder configuration supplies the latent layout, hidden widths and batching axes.

## 1. Construct the generation covariates and pretrain the estimator

Equation (14) specifies population activity, working-age share, four educational-attainment shares, average household size, weekend/holiday status, time of day, season and event intensity.

```text
PROCEDURE: PREPARE TRIP-GENERATION INPUTS

FOR EACH station m:
  Estimate catchment population from population raster data.
  Use the 800 m station catchment described in Section 5.1.2.
  Combine district census and raster information to estimate:
    working-age share,
    education shares: primary, junior middle, senior high, tertiary,
    average household size.

FOR EACH station m and timestamp t:
  Derive weekend/public-holiday status, time of day in hours, and season.
  Construct the event-intensity input from available event information.
  Assemble K_TG[m,t] using the variables in Equation (14).
  Preserve the time at which each input became available.

PROCEDURE: PRETRAIN THE GENERATION ESTIMATOR

INPUT:
  Station covariate sequences, the weighted directed graph G_TAZ,
  and aligned station production and attraction training targets

Use the DCRNN within the Seq2Seq formulation described in Section 4.3.2:
  diffusion convolution represents spatial influence;
  recurrent updates represent temporal evolution.
Pretrain to estimate station production and attraction.
Retain the fitted estimator and the feature-processing specification.

AT A FORECAST CUTOFF:
  Use only observed, announced, or forecastable covariates available then.
  Produce P_est and A_est for the same times as the OD predictions.
```

Event intensity lies in `[0,1]` and uses event attendance and timing near the station. Inputs come from official releases and calendars. Configuration supplies event weights, categorical encodings, spatial disaggregation, normalization, DCRNN architecture and pretraining settings. The estimator interface returns separate production and attraction vectors.

## 2. Compare station totals in original flow units

Use the fixed retained-destination mapping and training-period residual shares from Equation (15). For each predicted matrix, compute origin totals by summing all `K` slots, and approximate actual destination totals by their station identifiers.

```text
PROCEDURE: GENERATION CONSISTENCY FOR ONE TARGET TIME

INPUT:
  Y_hat[N,K], P_est[N], A_est[N], frozen RDS and w_res
  All quantities expressed in the original passenger-flow scale

FOR EACH origin m:
  P_from_OD[m] <- sum over all slots k of Y_hat[m,k]

Initialize A_from_OD[N] to zero.
FOR EACH origin m:
  FOR EACH retained slot k = 1,...,R:
    destination n <- RDS[m,k]
    Add Y_hat[m,k] to A_from_OD[n].
  FOR EACH non-retained destination n:
    Add w_res[m,n] * Y_hat[m,K] to A_from_OD[n].

generation_error_at_t <-
    sum over m of (P_est[m] - P_from_OD[m]) squared
  + sum over n of (A_est[n] - A_from_OD[n]) squared

Preserve the differentiable dependence on Y_hat throughout this procedure.
```

Equation (16) is a sum of squared station discrepancies over all included target times:

$$
\mathcal L_g
= \sum_{q=1}^{Q}\left[
\sum_{m=1}^{N}\left(P^{\mathrm{est}}_{qm}-\sum_{k=1}^{K}\widehat Y_{qmk}\right)^2
+\sum_{n=1}^{N}\left(A^{\mathrm{est}}_{qn}-\operatorname{DestTotal}_n(\widehat Y_q)\right)^2
\right].
$$

Compare production and attraction vectors elementwise after aligning their `[N]` station axes. Attraction totals gather compressed slots by their real destination IDs.

## 3. Apply the progressive training objective

Equation (18) specifies mean absolute error in the original flow-value scale. If the model uses normalized targets internally, its inverse transformation must preserve gradients from this loss to its predictions.

$$
\mathcal L_{\mathrm{OD}}
=\frac{1}{QNK}\sum_{q=1}^{Q}\sum_{m=1}^{N}\sum_{k=1}^{K}
\left|\widehat Y_{qmk}-Y_{qmk}\right|.
$$

Here `Q` counts target OD matrices and equals the paper's `N_train`. With `Q_windows` forecast windows and all `lf` horizons included, `Q = Q_windows × lf`. The [shared interfaces](../specs/interface_contracts.md) use `Q` for windows and `N_train` for matrices; the overlap policy defines multiplicity.

$$
w_g(\varepsilon)=w_g(0)\max(1-\alpha_{\mathrm{decay}}\varepsilon,0),
\qquad
\mathcal L_{\mathrm{total}}^{(\varepsilon)}
=\mathcal L_{\mathrm{OD}}+w_g(\varepsilon)\mathcal L_g.
$$

```text
PROCEDURE: MAIN TRAINING OBJECTIVE

Obtain the OD forecast Y_hat from the current predictor.
Express Y_hat, actual targets, and estimated station totals in flow units.
Compute original-scale OD MAE.
Compute the sum of squared production and attraction discrepancies.
Compute the epoch-dependent weight using Equation (17).
Add the weighted generation term to OD MAE.
Backpropagate through the OD prediction and all jointly trained components
on the prediction graph, including the pretrained distribution encoder.
Update the declared main-model parameters.
```

**Gradient flow:** the auxiliary loss reaches the OD predictor through predicted station totals. `RDS`, residual shares and offline gravity coefficient `zeta` remain fixed. The distribution encoder is fine-tuned by prediction loss. The generation estimator's configuration selects frozen or fine-tuned parameters; prediction-side operations retain gradients in either case.

**Configuration:** formula epoch origin, minibatch estimator, reduction and gradient accumulation. Eq. (16) sums squared station residuals; Eq. (18) averages absolute OD errors. Their relative scaling is part of the objective.

## 4. Construct a doubly constrained gravity distribution at each time

Sections 4.4.1–4.4.2 require nonnegative production and attraction vectors with equal total volume. Individual station values can differ. Generalized impedance reflects travel time, monetary cost and service conditions available at the forecast cutoff.

The deterrence coefficient `zeta > 0` is calibrated offline from historical data and held fixed in main-model training. Equation (20) uses an exponential deterrence function:

$$
f_{mn,t}=\exp(-\zeta c_{mn,t}),\qquad
M^{\mathrm{TD}}_{mn,t}
= B^O_{m,t}P_{m,t}B^D_{n,t}A_{n,t}f_{mn,t}.
$$

The following IPF updates apply to positive, feasible margins. Configuration supplies stopping tolerance and handling of zero or infeasible margins.

```text
PROCEDURE: GRAVITY FEATURE AT TIME t

INPUT:
  P[N], A[N], c[N,N], and the fixed positive coefficient zeta

REQUIRE:
  P and A are nonnegative and sum(P) = sum(A).
  Inputs and any forecasts used to construct them are available at t0.

f[m,n] <- exp(-zeta * c[m,n])
Initialize valid positive balancing coefficients.

REPEAT the IPF updates for this time t until the declared criterion is met:
  FOR EACH origin m:
    B_origin[m] <- 1 / sum over n of B_destination[n] * A[n] * f[m,n]
  FOR EACH destination n:
    B_destination[n] <- 1 / sum over m of B_origin[m] * P[m] * f[m,n]

FOR EACH origin m and destination n:
  M_TD[m,n] <- B_origin[m] * P[m] * B_destination[n] * A[n] * f[m,n]

CHECK, under the declared numerical tolerance:
  sum over n of M_TD[m,n] = P[m] for each m
  sum over m of M_TD[m,n] = A[n] for each n

RETURN M_TD[N,N].
Recompute the balancing coefficients when the time-dependent inputs change.
```

Origin coefficients multiply rows and attraction coefficients multiply columns. IPF recomputes balancing factors for each time-specific pair of marginals and impedance matrix.

**Configuration:** the offline calibration objective for `zeta`, marginal reconciliation, nonnegative output treatment, impedance composition, diagonal and structural-zero rules, IPF initialization, tolerance, iteration limit and failure handling.

## 5. Pretrain the distribution autoencoder, then fine-tune its encoder

Section 4.4.3 applies the same fixed selective aggregation to the gravity matrix before autoencoder compression. It describes two distinct training stages.

```text
PROCEDURE: DISTRIBUTION REPRESENTATION

FOR EACH training timestamp:
  Construct M_TD using the time-specific gravity procedure.
  M_compressed[N,K] <- apply the fixed RDS and residual aggregation.

STAGE A: INDEPENDENT RECONSTRUCTION PRETRAINING
  Encode M_compressed into a declared latent representation.
  Decode that representation back to shape [N,K].
  Optimize encoder and decoder parameters to minimize reconstruction MSE.
  Retain the fitted encoder parameters and input-processing specification.

STAGE B: MAIN FORECASTING TRAINING
  Initialize the integrated distribution encoder from Stage A.
  Produce Z_TD from the aligned M_compressed at each required timestamp.
  Supply Z_TD as an augmented feature of the main predictor.
  Fine-tune the encoder jointly with the predictor from prediction loss.
  Keep the offline deterrence coefficient zeta fixed.

INFERENCE
  Use the trained encoder and forecasting model without parameter updates.
  Construct each required gravity input from information available at t0.
```

Pretraining minimizes reconstruction MSE; main training fine-tunes the encoder through forecasting loss. Layer layouts, pretraining schedules and optimizer settings come from configuration. Fit calibration, transforms and autoencoders on training data.

## Invariants

- Generation covariates represent the variables in Equation (14), with declared encodings and availability times.
- Production and attraction targets use the correct origin and destination indices.
- The attraction loss uses Equation (15), including fixed residual shares and their uniform zero-total fallback.
- Equation (16) retains its squared-error form, Equation (18) its original-scale MAE, and Equation (17) its nonnegative linear decay.
- The loss graph reaches the OD predictor and jointly trained distribution encoder.
- Gravity uses exponential impedance, a fixed offline `zeta`, and time-specific IPF balancing.
- The distribution encoder starts from reconstruction pretraining and is fine-tuned for forecasting.
- Configuration records model dimensions, calibration, preprocessing and objective scaling.

