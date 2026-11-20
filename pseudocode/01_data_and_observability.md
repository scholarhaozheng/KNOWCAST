# Data, observability, and selective aggregation

The source is Sections 4.2.1-4.2.4, Equations (2)-(9), and the residual-allocation rule in Section 4.3.2, Equation (15), of the [published paper](https://doi.org/10.1016/j.trc.2026.105710), PDF pages 11-14. Symbols below make the time and tensor dimensions explicit without prescribing a software implementation.

## Inputs and outputs

Let `N = |STN|`, `R = |RDS|`, and `K = R + 1`. There are `R` explicitly retained destinations per origin and one residual slot. This specification covers the selective-aggregation setting `R < N`. Let `L_h` and `L_f` be the lookback and forecast lengths.

| Object | Shape for one interval | Meaning |
| --- | --- | --- |
| `A_raw` | `[N, N]` | Actual OD demand, resolved from completed AFC records |
| `F_raw(t0)` | `[N, N]` | Journeys completed by forecast cutoff `t0` |
| `U_raw(t0)` | `[N, 1]` | Entries whose journeys remain unfinished at `t0` |
| `RDS` | `[N, R]` station identifiers | Ordered, origin-specific retained destinations |
| `A`, `F(t0)` | `[N, K]` | Compressed actual and finished matrices |
| `w_res` | `[N, N]` | Fixed training-period shares for non-retained destinations |
| `D_delta`, `U_alloc_delta` | `[N, K]` each | Historical destination shares and allocated unfinished counts |

For one forecast sample, finished inputs have shape `[L_h, N, K]`, unfinished counts have shape `[L_h, N, 1]`, and each of the three allocated unfinished sequences has shape `[L_h, N, K]`. Actual future targets have shape `[L_f, N, K]`. A batching implementation may prepend a batch dimension; this does not change any station or destination meaning.

## 1. Keep observation time separate from label resolution

An AFC journey supplies its origin and tap-in time when the passenger enters. Its destination and tap-out time become observations only when the journey ends. An offline record containing the eventual destination must not make that destination available earlier in a simulated forecast.

For every lookback interval, use the same sample-specific cutoff `t0`. A journey in that interval may be unfinished for one forecast and finished for a later forecast.

```text
CONCEPTUAL PROCEDURE: BUILD AN OBSERVABLE LOOKBACK AT t0

INPUT:
  AFC journey records, historical entry intervals, and forecast cutoff t0
  Training-period station order and fixed compression mapping

FOR EACH lookback interval tau, in chronological order:
  Initialize F_raw[tau; t0] and U_raw[tau; t0] to zero.

  FOR EACH journey entering in tau:
    Its origin and entry time are observable by t0.
    IF its tap-out has occurred at or before t0:
      Add one to F_raw[tau; t0, origin, observed destination].
    OTHERWISE:
      Add one to U_raw[tau; t0, origin].
      Do not expose the eventual destination to the model.

  Compress F_raw using the fixed RDS mapping.
  Keep U_raw as an origin-level vector.

SEPARATE OFFLINE LABEL PROCEDURE:
  Resolve complete journeys by entry interval and eventual destination.
  Construct A_raw and compress it using the same fixed RDS mapping.
  Supply future A matrices as supervised targets, not inference inputs.

OUTPUT:
  Observable finished and unfinished lookback tensors
  Separately stored actual-demand labels
```

For each origin and historical interval, Equation (2) requires

$$
\sum_n A^{\mathrm{raw}}_{mn}
= \sum_n F^{\mathrm{raw}}_{mn}(t_0) + U^{\mathrm{raw}}_m(t_0).
$$

This is a count-conservation check over the same journey population. In particular, an actual OD row cannot simultaneously be treated as a finished row and its entire row total as unfinished demand. A legitimate empty historical row and a missing or invalid record are different data conditions.

**Implementation decisions not supplied by the paper:** tap-time precision and interval boundary conventions; service-day boundaries; duplicate, unmatched, invalid, or delayed AFC records; and how an offline label set is finalized. A future implementation must document these decisions and reconcile the journey population before claiming the conservation identity holds.

## 2. Fit one origin-specific destination mapping on training data

Sections 4.2.2 and Equations (3)-(4) retain destination identity separately for each origin. Column `k` is not a common destination station across all rows.

```text
CONCEPTUAL PROCEDURE: FIT AND FREEZE THE DESTINATION MAPPING

INPUT:
  Actual raw OD matrices from the training period only
  Ordered station identifiers and requested retained count R

FOR EACH origin m:
  training_flow[m,n] <- sum of A_raw[m,n] over training intervals
  RDS[m] <- the R destinations with greatest training_flow[m,n], ordered
  residual_set[m] <- all remaining destination stations

STORE AND FREEZE:
  Station order, RDS, residual sets, and the training period used to fit them

USE:
  The identical mapping for every training, validation, and test interval
  Do not re-rank destinations on validation or test flows.
```

The transformation for either `X = A_raw` or `X = F_raw(t0)` is:

```text
CONCEPTUAL PROCEDURE: COMPRESS A RAW OD MATRIX X

FOR EACH origin m:
  FOR retained slot k = 1,...,R:
    X_compressed[m,k] <- X[m, RDS[m,k]]
  X_compressed[m,K] <- sum of X[m,n] over residual_set[m]

RETURN X_compressed with shape [N,K]
```

Each row total must be preserved. A dimension-retention ratio describes the compressed matrix size, not the fraction of passenger volume retained. The final slot has no unique destination, so it has no unique candidate-path set.

**Implementation decisions:** tie-breaking in training-period ranks and how an approximate reported retention ratio is translated into an integer `R`. The paper does not justify recovering either decision from an old filename or configuration.

## 3. Store the fixed residual allocation used for destination totals

Equation (15) uses training-period shares to approximately assign a predicted residual total back to non-retained destination stations for the attraction-consistency loss. This operation does not recover individual residual OD flows or authorize assigning the residual slot to one path.

```text
CONCEPTUAL PROCEDURE: FIT RESIDUAL DESTINATION SHARES

FOR EACH origin m:
  total_residual <- sum of training_flow[m,n] over residual_set[m]
  Set w_res[m,n] = 0 for retained destinations.

  FOR EACH n in residual_set[m]:
    IF total_residual > 0:
      w_res[m,n] <- training_flow[m,n] / total_residual
    OTHERWISE:
      w_res[m,n] <- 1 / (N - R)

FREEZE w_res together with RDS.
CHECK that its shares over residual_set[m] sum to one for every origin.
```

Given an OD prediction `Y_hat[N,K]`, the corresponding destination-total operator is

$$
\operatorname{DestTotal}_n(\widehat Y)
= \sum_{m,k:\,\mathrm{RDS}[m,k]=n}\widehat Y_{mk}
+ \sum_{m:\,n\notin\mathrm{RDS}_m}w^{\mathrm{res}}_{mn}\widehat Y_{mK}.
$$

Thus, origin totals and reconstructed destination totals sum to the same overall predicted flow. Directly summing compressed columns across origins does not implement this operator.

## 4. Build day, week, and month references at the correct timestamps

Equation (5) organizes the offsets `{0, 1, 7, mo}`. Equations (7)-(9) allocate unfinished trips using only `{1, 7, mo}`: the previous day, the same weekday in the previous week, and the same calendar day in the previous month. The current-day actual destination distribution is excluded because it is not fully observable at the forecast cutoff.

```text
CONCEPTUAL PROCEDURE: ALLOCATE UNFINISHED TRIPS

FOR EACH lookback timestamp tau, from oldest to newest:
  FOR EACH reference delta in {previous day, previous week, previous month}:
    reference_tau <- the corresponding calendar reference for tau
    H <- the actual OD matrix at reference_tau, compressed with fixed RDS
    REQUIRE H to be available as historical information by t0.

    FOR EACH origin m:
      IF sum over k of H[m,k] > 0:
        D_delta[m,k] <- H[m,k] / sum over slots of H[m,slot]
      OTHERWISE, when the observed historical row is genuinely zero:
        D_delta[m,k] <- 1 / K for every compressed slot k

      U_alloc_delta[m,k] <- U_raw[tau; t0,m] * D_delta[m,k]

RETURN three distinct sequences, each with shape [L_h,N,K].
PASS each sequence, in chronological order, to its corresponding
unfinished-trip GATRU branch in Equation (9).
```

For every origin, timestamp, and reference source, the shares sum to one and the allocated row sums to the current unfinished count. These are three alternative allocations of the same unfinished count; adding them as passenger counts would triple-count that demand. Their representations are fused by the model's gating mechanism.

**Implementation decisions:** an absent calendar day in the previous month, missing historical intervals, and history extending before a dataset split. The paper's uniform fallback covers an observed zero row; it does not specify a missing-history policy. A future implementation must declare that policy, preserve timestamp alignment, and never substitute a later observation or wrap a negative index to the end of a split. Available operational history and the periods used to fit model parameters must be recorded separately.

## Method-level checks

- `A_raw`, `F_raw`, and `U_raw` refer to the same entry interval and journey population; only `F_raw` and `U_raw` depend on `t0`.
- Every compressed slot retains its origin-specific meaning across all dataset splits.
- Compression and the residual-share operator preserve total flow.
- Every known-zero reference row yields uniform shares, including the residual slot.
- Every historical reference and model input is available at its forecast cutoff; future actual demand remains a label.
- Day, week, and month sources remain distinct and chronologically aligned throughout batching.

