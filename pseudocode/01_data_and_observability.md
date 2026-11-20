# Data, observability, and selective aggregation

Sections 4.2.1–4.2.4 and 4.3.2 of the [paper](../docs/assets/knowcast-paper.pdf), pp. 11–14, define cutoff observations, selective aggregation, unfinished allocation and residual shares in Eqs. (2)–(9) and (15).

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

For one forecast sample, finished inputs have shape `[L_h, N, K]`, unfinished counts have shape `[L_h, N, 1]`, and each allocated unfinished sequence has shape `[L_h, N, K]`. Future targets have shape `[L_f, N, K]`. Batching prepends the batch axis to each tensor.

## 1. Keep observation time separate from label resolution

An AFC journey supplies its origin and tap-in time at entry. Its destination and tap-out time become observable when the journey ends. Build every input from information available at the forecast cutoff.

For every lookback interval, use the same sample-specific cutoff `t0`. A journey in that interval may be unfinished for one forecast and finished for a later forecast.

```text
PROCEDURE: BUILD AN OBSERVABLE LOOKBACK AT t0

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
      Store the unfinished journey as an origin-level count.

  Compress F_raw using the fixed RDS mapping.
  Keep U_raw as an origin-level vector.

SEPARATE OFFLINE LABEL PROCEDURE:
  Resolve complete journeys by entry interval and eventual destination.
  Construct A_raw and compress it using the same fixed RDS mapping.
  Store future A matrices as supervised targets.

OUTPUT:
  Observable finished and unfinished lookback tensors
  Separately stored actual-demand labels
```

For each origin and historical interval, Equation (2) requires

$$
\sum_n A^{\mathrm{raw}}_{mn}
= \sum_n F^{\mathrm{raw}}_{mn}(t_0) + U^{\mathrm{raw}}_m(t_0).
$$

The identity uses the same entry interval and journey population on both sides. Finished and unfinished journeys form disjoint groups. Keep observed zero rows separate from missing records.

**Configuration:** timestamp precision, interval and service-day boundaries, duplicate and unmatched-record handling, delayed tap-outs and label-finalization rules.

## 2. Fit one origin-specific destination mapping on training data

Sections 4.2.2 and Eqs. (3)–(4) retain destination identity separately for each origin: slot `k` maps through `RDS[m,k]`.

```text
PROCEDURE: FIT AND FREEZE THE DESTINATION MAPPING

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
```

The transformation for either `X = A_raw` or `X = F_raw(t0)` is:

```text
PROCEDURE: COMPRESS A RAW OD MATRIX X

FOR EACH origin m:
  FOR retained slot k = 1,...,R:
    X_compressed[m,k] <- X[m, RDS[m,k]]
  X_compressed[m,K] <- sum of X[m,n] over residual_set[m]

RETURN X_compressed with shape [N,K]
```

Compression preserves each origin total. The dimension-retention ratio measures matrix size; passenger counts in nonretained destinations remain in the residual slot. Path assignment uses the retained real destinations.

**Configuration:** tie-breaking in training-period destination ranks and the integer retained count `R`.

## 3. Store the fixed residual allocation used for destination totals

Equation (15) distributes a predicted residual total over nonretained destinations using training-period shares, supplying station attractions for the consistency loss.

```text
PROCEDURE: FIT RESIDUAL DESTINATION SHARES

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

Origin totals and estimated destination totals sum to the same predicted flow. Destination totals gather retained slots by station ID and add residual shares.

## 4. Build day, week, and month references at the correct timestamps

Equation (5) organizes offsets `{0, 1, 7, mo}`. Equations (7)–(9) allocate unfinished trips from the previous day, the same weekday in the previous week and the same calendar day in the previous month. These references are completed historical OD available at the forecast cutoff.

```text
PROCEDURE: ALLOCATE UNFINISHED TRIPS

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

For each origin, timestamp and reference, destination shares sum to one and the allocated row sums to the unfinished count. The three allocations describe alternative destination distributions; the model fuses their recurrent representations.

**Configuration:** missing calendar dates, missing intervals and history across split boundaries. Observed zero rows use uniform shares; missing rows follow the configured reference policy. Every selected reference carries its timestamp and availability.

## Invariants

- `A_raw`, `F_raw`, and `U_raw` refer to the same entry interval and journey population; only `F_raw` and `U_raw` depend on `t0`.
- Every compressed slot retains its origin-specific meaning across all dataset splits.
- Compression and the residual-share operator preserve total flow.
- Every known-zero reference row yields uniform shares, including the residual slot.
- Every historical reference and model input is available at its forecast cutoff; future actual demand remains a label.
- Day, week, and month sources remain distinct and chronologically aligned throughout batching.

