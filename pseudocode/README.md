# Algorithm reading guide

These five chapters explain the KNOWCAST method. Use the [repository guide](../PSEUDOCODE_GUIDE.md) and [46-module index](../validation/MODULES.md) to find the detailed `.pseudo` algorithms.

## Reading order

1. [Data and observability](01_data_and_observability.md): define what is known at the forecast origin, construct a fixed OD representation and allocate unfinished entries.
2. [Generation and distribution](02_generation_and_distribution.md): define station-level guidance, the progressive loss and gravity-based features.
3. [Path choice and service](03_path_choice_and_service.md): construct feasible path probabilities and service features, then infer path/section flows.
4. [KITP and forecasting](04_kitp_and_forecasting.md): combine these mechanisms in the recurrent encoder and decoder.
5. [Training and evaluation](05_training_and_evaluation.md): prepare auxiliary components, tune on validation data and evaluate held-out predictions.

## Notation and conventions

| Symbol or name | Meaning |
| --- | --- |
| `forecast_origin` / `t0` | The instant at which a forecast is issued; the information boundary for inputs |
| `N` / `STN` | Number / set of real stations |
| `m`, `n` | Origin and destination station identifiers |
| `R` / `\|RDS\|` | Number of explicitly retained destinations per origin |
| `K = R + 1` | Compressed OD width, including the residual slot |
| `RDS[m]` | Training-derived ordered real destination identifiers for origin `m` |
| `residual` | Last compressed slot, representing all other destinations for that origin |
| `l_h`, `l_f` | Historical window length and forecast horizon |
| `d`, `t` | Day and within-day interval in paper equations |
| `p`, `omega` | Feasible candidate path and directed network section |

Each chapter defines its tensor layout. `RDS[m]` maps compressed slots to real station IDs for each origin. Equations may omit the batch axis.

Chapter equations use one-based destination labels: retained `1..R`, residual `K=R+1`. Array interfaces use retained slots `0..R-1` and residual index `R`. Horizon label `lambda=1..lf` maps to array index `h=lambda-1`.

Names such as `GATRU`, `PRETRAIN_AUTOENCODER` and `AVAILABLE_AT` denote algorithm operations. Indentation shows control flow; assertions state input conditions and invariants.

## Related specifications

Method steps cite their sections or equations. [Implementation parameters](../specs/implementation_parameters.md) lists configurable choices.

Use the [equation crosswalk](../specs/paper_to_pseudocode.md), [reported settings](../specs/paper_settings.yaml) and [data and configuration](../REPRODUCIBILITY.md) alongside the algorithms.
