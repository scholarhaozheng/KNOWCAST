# Algorithm reading guide

These five chapters explain the KNOWCAST method. The full per-file algorithms live in the original source directories as standalone `.pseudo` files; use [the repository guide](../PSEUDOCODE_GUIDE.md) and [42-file coverage inventory](../audit/SOURCE_COVERAGE.md) to navigate them.

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

Each chapter declares its local tensor layout. A destination slot is not interchangeable with a real station identifier. Batch dimensions may be omitted in equations and restored by an implementation without changing the semantics.

Chapter equations and conceptual algorithms use one-based mathematical destination labels where shown: retained `1..R`, residual `K=R+1`. The detailed standalone software contracts use retained array indices `0..R-1` and residual index `R`. Horizon label `lambda=1..lf` similarly maps to array index `h=lambda-1`; these are explicit index conversions, not different target definitions.

Procedures such as `GATRU`, `PRETRAIN_AUTOENCODER` and `AVAILABLE_AT` denote operations or contracts, not imported functions. Indentation expresses control flow. Assertions specify validity conditions; they do not indicate that tests or experiments have already run.

## Related specifications

Paper-grounded steps cite sections or equations. Configurable choices are collected in [implementation parameters](../specs/implementation_parameters.md).

Use the [equation crosswalk](../specs/paper_to_pseudocode.md), [reported settings](../specs/paper_settings.yaml) and [reproduction requirements](../REPRODUCIBILITY.md) alongside the algorithms.
