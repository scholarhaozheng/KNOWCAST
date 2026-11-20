# Implementation parameters

The algorithms accept the choices below through configuration. [Paper settings](paper_settings.yaml) records values and ranges stated in the article; this table names the remaining inputs without repeating them in each procedure.

| Area | Parameters / artifacts | Consumer |
| --- | --- | --- |
| Samples | Split dates, missing-reference rule, calendar-month rule, target multiplicity and observation policy | Pipeline, loader, training |
| Generation | Covariate profile, auxiliary window/offset, model/scaler identities, main-training parameter policy | Generation estimators, Net |
| Gravity | Impedance matrices with units/time/availability, fitted zeta, IPF tolerance and iteration limit | Knowledge producer, Net |
| Autoencoders | Layer widths, activations, normalization, pretraining schedule and checkpoint | Encoder preparation |
| Feature fusion | Attention head count, projection widths, readout and initialization | GatingLayer |
| Case 3b | Ordered raw feature schema, station alignment, missingness rule, frozen input transform, MLP widths/activations/dropout | ConfiguredMLPEncoder |
| Training | Formula epoch origin, batch estimator, sampler, accumulation, checkpoint tie rule and seeds | Training entry point |
| Baselines | Forecast adapter, architecture, readout, preprocessing and search space | Controlled comparison |
| Runtime comparison | Warm-up, timing boundaries, five measured epochs and inference aggregation | Runtime benchmark |
| Sensitivity | Actual retained count R, window lengths, tuning policy and evaluation population | Sensitivity procedures |
| Operational assignment | Prediction mask, negative-flow policy, spatial coverage rule and forecast-record selection | Assignment and maps |
| Operational calibration | Carriage expansion and residual compensation by section/line/direction/time | Load assessment |
| Temporal dispersion | Delay/spread versus hops, time unit, entry timing, Gaussian tail policy and required entry support | Interval kernel |
| Train loads | Frequency series, train matching/interpolation, horizon selection and observation tolerance | Load conversion and matching |

## Configured encoders

Case 3b uses `encoder_spec={input_dim,widths,activations,dropout,feature_schema,input_transform,output_dim}`. The last width equals `output_dim`; each dense layer has one activation and dropout setting. `Net` constructs and registers one encoder. Its forward path applies the transform, then the dense/activation/dropout sequence.

## Temporal dispersion

For each path-to-section contribution, the kernel maps hop distance to a Gaussian delay mean and spread. Receiving weights integrate that distribution over the receiving interval, with optional quadrature over the entry interval. The mass ledger records received mass, discarded negative-delay mass and mass outside the receiving grid.

Select `full_gaussian`, `causal_discard` or `causal_renormalized` explicitly. A zero spread gives a deterministic delay. Required entry support and its tail tolerance determine coverage at grid boundaries; a missing earlier cohort is recorded as incomplete coverage.

With uniform arrivals, section passengers over an interval divided by the number of trains gives passengers per train. Under constant headway `h` and interval duration `Delta`, this is `volume*h/Delta`. Calibrated alternatives supply their conversion function and parameters.
