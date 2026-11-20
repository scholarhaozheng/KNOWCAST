# Path choice, service features, and flow assignment

Source: [published paper](../docs/assets/knowcast-paper.pdf), Sections 4.5-4.7, Eqs. (21)-(30); autoencoder training also follows Section 4.4.3. The reported utility initialization is in Section 5.1.4. Section and equation numbers below refer to that paper.

## Inputs and indexing

- `t` denotes a target interval. For forecasting, every service input at `t` must have been known, announced, or forecastable at the prediction origin `t0` (Section 3.2).
- `V` is the station set, `Omega` the directed section set, and `Lines` a fixed ordered list of metro lines.
- `RDS[m, s]` maps retained compressed slot `s` of origin `m` to its real destination. There are `R` retained slots plus one residual slot. The map is fitted on training data and then fixed (Section 4.2.2).
- `P[od, t]` contains distinct feasible candidate paths for a real OD pair at interval `t`. The candidate-path generation procedure is an abstract primitive; this document does not assume a fixed path count.
- `u[p, omega]` is 1 when path `p` traverses directed section `omega`, and 0 otherwise. Reverse directions are distinct.
- `K7[t, eta, mu, line]` is departure frequency, not a headway duration. Invalid sections and sections not served by that line have zero frequency (Section 4.5.1).

This chapter uses one-based mathematical slot and horizon labels, as in the equations. The standalone `.pseudo` software contracts use zero-based arrays: retained slots `0..R-1`, residual `R`, horizon index `h=lambda-1`. Convert labels explicitly when crossing that boundary. Unknown service information is distinct from known zero service; the detailed modules preserve raw frequency, encoder frequency and known-status fields separately.

## Utilities and probabilities remain differentiable

The full model trains utility coefficients through the OD forecasting objective. The normalization itself has no trainable parameters. The fixed-utility ablation freezes the coefficients explicitly (Section 5.3), rather than changing the meaning of Softmax.

```text
INITIALIZE PATH-CHOICE PARAMETERS
    beta_stations <- -0.30 per station
    beta_transfer <- -0.573 per transfer
    record the feature definitions and units associated with these coefficients
    mark beta trainable for the full model
    if reproducing the fixed-utility ablation, explicitly freeze beta
    do not initialize additional, undocumented utility coefficients by assumption

PATH_CHOICE(t, service_snapshot, candidate_paths, section_features, beta)
    for each real retained OD pair od:
        P <- FEASIBLE_DISTINCT_PATHS(od, t, service_snapshot, candidate_paths)
        if P is empty:
            apply the declared fail or unassignable-demand policy
            record the empty set and continue to the next OD if the policy permits
            do not reach MAX or normalize an empty candidate set

        for each p in P:
            for each documented utility feature phi:
                psi[phi, p, t] <- SUM over sections omega in p
                                  of section_features[phi, t, od, p, omega]
                                  [Eq. (21)]
            U[p] <- SUM over phi of beta[phi] * psi[phi, p, t]
                    [Eq. (22)]

        shift <- MAX over p in P of U[p]
        denominator <- SUM over p in P of EXP(U[p] - shift)
        for each p in P:
            probability[od, p, t] <- EXP(U[p] - shift) / denominator
                                     [Eq. (23), algebraically equivalent form]
        for every excluded or padded path:
            probability[od, p, t] <- 0

        for each directed section omega:
            section_probability[od, omega, t] <-
                SUM over p in P of probability[od, p, t] * u[p, omega]
                [Eq. (24)]

    preserve the computation graph from beta through probabilities
    return probability, section_probability, the feasible sets and incidences
```

Subtracting the maximum utility is a numerical implementation choice that leaves Eq. (23) unchanged. Inputs to a forward pass are fixed observations or plans, while `beta` and downstream encoders remain trainable. Feasibility is checked before normalization: a service closure must not leave a partially traversable route masquerading as a valid origin-to-destination path.

## Two autoencoder families provide path and service features

Section 4.6 defines an OD-specific section-probability matrix and a line-specific frequency matrix. It does not replace the directed station-pair-by-line service tensor with a station-only headway table.

```text
CONSTRUCT SERVICE MATRICES(t, section_probability, K7)
    for each real retained OD pair od:
        for each ordered station pair (eta, mu):
            Mpl[od, eta, mu] <- section_probability[od, (eta, mu), t]
                                if (eta, mu) is a valid section
                                else 0
                                [Eq. (25)]

    for each line in the fixed order Lines:
        for each ordered station pair (eta, mu):
            Mdf[line, eta, mu] <- K7[t, eta, mu, line]
                                 [Eq. (26)]
            require zero for invalid or unserved directed sections

    return Mpl, Mdf

PRETRAIN PATH AND SERVICE AUTOENCODERS(training_matrix_corpora)
    declare separate path and service encoder-decoder families
    declare their dimensions, parameter-sharing rules and matrix vectorization
    record which utility parameter state generated the path training matrices

    for each family independently:
        optimize its encoder and reconstruction decoder to minimize
            MEAN_SQUARED_RECONSTRUCTION_ERROR(matrix, decoder(encoder(matrix)))
        use training-period matrices, with the preprocessing fitted on training data
        retain the learned encoder parameters for the forecasting model
        record the pretraining configuration and resulting parameter artifact
        [Section 4.6, using the two-stage procedure in Section 4.4.3]

ENCODE SERVICE KNOWLEDGE(t, Mpl, Mdf, pretrained_encoders)
    for each line in Lines:
        service_embedding[line] <- SERVICE_ENCODER(Mdf[line])
    shared_service <- CONCATENATE service_embedding[line] in the fixed line order
    compute this shared block once for t and reuse it across OD pairs

    for each real retained OD pair od:
        path_embedding[od] <- PATH_ENCODER(Mpl[od])
        F_KI[od, t] <- CONCATENATE(path_embedding[od], shared_service)
                       [Eq. (27)]

    return F_KI

JOINT FORECASTING TRAINING
    initialize encoders with the independently pretrained parameters
    construct probabilities and embeddings within the current forward pass
    update those encoders and trainable utility coefficients through the
        forecasting objective, together with the core predictor
    retain the separately pre-calibrated gravity deterrence parameter as fixed
    do not silently detach probabilities or freeze the feature encoders
    do not claim an additional reconstruction penalty during joint training
        unless a separate implementation decision explicitly introduces it
```

`F_KI` contains knowledge features only. The primary OD values are appended later when constructing station features (Section 4.8). The distribution embedding from Section 4.4 is a separate augmentation and is not silently substituted for either family above.

## Assign the forecast using probabilities for the same interval

For prediction origin `t0` and horizon index `lambda = 1, ..., l_f`, the target interval is `t = t0 + lambda - 1`. Use the same model parameter state and target-time service snapshot for its OD forecast and assignment probabilities. Historical path probabilities or a probability array from the first available timestamp are not interchangeable with these quantities.

```text
ASSIGN_FORECAST(t0, lambda, predicted_compressed_OD, probability,
                feasible_sets, incidence, RDS)
    t <- t0 + lambda - 1
    convert the OD prediction to the original passenger-flow scale exactly once
    apply the same declared output policy used for the corresponding OD evaluation/export
    require that probability, feasible_sets and incidence refer to this same forecast and t
    initialize LinkFlow[t0, lambda, omega] <- 0 for every directed section

    for each origin m:
        for each explicitly retained slot s from 1 through R:
            n <- RDS[m, s]
            od <- (m, n)
            demand <- predicted_compressed_OD[m, s]

            if feasible_sets[od, t] is empty:
                apply the declared fail or unassignable policy
                record unassigned demand and continue if permitted

            for each feasible p in feasible_sets[od, t]:
                PathFlow[t0, lambda, od, p] <-
                    demand * probability[od, p, t]
                    [Eq. (28)]
                for each distinct physical directed section omega in the path's incidence set:
                    LinkFlow[t0, lambda, omega] <-
                        LinkFlow[t0, lambda, omega]
                        + PathFlow[t0, lambda, od, p] * incidence[p, omega]
                        [Eq. (29)]

        exclude residual slot R + 1 from path and section assignment
        do not interpret this aggregate as a real destination

    return PathFlow, LinkFlow

AGGREGATE SECTION ESTIMATES(evaluation_prediction_origins)
    for each horizon lambda separately and each directed section omega:
        mean_section_flow[lambda, omega] <-
            MEAN over evaluation origins t0 of LinkFlow[t0, lambda, omega]
            [Eq. (30)]
```

These section flows cover explicitly retained OD demand. Section 5.5 describes additional compensation, temporal dispersion and headway conversion for comparison with sampled train loads; Eqs. (28)-(30) alone do not implement that operational validation. No path-level ground-truth labels are implied.

## Implementation decisions that remain explicit

| Detail | Paper-grounded constraint | Decision required for an executable implementation |
| --- | --- | --- |
| Candidate paths | Eqs. (21)-(24) normalize within the available paths for each OD pair | Path enumeration, truncation, deduplication, closure filtering and the empty-set policy |
| Station and transfer attributes | Section 5.1.4 supplies the two initial disutilities and their units | Exact endpoint-counting conventions and transfer detection; preserve those conventions in preprocessing |
| Additional service effects on utility | Sections 4.5.1-4.5.2 permit time-varying path characteristics | Specify any extra attributes and coefficients; none are invented here |
| Path/service autoencoders | Section 4.6 requires separate reconstruction pretraining and subsequent encoder fine-tuning | Layer structures, latent dimensions, sharing across ODs or lines, normalization and optimizer settings |
| Pretraining corpus | Section 4.6 describes the matrix families but does not fully define corpus construction | This reconstruction uses training-only fitting; document utility state and snapshot sampling used during pretraining |
| Nonnegative flow outputs | Section 4.7 interprets predictions as flow volumes | Declare the output parameterization or any clipping rule; the paper equations do not specify a clipping operation |
| Disconnected and same-station OD cases | Assignment requires a meaningful candidate path set | Declare their treatment consistently with demand targets; do not invent a network traversal |

Mathematical checks for any later implementation: candidate probabilities sum to one for each OD with a nonempty feasible set; excluded paths have zero probability; path flows sum to the retained OD prediction; section contributions use directed incidences; line embeddings are reused at the same interval; and the residual slot never receives a fabricated route.
