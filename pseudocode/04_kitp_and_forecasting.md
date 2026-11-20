# KITP, GATRU, and sequence forecasting

Section 3.2 and Fig. 3 of the [paper](../docs/assets/knowcast-paper.pdf) define sequence forecasting; Sections 4.1–4.8 and Fig. 4 define KITP. This chapter uses Eqs. (7)–(13) for allocation and fusion, (17)–(19) for the objective and (31)–(33) for GATRU. [Path and service features](03_path_choice_and_service.md) supply the assignment operations.

## Feature and time contracts

The forecast origin is `t0`. Historical inputs cover `t0 - l_h` through `t0 - 1`; targets cover `t0` through `t0 + l_f - 1`. Static inputs include the pre-calibrated distribution preference parameter `K3` and utility initialization `K5`. Dynamic inputs `K2`, `K4`, `K6`, and `K7` concern trip generation, distribution, path attributes, and service conditions, respectively (Section 3.2).

Future knowledge contains plans, announced conditions or forecasts available at `t0`. During training, ground-truth OD enters through the separate teacher-forcing input.

The conceptual formulas here use one-based retained-slot/horizon labels. Standalone array contracts use retained slots `0..R-1`, residual `R`, and horizon index `h=lambda-1`. The [shared interfaces](../specs/interface_contracts.md) define the explicit conversion and representation tags.

```text
KNOWLEDGE_AT(t, information_available_at_t0)
    select the time-aligned K2, K4, K6, K7 allowed by the information cutoff
    resolve complete auxiliary_context_by_task windows for production and attraction
    preserve auxiliary lengths, checkpoint/scaler/graph IDs and target_offset
    select complete context sequences with inputs available at t0
    generation_estimates <- INDEPENDENT_PRETRAINED_GENERATION_MODEL(
        full auxiliary covariate windows including K2, target_offset)
    distribution_embedding <- DISTRIBUTION_KNOWLEDGE(
        generation_estimates, K4, fixed_precalibrated_K3,
        pretrained_then_finetuned_distribution_encoder)
        [Section 4.4: doubly constrained gravity + IPF + autoencoder]

    path_probabilities, section_probabilities, feasible_sets, incidence <-
        PATH_CHOICE(t, K7, candidate_paths, K6, current_utility_parameters)
    Mpl, Mdf <- CONSTRUCT_SERVICE_MATRICES(t, section_probabilities, K7)
    F_KI <- ENCODE_SERVICE_KNOWLEDGE(t, Mpl, Mdf, current_feature_encoders)

    return distribution_embedding, F_KI, generation_estimates,
           path_probabilities, feasible_sets, incidence

BUILD_NODE_FEATURES(primary_compressed_OD, knowledge, fixed_RDS)
    for each origin node m:
        primary_row <- primary_compressed_OD[m, all R + 1 slots]
        ordered_KI <- CONCATENATE over retained s = 1, ..., R
                      of knowledge.F_KI[(m, RDS[m, s])]
        node_feature[m] <- CONCATENATE(
            primary_row,
            ordered_KI,
            ALIGNED_DISTRIBUTION_FEATURES(m, knowledge.distribution_embedding))
        [Section 4.1 and Section 4.8]

    preserve the fixed origin-specific destination order
    retain the residual OD value as an aggregate demand channel
    return node_feature
```

`ALIGNED_DISTRIBUTION_FEATURES` maps the distribution embedding to station features using the configured reshape or projection. Normalization preserves station and retained-destination order, with an inverse transform to passenger counts.

## GATRU follows the paper's update-gate convention

The directed graph defines allowed attention neighborhoods. Eq. (31) normalizes learned attention over incoming neighbors and the node itself. Configuration supplies the scoring function and activation.

For one interval, node inputs have shape `[N,d_in]`. With `K_head` heads of width `d_head`, concatenated attention features have shape `[N,K_head*d_head]`. The recurrent width is `h`; reset and update gates, candidate state, previous state and new state all have shape `[N,h]`.

```text
GATRU_STEP(X, H_previous, directed_graph, parameters)
    for each node m:
        neighborhood <- ALLOWED_MESSAGE_SENDERS(m, directed_graph) UNION {m}
        for each attention head k:
            scores[n] <- LEARNED_ATTENTION_SCORE(k, X[m], X[n])
                         for n in neighborhood
            alpha[k, m, :] <- SOFTMAX(scores over this neighborhood only)
            head_output[k] <- NONLINEAR_ACTIVATION(
                SUM over n in neighborhood of alpha[k, m, n] * W_gat[k] * X[n])
        F[m] <- CONCATENATE(head_output[k] over heads)
                [Eq. (31)]

    for each node m, using the previous-step states:
        r[m] <- SIGMOID(W_r * CONCATENATE(H_previous[m], F[m]) + b_r)
        z[m] <- SIGMOID(W_z * CONCATENATE(H_previous[m], F[m]) + b_z)
        candidate[m] <- TANH(
            W * CONCATENATE(r[m] ELEMENTWISE_MULTIPLY H_previous[m], F[m]) + b)
        H_new[m] <- (1 - z[m]) ELEMENTWISE_MULTIPLY H_previous[m]
                    + z[m] ELEMENTWISE_MULTIPLY candidate[m]
                    [Eq. (32)]

    return the collection H_new for all nodes [Eq. (33)]
```

In Eq. (32), `z` weights the candidate state and `1-z` weights the preceding state.

## Fuse the three unfinished-trip references

Section 4.2.4 uses previous-day, previous-week and previous-calendar-month references. The destination map is fitted on the training period and shared across splits.

```text
UNFINISHED_ALLOCATIONS(t, unfinished_counts, historical_reference_OD)
    for each reference delta in {previous_day, previous_week, previous_month}:
        for each origin m:
            reference_row <- historical_reference_OD[delta, t, m, all R + 1 slots]
            total <- SUM(reference_row)
            if total > 0:
                destination_share <- reference_row / total
            otherwise:
                destination_share <- UNIFORM_SHARE(1 / (R + 1))
                                     over all R + 1 slots
                                     [Eq. (7)]
            allocation[delta, m, :] <-
                destination_share * unfinished_counts[t, m]
                [Eq. (8)]
    return the three allocations

FUSE_UNFINISHED_AND_FINISHED(H_day, H_week, H_month, H_finished)
    require all four branch states to have the same shape [N, h]
    for each origin node m:
        H_combined <- CONCATENATE(H_day[m], H_week[m], H_month[m])
                      [width 3 * h]
        Z_combined <- H_combined + COMPATIBLE_POSITIONAL_ENCODING
        gate_output <- GATING_LAYER(Z_combined, Z_combined, Z_combined)
                       [Eq. (10): abstract attention-based gating primitive]
        require gate_output width 3 * h
        W_day, W_week, W_month <- SPLIT_IN_REFERENCE_ORDER(gate_output)
                                 [Eq. (11): three blocks, each width h]
        H_unfinished[m] <-
            SIGMOID(W_day) ELEMENTWISE_MULTIPLY H_day[m]
            + SIGMOID(W_week) ELEMENTWISE_MULTIPLY H_week[m]
            + SIGMOID(W_month) ELEMENTWISE_MULTIPLY H_month[m]
            [Eq. (12)]
        fused_output[m] <- H_unfinished[m] + H_finished[m]
                           [Eq. (13)]
    return fused_output

ENCODE_HISTORY(t0, history, information_available_at_t0)
    initialize the finished and three unfinished recurrent states to zero
    initialize any declared additional encoder states to zero

    for t from t0 - l_h through t0 - 1 in chronological order:
        knowledge <- KNOWLEDGE_AT(t, information_available_at_t0)
        allocation <- history.allocation_day/week/month at t
        require representation_kind = unfinished_allocation
        require its model-scale transform matches the finished-input transform
        [The loader already applied Eqs. (7)-(8) in count units and transformed once]

        branch_inputs <- ENCODER_FEATURE_INTERFACE(
            history.finished[t], allocation, knowledge, fixed_RDS)
        H_finished <- FINISHED_GATRU_STEP(branch_inputs.finished, H_finished)
        H_day <- DAY_GATRU_STEP(branch_inputs.day, H_day)
        H_week <- WEEK_GATRU_STEP(branch_inputs.week, H_week)
        H_month <- MONTH_GATRU_STEP(branch_inputs.month, H_month)
        [Separate temporal updates; unfinished branches implement Eq. (9)]

    fused_output <- FUSE_UNFINISHED_AND_FINISHED(
        H_day, H_week, H_month, H_finished)
    encoder_context <- ENCODER_TO_DECODER_STATE_INTERFACE(fused_output)
    require the decoder context to retain the actual fused contribution
    return encoder_context
```

Fusion uses the final branch states from Section 4.2.4. `ENCODER_FEATURE_INTERFACE` aligns finished OD, unfinished allocations and knowledge features by origin and retained destination. The standard loader supplies transformed allocations; a raw-reference adapter instead performs allocation before this interface. Configuration selects one representation and defines branch-specific feature reuse and the encoder-to-decoder state mapping.

## Decoder steps preserve dynamic knowledge while removing unfinished branches

Section 3.2 removes the three unfinished-trip GATRUs from the decoder because future unfinished counts are unavailable. It retains the current dynamic knowledge inputs. The initial OD input is a zero matrix; later training inputs are preceding ground-truth OD matrices, while inference inputs are preceding predictions.

```text
FORECAST(t0, mode, history, known_information, optional_training_targets)
    H <- ENCODE_HISTORY(t0, history, known_information)
    previous_OD_input <- ZERO_MATRIX_IN_THE_DECLARED_MODEL_INPUT_REPRESENTATION
    [Record whether preprocessing represents this zero before or after scaling]

    for lambda from 1 through l_f:
        t <- t0 + lambda - 1
        knowledge <- KNOWLEDGE_AT(t, known_information)
        X <- BUILD_NODE_FEATURES(previous_OD_input, knowledge, fixed_RDS)
        H <- DECODER_GATRU_STEP(X, H)
        prediction[lambda] <- OD_READOUT(H)

        preserve knowledge.path_probabilities and incidence for this same t
        path_flow[lambda], section_flow[lambda] <- ASSIGN_FORECAST(
            t0, lambda, prediction[lambda], knowledge.path_probabilities,
            knowledge.feasible_sets, knowledge.incidence, fixed_RDS)
        [Eqs. (28)-(29); the residual slot remains excluded]

        if lambda < l_f:
            if mode is TRAINING:
                previous_OD_input <- ENCODE_OD_INPUT(optional_training_targets[t])
                                     [Teacher forcing: Section 3.2]
            otherwise:
                previous_OD_input <- ENCODE_OD_INPUT(prediction[lambda])
                                     [Autoregressive inference: Section 3.2]

    return prediction, path_flow, section_flow
```

`ENCODE_OD_INPUT` converts between the prediction and recurrent-input representations using their transform IDs. `OD_READOUT` projects the recurrent state to compressed OD. Configuration supplies its structure and output treatment.

## Training connects the stated losses to the full forward graph

```text
TRAIN THE DECLARED FULL MODEL
    fit independent trip-generation and gravity components using the training split
    initialize utility coefficients from Section 5.1.4
    record that utility state and the feature definitions used for pretraining
    generate training-only path probabilities and section-probability matrices
        with that recorded utility state and time-aligned service snapshots
    construct training-only distribution and line-service matrix corpora
    pretrain distribution, path and service autoencoders for reconstruction
    initialize the forecasting encoders with their pretrained parameters

    declare the full Eq. (18) mean and full Eq. (16) sum objectives
    declare target multiplicities, sampling, minibatch scaling and accumulation policy
    declare the mapping from displayed training epochs to formula index epsilon
    for each training epoch with its declared epsilon:
        w <- w_initial * MAX(1 - alpha_decay * epsilon, 0) [Eq. (17)]
        for each training batch:
            predictions <- FORECAST(batch origin, TRAINING, batch history,
                                    cutoff-available knowledge, separate training targets)
            od_errors <- unreduced absolute errors in original flow units
                [All retained slots and the residual, with declared target multiplicity]
            generation_errors <- unreduced squared station discrepancies from
                TRIP_GENERATION_CONSISTENCY_LOSS(
                predictions, independent production and attraction estimates,
                fixed training-derived residual allocation)
                [Section 4.3.2, Eqs. (15)-(16)]
            L_step <- DECLARED_MINIBATCH_ESTIMATOR(
                full L_OD + w * full L_generation, od_errors, generation_errors,
                sampling probabilities, full denominators, accumulation policy)
            clear gradients only when beginning the declared accumulation group
            backpropagate L_step through the predictor, feature encoders,
                gating fusion and trainable utility parameters
            update the declared parameter set only when the group is complete
        finish any final partial group with its explicitly declared scaling

    use validation data for configuration and checkpoint selection
    evaluate the selected model autoregressively on the held-out test split
```

Generation-loss and distribution procedures follow their respective modules. Forecasting gradients update the predictor, gating, active utility parameters and feature encoders. Gravity `zeta` stays fixed; the generation-model configuration selects its main-training parameter policy.

## Configuration

| Detail | Method | Configuration |
| --- | --- | --- |
| Attention score and activation | Eq. (31) uses multi-head, neighborhood-normalized attention | Scoring function, activation, head count, dimensions and regularization |
| Attention-based gating | Eqs. (10)-(12) apply positional encoding, produce three aligned gate blocks and use sigmoid weighting | Position axis, encoding construction, attention layout, parameter sharing and gate projection |
| Encoder wiring | Eqs. (9)-(13) require three unfinished branches and actual fusion with the finished branch | Exact knowledge reuse across branches, extra layers and encoder-to-decoder state mapping |
| Node features | Sections 4.1, 4.4 and 4.8 require OD, distribution, path and service information | Tensor layouts and projections while preserving destination order and residual semantics |
| Initial zero input | Section 3.2 specifies a zero OD matrix at the first decoder step | Its relation to any model-input normalization |
| Forecast readout | Section 3.2 requires compressed OD predictions at every horizon | Projection structure, nonnegative output treatment and representation conversion |
| Missing historical dates | Section 4.2.3 uses previous calendar month, week and day | Separate policies for missing dates and observed zero rows |
| Teacher forcing | Section 3.2 feeds the preceding ground-truth OD during training | Training mode uses the preceding target; inference mode uses the preceding prediction |
| Optimization and checkpoints | Sections 4.3 and 5.1.4 specify the loss and validation-based tuning | Formula epoch origin, seeds, batch sizes, stopping and checkpoint rules |

## Invariants

All three unfinished branches contribute to fusion. Every decoder step uses target-time knowledge available at the forecast origin. Active knowledge modules receive forecasting gradients. OD outputs and assignment records share destination IDs, target times and model state.
