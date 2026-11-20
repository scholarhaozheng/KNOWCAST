# Contributing

Contributions should improve algorithm accuracy, completeness and readability. Start with the [pseudocode guide](PSEUDOCODE_GUIDE.md).

- Cite the paper section or equation supporting a methodological change.
- Put configurable choices in [implementation parameters](specs/implementation_parameters.md).
- Separate historical paper results from independently measured results.
- Preserve tensor semantics, forecast-time observability and training-only fitted artifacts.
- Supply meaningful numerical and gradient checks when adding an executable implementation.
- Exclude credentials, private AFC records and personal database connection details.

The primary algorithms are standalone `.pseudo` files at the original source paths: root entry points, `lib/`, `models/`, `dmn_knw_gnrtr/`, `metro_components/` and `metro_data_convertor/`. The [coverage inventory](audit/SOURCE_COVERAGE.md) maps them to the original modules. The five chapters in `pseudocode/` provide supplementary explanations.

Preserve the original responsibilities and actual `CLASS`/`PROCEDURE` declarations when editing algorithms. Keep source correspondence in the separate audit inventory, rather than embedding provenance tags in algorithm files. Changes should keep the [shared interfaces](specs/interface_contracts.md) consistent across producers and consumers. A declaration inventory is a structural check; also review control flow, units, identities, availability cutoffs and gradient paths.

Keep README entry points consistent with the current algorithms. The research homepage links to the repository guide; full pseudocode belongs in the module files. Edition provenance is recorded in [repository status](REPOSITORY_STATUS.md).
