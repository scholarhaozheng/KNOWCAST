# Contributing

Start with the [pseudocode guide](PSEUDOCODE_GUIDE.md) and [module index](validation/MODULES.md).

- Cite the paper section or equation supporting a method change.
- Keep procedures clear about inputs, outputs, units and tensor axes.
- Keep producers and consumers aligned with the [shared interfaces](specs/interface_contracts.md).
- Document configurable choices in [implementation parameters](specs/implementation_parameters.md).
- Maintain observation cutoffs, flow conservation and training-only fitting.
- Add numerical or gradient checks for changed calculations.
- Label reported paper results and new measurements by their source.

Place algorithms in their `.pseudo` modules and shorter explanations in `pseudocode/`. The README and research homepage link to the reading guide. Use aggregated examples for contributions involving AFC data, and keep credentials in environment configuration.
