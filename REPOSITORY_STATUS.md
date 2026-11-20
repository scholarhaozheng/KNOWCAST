# Repository status

**Edition:** paper-method pseudocode, updated November 20, 2026.

This edition reconstructs the published method from the [KNOWCAST paper](docs/assets/knowcast-paper.pdf) and preserves the module inventory of source snapshot `eced2272d74d0e30d0bc5a45a9c651fc27631736`. The original experiment-producing final code and checkpoints have not been recovered; the pseudocode is a method specification.

| Included | Location |
| --- | --- |
| 42 original module counterparts and four added modules | Original source directories; `.pseudo` extension |
| 200 original declarations and additional helpers | [Coverage inventory](audit/SOURCE_COVERAGE.md) |
| Seven original CSV reference tables | `dmn_knw_gnrtr/` |
| Paper equations, reported settings and shared interfaces | [Specifications](specs/paper_to_pseudocode.md) |
| Structural review and independent formula examples | [Audit records](audit/README.md) |

The README provides a pseudocode reading route, and the research homepage links to the repository guide. The edition includes the paper, figures and seven reference CSV tables alongside the method specifications.

Start with the [pseudocode guide](PSEUDOCODE_GUIDE.md). [Implementation inputs](REPRODUCIBILITY.md) lists the data and artifacts needed to turn this specification into an executable experiment. Paper-reported results remain identified separately from any future measured results.
