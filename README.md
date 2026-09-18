# connectome-kit

[![PyPI](https://img.shields.io/pypi/v/connectome-kit)](https://pypi.org/project/connectome-kit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)


Analysis toolkit for large connectomes, built and validated on the **Drosophila
BANC whole-CNS connectome** (~170k neurons / ~13.6M deduplicated directed
connections; Bates et al., *Nature* 2026, FlyWire consortium data). Every
module ships with degree-preserving random controls so enrichment results can
be checked against a matched null.

## Install

```bash
pip install -e .            # core (numpy/pandas/scipy/networkx)
pip install -e ".[gpu]"     # + torch for GPU Kuramoto (~170k-node graphs, ~90s/condition)
pip install -e ".[dev]"     # + pytest
```

## Quick start

```python
from connectome_kit import io, rings, spectrum, chemo, emergence

data = io.load_banc("path/to/banc_data")   # any BANC-style directory
edges, meta = data["edges"], data["meta"]

ids = np.unique(np.concatenate([edges.pre.values, edges.post.values]))
idmap = io.id_map(ids); n = len(ids)
pre = edges.pre.map(idmap).values; post = edges.post.map(idmap).values
```

## Modules

| Module | What it does |
|---|---|
| `io` | BANC loading, adjacency construction, ID mapping |
| `nullmodels` | Degree-preserving stub-matching shuffles, masked target swaps |
| `rings` | Sampled directed-ring enrichment (memory-safe, no matrix powers) |
| `spectrum` | Spectral gap, modular ladder, Fiedler localisation |
| `chemo` | Neurotransmitter-stratified topology (two-layer architecture) |
| `emergence` | Effective information / causal emergence |
| `kuramoto` | GPU phase synchronisation with perturbation protocols |

## Notes on the data

The 13.6M figure refers to deduplicated directed connections (pre→post pairs),
not raw synapse counts. If you work with FlyWire's ~50M raw synapses, expect
your own deduplicated counts to differ.

If you don't have BANC data locally, the toolkit's functions accept any
(pre, post) edge list + annotation table — the BANC-specific loader is a
convenience wrapper, not a requirement.

## Citing

If you use this toolkit, please cite the underlying data (Bates et al.,
*Nature* 2026; Dorkenwald et al., *Nature* 2024) and neurotransmitter
predictions (Eckstein et al., *Cell* 2024).

## License

MIT
