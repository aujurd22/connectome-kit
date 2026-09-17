# connectome-kit

[![PyPI](https://img.shields.io/pypi/v/connectome-kit)](https://pypi.org/project/connectome-kit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Analysis toolkit for large connectomes, built and validated on the **Drosophila BANC
whole-CNS connectome** (169,078 neurons / 13.6M synapses; Bates et al., *Nature* 2026,
FlyWire consortium data). Every module implements a validated experiment protocol with
degree-preserving random controls and tested against the real connectome end to end.

## Install

```bash
pip install -e .            # core (numpy/pandas/scipy/networkx)
pip install -e ".[gpu]"     # + torch for GPU Kuramoto (169k-node graphs: ~90s/condition)
pip install -e ".[dev]"     # + pytest
```

## Quick start

```python
from connectome_kit import io, rings, spectrum, chemo, emergence

data = io.load_banc("D:/djr82/connectome/banc_888")   # or any BANC-style dir
edges, meta = data["edges"], data["meta"]

ids = np.unique(np.concatenate([edges.pre.values, edges.post.values]))
idmap = io.id_map(ids); n = len(ids)
pre = edges.pre.map(idmap).values; post = edges.post.map(idmap).values
A = io.build_adjacency(pre, post, n)

# 1. Directed-ring enrichment vs degree-preserving shuffles (memory-safe sampling)
rings.sampled_ring_enrichment(A, n_sample=3000, n_rand=5)
# -> BANC: 2-ring 61x, 3-ring 11.5x, 4-ring 8.1x enrichment (z = 311-2172)

# 2. Spectral gap / modular-federation ladder
B = io.symmetrize_binary(A)
spectrum.spectral_summary(B)
# -> BANC: spectral gap 22x smaller than random; hierarchical slow-mixing ladder

# 3. Neurotransmitter-stratified topology (two-layer design)
nt = meta.set_index("root_id")["neurotransmitter_predicted"]
mat = chemo.nt_transition_matrix(nt.reindex(edges.pre), nt.reindex(edges.post))
chemo.config_model_enrichment(mat)
# -> BANC: fast transmitters ~1.0 (degree-random); monoamine self-loops 6-60x

# 4. Causal emergence (EI over uniform interventions)
emergence.effective_information(pre, post, edges.norm.values, n)
emergence.coarse_grained_ei(pre, post, edges.norm.values, n, cell_type_labels)
# -> BANC negative result: micro EI maximal (0.50); cell-type coarse-graining
#    indistinguishable from random grouping (0.285 vs 0.293)
```

GPU Kuramoto with the T116c layer-perturbation protocol:

```python
from connectome_kit.kuramoto import sync_curve, perturb_sync_comparison

sync_curve(A, ks=(0.32, 0.63, 1.26), n_init=4, device="cuda")
perturb_sync_comparison(pre, post, n, mask=is_modulatory_edge, device="cuda")
# -> BANC: real wiring delays sync threshold 2x vs shuffles;
#    excitatory-output shuffle is the only deterministic transition-shifter
```

## Module map

| Module | Protocol | BANC-validated headline result |
|---|---|---|
| `io` | — | load/dedup/adjacency helpers |
| `nullmodels` | controls | degree-preserving stub matching, masked target swaps |
| `rings` | T108b | 2/3/4-ring enrichment 61x / 11.5x / 8.1x |
| `spectrum` | T109/b/c | gap 22x smaller; lambda=2 bipartite mode; Fiedler bottleneck ~23 neurons |
| `chemo` | T113 | two-layer chemical architecture; monoamine self-loops 6-60x |
| `emergence` | T110 | no causal emergence at any given coarse-graining |
| `kuramoto` | T111/T116c | sync threshold 2x delayed; E vs M perturbation separation |
| `communities` | T115 | modulatory control is partitioned (sensory-motor tax-free) |
| `wiring` | T114 | wire cost 31% of shuffled; long-range carried by E, not M |

## Theory context

The toolkit operationalises the **two-layer design** working hypothesis: a
degree-random fast-transmitter backbone (the structural root of the "wiring is
invisible" theorem for passive processing), an insulated monoamine modulatory
control layer (the only wiring-specific stratum, carrying learning/error gating),
and a modular-federation geometry whose seams suppress whole-brain runaway.
The C. elegans cross-species check (T118) shows both layers are conserved across
3 orders of magnitude of network scale.

## Citing

If you use this toolkit, please cite the underlying data:
Bates et al., *Nature* 2026 (BANC connectome) and Schlegel et al., *Nature* 2024
(cell-type annotations), plus neurotransmitter predictions (Eckstein et al.,
*Cell* 2024). Protocol descriptions: see `docs/` of the companion
night-research workspace (FINDINGS.md, T108-T118).

## License

MIT
