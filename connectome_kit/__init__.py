"""connectome-kit: analysis toolkit for large connectomes.

Built and validated on the Drosophila BANC whole-CNS connectome
(169,078 neurons / 13.6M synapses; Bates et al., Nature 2026) during a
night-research session (T108-T118, 2026-09-17). Every module corresponds to
a validated experiment protocol with degree-preserving random controls:

- :mod:`connectome_kit.io` -- BANC loading, adjacency construction
- :mod:`connectome_kit.nullmodels` -- degree-preserving stub-matching shuffles
- :mod:`connectome_kit.rings` -- sampled directed-ring enrichment (61x/11.5x/8.1x)
- :mod:`connectome_kit.spectrum` -- spectral gap / modular-federation ladder
- :mod:`connectome_kit.chemo` -- neurotransmitter-stratified topology (two-layer design)
- :mod:`connectome_kit.emergence` -- effective information / causal emergence
- :mod:`connectome_kit.kuramoto` -- phase synchronisation, layer perturbation
- :mod:`connectome_kit.communities` -- Louvain modules x chemical signatures
- :mod:`connectome_kit.wiring` -- spatial wiring economy

Two-layer design summary (the theory this toolkit tests): the fast-transmitter
backbone wires at degree-random enrichment ~1.0 (the structural root of the
"wiring is invisible" theorem), while monoamine systems form a strongly
self-looped, cross-linked modulatory control layer (6-60x) insulated from
the backbone and conserved across 3 orders of magnitude of network scale.
"""

__version__ = "0.1.0"

__all__ = ["io", "nullmodels", "rings", "spectrum", "chemo", "emergence",
           "kuramoto", "communities", "wiring", "__version__"]
