"""Degree-preserving null models.

The stub-matching randomisation used throughout the night-research experiments
(T108-T118): keep the exact out-degree sequence, redraw targets from the
in-degree stub pool. Memory-safe by construction (edge-count arrays only).

Lesson learned in T108: full-matrix ring counting (tr A^3) on a 169k-node
graph peaks at ~16.5 GB. Always prefer the sampled estimators in
``connectome_kit.rings``; the randomisation here is the matching control.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def stub_matching_targets(pre: np.ndarray, post: np.ndarray, n: int, seed: int = 0):
    """Return (src, dst) with identical out-degree sequence and redrawn targets.

    Targets are drawn from the in-degree stub pool, so the in-degree
    *multiset* is preserved exactly while pairings are randomised.
    """
    rng = np.random.default_rng(seed)
    in_deg = np.bincount(post, minlength=n)
    pool = np.repeat(np.arange(n), in_deg)
    rng.shuffle(pool)
    return pre, pool[: len(pre)]


def swap_targets(pre: np.ndarray, post: np.ndarray, mask: np.ndarray | None = None,
                 seed: int = 0, n_swaps: int | None = None):
    """Swap targets between a masked edge subset and random other edges.

    Used by the T116 layer-perturbation protocol: ``mask`` selects the edges
    of the subpopulation to randomise (e.g. modulatory-neuron outputs).
    Preserves total edge count and both subsets' sizes, not per-node degrees.
    """
    rng = np.random.default_rng(seed)
    dst = post.copy()
    if mask is None:
        idx_m = np.arange(len(dst))
        idx_o = np.array([], dtype=int)
    else:
        idx_m = np.where(mask)[0]
        idx_o = np.where(~mask)[0]
    if n_swaps is None:
        n_swaps = min(len(idx_m), len(idx_o))
    pm = rng.choice(idx_m, size=n_swaps, replace=False)
    po = rng.choice(idx_o, size=n_swaps, replace=False)
    dst[pm], dst[po] = dst[po].copy(), dst[pm].copy()
    return pre, dst


def random_graph_like(A: sp.spmatrix, seed: int = 0) -> sp.csr_matrix:
    """Degree-preserving random graph from a CSR adjacency."""
    n = A.shape[0]
    Ac = A.tocsr()
    src = np.repeat(np.arange(n), np.diff(Ac.indptr))
    dst = Ac.indices
    src_r, dst_r = stub_matching_targets(src, dst, n, seed=seed)
    Ar = sp.csr_matrix((np.ones(len(src_r)), (src_r, dst_r)), shape=(n, n))
    Ar.sum_duplicates()
    Ar.data[:] = 1.0
    return Ar
