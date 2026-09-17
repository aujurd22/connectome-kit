"""Community structure x chemical signatures (T115 protocol).

Louvain partition chemistry: each topological module gets a neurotransmitter
signature (E/I/M fractions, DA fraction); inter-module paths get a
"modulatory tax" (fraction of path edges from modulatory neurons).

Validated findings on BANC (248 modules): sensory-motor mega-modules are
nearly tax-free (ctl < 4.5%) while an olfactory-associative-central axis
pays 16-21% -- modulatory control is a partitioned economy, not a uniform
blanket. The DA core module (19.6k neurons, 21.8% M) tops both inflow and
outflow ranks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from networkx.algorithms.community import louvain_communities
    import networkx as nx
    _HAS_NX = True
except ImportError:  # pragma: no cover
    _HAS_NX = False


def louvain_labels(B, resolution: float = 1.0, seed: int = 42) -> np.ndarray:
    """Louvain communities on the symmetrised graph -> dense label array."""
    if not _HAS_NX:  # pragma: no cover
        raise ImportError("networkx>=3.0 required")
    G = nx.from_scipy_sparse_array(B, create_using=nx.Graph)
    comms = sorted(louvain_communities(G, seed=seed, resolution=resolution),
                   key=len, reverse=True)
    lab = np.zeros(B.shape[0], dtype=int)
    for ci, cset in enumerate(comms):
        for x in cset:
            lab[x] = ci
    return lab


def module_chemical_signatures(lab: np.ndarray, roles: np.ndarray,
                               top_k: int = 25) -> pd.DataFrame:
    """Per-module role composition (E/I/M/O/U fractions and EI ratio)."""
    rows = []
    for ci in range(min(len(np.unique(lab)), top_k)):
        m = lab == ci
        vals, counts = np.unique(roles[m], return_counts=True)
        frac = dict(zip(vals, counts / max(m.sum(), 1)))
        rows.append({
            "module": ci, "size": int(m.sum()),
            "frac_E": frac.get("E", 0.0), "frac_I": frac.get("I", 0.0),
            "frac_M": frac.get("M", 0.0), "frac_O": frac.get("O", 0.0),
            "frac_U": frac.get("U", 0.0),
            "EI_ratio": frac.get("E", 0.0) / (frac.get("I", 0.0) + 1e-9),
        })
    return pd.DataFrame(rows)


def intermodule_control_tax(lab: np.ndarray, pre: np.ndarray, post: np.ndarray,
                            edge_mask: np.ndarray | None = None,
                            top_k: int = 25, min_edges: int = 200) -> pd.DataFrame:
    """Modulatory (or masked-subset) fraction of each inter-module path.

    ``edge_mask``: boolean per-edge mask of the controlling edge subset
    (e.g. modulatory-neuron outputs). Rows with >= ``min_edges`` total edges.
    """
    K = min(int(lab.max()) + 1, top_k)
    glab = lab
    m = (glab[pre] < K) & (glab[post] < K)
    flow = np.zeros((K, K))
    flow_m = np.zeros((K, K))
    np.add.at(flow, (glab[pre][m], glab[post][m]), 1)
    sel = m & edge_mask if edge_mask is not None else m
    np.add.at(flow_m, (glab[pre][sel], glab[post][sel]), 1)
    ctl = flow_m / np.maximum(flow, 1)
    rows = []
    for i in range(K):
        for j in range(K):
            if i != j and flow[i, j] >= min_edges:
                rows.append({"from": i, "to": j, "edges": int(flow[i, j]),
                             "mask_fraction": float(ctl[i, j])})
    return pd.DataFrame(rows).sort_values("mask_fraction", ascending=False)
