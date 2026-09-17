"""I/O and graph construction helpers for connectome datasets.

Validated on the Drosophila BANC whole-CNS connectome (Bates et al., Nature 2026):
169,078 neurons / 13,620,865 deduplicated synapses with predicted neurotransmitter
labels and functional annotations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


def load_banc(data_dir: str) -> dict:
    """Load a BANC-style connectome directory.

    Expects ``edgelist_simple_v3.feather`` (pre, post, count, norm, ...),
    ``meta.feather`` (root_id + annotations incl. neurotransmitter_predicted,
    cns_network, super_cluster, malecns_cell_type) and optionally
    ``positions_um.feather`` (root_id, x, y, z).

    Returns dict with keys: ``edges``, ``meta``, ``positions``.
    """
    edges = pd.read_feather(f"{data_dir}/edgelist_simple_v3.feather")
    meta = pd.read_feather(f"{data_dir}/meta.feather").drop_duplicates("root_id", keep="first")
    out = {"edges": edges, "meta": meta}
    try:
        out["positions"] = pd.read_feather(f"{data_dir}/positions_um.feather").drop_duplicates("root_id", keep="first")
    except (FileNotFoundError, OSError):
        out["positions"] = None
    return out


def id_map(ids: np.ndarray | pd.Index) -> dict:
    """Root IDs (18-digit) -> dense 0..n-1 indices."""
    return {v: i for i, v in enumerate(np.asarray(ids))}


def build_adjacency(pre: np.ndarray, post: np.ndarray, n: int,
                    weight: np.ndarray | None = None, dtype=np.float64) -> sp.csr_matrix:
    """Directed CSR adjacency from edge index arrays."""
    if weight is None:
        weight = np.ones(len(pre), dtype=dtype)
    return sp.csr_matrix((weight, (pre, post)), shape=(n, n))


def symmetrize_binary(A: sp.spmatrix) -> sp.csr_matrix:
    """A + A^T with duplicates collapsed and all weights set to 1."""
    B = (A + A.T).tocsr()
    B.sum_duplicates()
    B.data[:] = 1.0
    return B


def degree_stats(A: sp.spmatrix) -> dict:
    out_deg = np.diff(A.tocsr().indptr)
    in_deg = np.bincount(A.tocsc().indices, minlength=A.shape[0])
    return {"out": out_deg, "in": in_deg, "mean_out": float(out_deg.mean())}
