"""Spectral analysis of the symmetrised connectome (T109/T109b/T109c protocol).

Key validated findings on BANC: spectral gap 22x smaller than the
degree-preserving random control (hierarchical slow-mixing ladder --
the "modular federation" backbone), a lambda=2 bipartite mode, and a
Fiedler bottleneck localised on ~23 unannotated neurons.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components


def normalized_laplacian(B: sp.spmatrix) -> sp.csr_matrix:
    """Symmetric normalised Laplacian of an undirected (symmetrised) graph."""
    n = B.shape[0]
    deg = np.asarray(B.sum(axis=1)).ravel()
    d12 = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1)), 0.0)
    return (sp.identity(n) - sp.diags(d12) @ B @ sp.diags(d12)).tocsr()


def low_spectrum(B: sp.spmatrix, k: int = 30, tol: float = 1e-6):
    """Smallest eigenpairs of the normalised Laplacian (ascending)."""
    vals, vecs = spla.eigsh(normalized_laplacian(B), k=k, which="SA", tol=tol)
    order = np.argsort(vals)
    return vals[order], vecs[:, order]


def component_profile(B: sp.spmatrix) -> dict:
    """Connected component sizes (largest first)."""
    ncomp, labels = connected_components(B, directed=False)
    sizes = np.sort(np.bincount(labels))[::-1]
    return {"n_components": int(ncomp), "sizes_top10": [int(x) for x in sizes[:10]]}


def spectral_summary(B: sp.spmatrix, k: int = 30) -> dict:
    """lambda2 (Fiedler), lambda_max, component count, IPR of the low modes."""
    vals, vecs = low_spectrum(B, k=k)
    nz = np.where(vals > 1e-8)[0]
    fiedler_idx = int(nz[0]) if len(nz) else k - 1
    v = vecs[:, fiedler_idx]
    ipr = float((v**4).sum() / ((v**2).sum() ** 2))
    lam_hi = float(spla.eigsh(normalized_laplacian(B), k=2, which="LA", tol=1e-6)[0].max())
    return {
        "n_zero_modes": int(len(vals[vals <= 1e-8])),
        "lambda2": float(vals[fiedler_idx]),
        "lambda_max": lam_hi,
        "spectral_gap_ratio": float(vals[fiedler_idx] / lam_hi),
        "ladder": [float(x) for x in vals[: min(5, len(vals))]],
        "ipr_fiedler": ipr,
        "fiedler_effective_support": float(1.0 / ipr) if ipr > 0 else None,
    }


def fiedler_partition_load(B: sp.spmatrix, groups: np.ndarray | None = None):
    """Fiedler vector load per group label (anatomical localisation diagnostic).

    ``groups``: array of length n with group names (e.g. cns_network labels).
    Returns (summary dict, per-group load dict, per-group sign-load DataFrame).
    """
    import pandas as pd

    vals, vecs = low_spectrum(B, k=5)
    nz = np.where(vals > 1e-8)[0]
    v = vecs[:, int(nz[0]) if len(nz) else 1]
    res = {"fiedler_lambda": float(vals[int(nz[0]) if len(nz) else 1])}
    if groups is None:
        return res, None, None
    df = pd.DataFrame({"group": groups, "v2": v**2, "sign": np.sign(v)})
    load = df.groupby("group")["v2"].sum()
    load = (load / load.sum()).sort_values(ascending=False)
    sl = df.groupby(["group", "sign"])["v2"].sum().unstack(fill_value=0)
    sl = sl.div(sl.sum(axis=1), axis=0)
    res["load_top10"] = {str(k): float(x) for k, x in load.head(10).items()}
    return res, load, sl
