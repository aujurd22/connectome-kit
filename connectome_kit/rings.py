"""Sampled directed-ring statistics (T108b protocol).

Directed 2/3/4-ring counts per node against degree-preserving random controls.
On the BANC this recovered 61x / 11.5x / 8.1x enrichment respectively
(z = 311-2172), i.e. a strongly loop-enriched local architecture dominated
by reciprocal pairs (2-rings coincide with mutual pairs).

Memory-safe: per-node matvecs only (vector-level RAM), never matrix powers.
"""
from __future__ import annotations

import json
import time

import numpy as np
import scipy.sparse as sp

from .nullmodels import random_graph_like


def ring_stats(A: sp.spmatrix, sample: np.ndarray) -> dict:
    """Mean per-node 2/3/4-ring counts over ``sample`` node indices."""
    At = A.T.tocsr()
    A64 = A.tocsr().astype(np.float64)
    s2 = s3 = s4 = 0.0
    mutual = 0
    for i in sample:
        oi = set(A64.indices[A64.indptr[i]:A64.indptr[i + 1]].tolist())
        ii = set(At.indices[At.indptr[i]:At.indptr[i + 1]].tolist())
        mutual += len(oi & ii)
        vi = A64[i, :].toarray().ravel()
        v2 = vi @ A64
        v3 = v2 @ A64
        s2 += v2[i]
        s3 += v3[i]
        s4 += v3 @ A64[i, :].toarray().ravel()
    k = len(sample)
    return {"ring2_per_node": s2 / k, "ring3_per_node": s3 / k,
            "ring4_per_node": s4 / k, "mutual_pairs": mutual}


def sampled_ring_enrichment(A: sp.spmatrix, n_sample: int = 3000, n_rand: int = 5,
                            seed: int = 123, verbose: bool = False) -> dict:
    """Real vs degree-preserving-random ring enrichment with z-scores.

    Returns per-ring-type dicts with ``real``, ``rand_mean``, ``rand_std``,
    ``z`` and ``ratio``. Runtime on BANC (n_sample=3000, n_rand=5): ~9 min.
    """
    t0 = time.time()
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    sample = rng.choice(n, size=n_sample, replace=False)
    real = ring_stats(A, sample)
    rands = []
    for s in range(n_rand):
        Ar = random_graph_like(A, seed=s)
        rands.append(ring_stats(Ar, sample))
        if verbose:
            print(f"rand seed{s}: {rands[-1]}", flush=True)
    out = {"sample_size": n_sample, "N": n, "E": int(A.nnz), "runtime_s": time.time() - t0}
    for key in ("ring2_per_node", "ring3_per_node", "ring4_per_node"):
        vals = np.array([r[key] for r in rands])
        out[key] = {
            "real": real[key], "rand_mean": float(vals.mean()),
            "rand_std": float(vals.std() + 1e-9),
            "z": float((real[key] - vals.mean()) / (vals.std() + 1e-9)),
            "ratio": float(real[key] / (vals.mean() + 1e-9)),
        }
    return out
