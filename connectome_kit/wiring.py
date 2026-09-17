"""Wiring-economy analysis in physical space (T114 protocol).

Distance distributions vs degree-preserving shuffles, and the
neurotransmitter identity of long-range "exception" connections.

Validated findings on BANC: real wiring costs 31% of the shuffled total
(median 111 um vs 444 um), and long-range edges are dominated by
excitatory neurons (per-neuron long-range propensity E 1.24x > M 0.92x
> I 0.74x) -- fast excitatory projection neurons, not modulators, carry
long-range communication.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .nullmodels import stub_matching_targets


def distance_stats(pos: pd.DataFrame, pre_ids: np.ndarray, post_ids: np.ndarray,
                   n_rand: int = 1, seed: int = 11) -> dict:
    """Euclidean soma-distance statistics of real vs shuffled edges.

    ``pos``: DataFrame indexed by root_id with x/y/z columns (um).
    """
    p = pos[["x", "y", "z"]]
    d_real = _edge_distances(p, pre_ids, post_ids)
    d_rand = []
    for s in range(n_rand):
        _, post_r = stub_matching_targets(pre_ids, post_ids, len(pos), seed=seed + s)
        d_rand.append(_edge_distances(p, pre_ids, post_r))
    d_rand = np.concatenate(d_rand)
    return {
        "median_real_um": float(np.median(d_real)),
        "median_rand_um": float(np.median(d_rand)),
        "mean_real_um": float(d_real.mean()),
        "mean_rand_um": float(d_rand.mean()),
        "wire_cost_ratio": float(d_real.sum() / d_rand.sum()),
        "p99_threshold_um": float(np.percentile(d_real, 99)),
    }


def _edge_distances(p, src_ids, dst_ids) -> np.ndarray:
    xy = p.reindex(src_ids).values
    xz = p.reindex(dst_ids).values
    ok = ~(np.isnan(xy).any(axis=1) | np.isnan(xz).any(axis=1))
    return np.linalg.norm(xy[ok] - xz[ok], axis=1)


def long_range_role_fraction(pos: pd.DataFrame, pre_ids: np.ndarray,
                             post_ids: np.ndarray, roles: pd.Series,
                             percentile: float = 99) -> dict:
    """Role composition of long-range (>= percentile) vs short (<= p20) edges.

    ``roles``: Series indexed by root_id -> 'E'/'I'/'M'/'O'/'U' (see
    :func:`connectome_kit.chemo.assign_roles`). Returns the normalised
    per-role long-range propensity (far_frac / all_frac).
    """
    p = pos[["x", "y", "z"]]
    d = _edge_distances(p, pre_ids, post_ids)
    ok = np.zeros(len(pre_ids), dtype=bool)
    ok[:] = False
    xy = p.reindex(pre_ids).values
    xz = p.reindex(post_ids).values
    ok[:] = ~(np.isnan(xy).any(axis=1) | np.isnan(xz).any(axis=1))
    r = roles.reindex(pd.Index(pre_ids)).fillna("U").values
    far = d >= np.percentile(d, percentile)
    near = d <= np.percentile(d, 20)
    all_f = pd.Series(r[ok]).value_counts(normalize=True)
    far_f = pd.Series(r[ok][far]).value_counts(normalize=True)
    near_f = pd.Series(r[ok][near]).value_counts(normalize=True)
    propensity = {k: float(far_f.get(k, 0.0) / (all_f.get(k, 1e-9))) for k in all_f.index}
    return {"far_fraction": {str(k): float(v) for k, v in far_f.items()},
            "near_fraction": {str(k): float(v) for k, v in near_f.items()},
            "long_range_propensity": propensity}
