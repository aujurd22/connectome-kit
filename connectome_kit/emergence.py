"""Effective information and causal-emergence analysis (T110 protocol).

EI = Determinism - Degeneracy over the row-normalised transition matrix with
uniform interventions (Hoel et al.). Validated negative result on BANC:
micro-level normalised EI is maximal (0.50) and every biologically given
coarse-graining (cell type / functional cluster / brain partition) reduces
it -- cell-type coarse-graining is indistinguishable from a random grouping
(0.285 vs 0.293), i.e. the fly brain shows no causal emergence at any
biologist-defined macro scale.

Note: this measures *given* groupings, not EI-maximising optimal
coarse-grainings (Causal Emergence 2.0 sense).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def effective_information(pre: np.ndarray, post: np.ndarray, weights: np.ndarray,
                          n_nodes: int) -> dict:
    """Micro-level EI of a weighted directed graph (row-normalised TPM)."""
    out_sum = np.bincount(pre, weights=weights, minlength=n_nodes)
    has_out = out_sum > 0
    keep = has_out[pre]
    s, t, w = pre[keep], post[keep], weights[keep]
    p = w / out_sum[s]
    N_eff = int(has_out.sum())
    det = float(np.sum(p * (np.log(p) + np.log(N_eff))) / N_eff)
    D = np.bincount(t, weights=p / N_eff, minlength=n_nodes)
    nz = D > 0
    deg = float(np.sum(D[nz] * (np.log(D[nz]) + np.log(N_eff))))
    EI = det - deg
    return {"EI": EI, "determinism": det, "degeneracy": deg,
            "N_eff": N_eff, "ei_norm": float(EI / np.log2(N_eff))}


def coarse_grained_ei(pre: np.ndarray, post: np.ndarray, weights: np.ndarray,
                      n_nodes: int, groups: np.ndarray,
                      n_random_controls: int = 1, seed: int = 99) -> dict:
    """EI of a group-level TPM (within-group row averaging) vs random groupings.

    ``emergence_delta`` = ei_norm(macro) - ei_norm(micro); negative values mean
    coarse-graining destroys causal power (no emergence). The random-grouping
    control separates "this grouping carries structure" from "fewer states
    trivially raise EI".
    """
    import scipy.sparse as sp

    micro = effective_information(pre, post, weights, n_nodes)
    uniq, codes = np.unique(groups, return_inverse=True)
    K = len(uniq)

    def group_ei(gs: np.ndarray, gd: np.ndarray, pw: np.ndarray) -> float:
        Keff = len(np.unique(gs))
        out_sum = np.zeros(K)
        np.add.at(out_sum, gs, pw)
        keep = out_sum[gs] > 0
        rr, cc, pp = gs[keep], gd[keep], pw[keep] / out_sum[gs][keep]
        det = float(np.sum(pp * (np.log(pp) + np.log(Keff))) / Keff)
        D = np.bincount(cc, weights=pp / Keff, minlength=K)
        nz = D > 0
        deg = float(np.sum(D[nz] * (np.log(D[nz]) + np.log(Keff))))
        return (det - deg) / np.log2(Keff)

    gc = pd.DataFrame({"gs": codes[pre], "gd": codes[post], "p": weights})
    agg = gc.groupby(["gs", "gd"])["p"].mean().reset_index()
    ei_macro = group_ei(agg.gs.values, agg.gd.values, agg.p.values)

    rng = np.random.default_rng(seed)
    ei_rand = []
    for _ in range(n_random_controls):
        codes_r = rng.permutation(codes)
        gc_r = pd.DataFrame({"gs": codes_r[pre], "gd": codes_r[post], "p": weights})
        agg_r = gc_r.groupby(["gs", "gd"])["p"].mean().reset_index()
        ei_rand.append(group_ei(agg_r.gs.values, agg_r.gd.values, agg_r.p.values))

    return {"micro_ei_norm": micro["ei_norm"], "K_groups": int(K),
            "macro_ei_norm": float(ei_macro),
            "macro_ei_norm_rand_mean": float(np.mean(ei_rand)),
            "emergence_delta": float(ei_macro - micro["ei_norm"]),
            "delta_vs_random_groups": float(ei_macro - np.mean(ei_rand))}
