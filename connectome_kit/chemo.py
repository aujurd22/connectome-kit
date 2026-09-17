"""Chemoconnectome analysis (T113 protocol).

Neurotransmitter-stratified network topology: transition-matrix enrichment
under a configuration-model baseline, E/I/M role balance per brain partition.

Validated findings on BANC: fast transmitters (ACh/GABA/Glu) wire at
degree-random enrichment ~1.0 while all monoamine systems show strong
self-loops (DA 6.9x, 5HT 7.3x, OA 6.1x, TYR 60x) and monoamine-to-monoamine
cross-talk at 2.3-3.1x -- the "layer 2" signature, conserved down to
C. elegans (T118).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_ROLES = {
    "E": {"acetylcholine"},
    "I": {"gaba", "glutamate"},
    "M": {"dopamine", "octopamine", "serotonin", "tyramine"},
}


def assign_roles(nt: pd.Series, roles: dict | None = None) -> pd.Series:
    roles = roles or DEFAULT_ROLES
    def role(x):
        for tag, names in roles.items():
            if x in names:
                return tag
        return "O" if isinstance(x, str) and x != "unknown" else "U"
    return nt.map(role).fillna("U")


def nt_transition_matrix(pre_nt: pd.Series, post_nt: pd.Series) -> pd.DataFrame:
    """Directed count matrix pre-NT x post-NT."""
    return pd.crosstab(pre_nt, post_nt)


def config_model_enrichment(mat: pd.DataFrame) -> pd.DataFrame:
    """Observed / expected edge counts under the configuration model.

    enrichment[i, j] = mat[i, j] * E / (rowsum[i] * colsum[j]).
    >1 enriched, <1 avoided. BANC validation: fast-transmitter block ~1.0,
    monoamine diagonal 6-60x (T113).
    """
    M = mat.values.astype(float)
    E = M.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        enr = M * E / np.outer(M.sum(1), M.sum(0))
    return pd.DataFrame(np.nan_to_num(enr), index=mat.index, columns=mat.columns)


def partition_ei_balance(nt: pd.Series, partition: pd.Series,
                         roles: dict | None = None) -> pd.DataFrame:
    """Role composition and E/I ratio per partition (brain region etc.)."""
    role = assign_roles(nt, roles)
    tab = pd.crosstab(partition, role)
    for col in ("E", "I", "M", "O", "U"):
        if col not in tab.columns:
            tab[col] = 0
    tab["E/I"] = tab["E"] / (tab["I"] + 1e-9)
    return tab


def transition_enrichment_by_group(pre_groups: np.ndarray, post_groups: np.ndarray,
                                    pre_mask: np.ndarray) -> pd.DataFrame:
    """Fraction of edges on each inter-group path that come from a masked
    (e.g. modulatory) edge subset (T115c protocol)."""
    import scipy.sparse as sp

    K = int(max(pre_groups.max(), post_groups.max())) + 1
    flow = np.zeros((K, K))
    flow_m = np.zeros((K, K))
    np.add.at(flow, (pre_groups, post_groups), 1)
    np.add.at(flow_m, (pre_groups[pre_mask], post_groups[pre_mask]), 1)
    ctl = flow_m / np.maximum(flow, 1)
    return pd.DataFrame(ctl)
