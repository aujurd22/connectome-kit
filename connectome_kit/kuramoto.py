"""Kuramoto phase synchronisation on a connectome (T111/T116c protocol).

Row-normalised coupling dtheta_i/dt = K * sum_j W_ij sin(theta_j - theta_i),
explicit Euler on GPU (torch, optional) or CPU (scipy).

Validated findings on BANC: the real wiring delays the global synchronisation
threshold 2x relative to degree-preserving shuffles (K* = 1.26 vs 0.63) --
module seams suppress whole-brain coherence; T116c: randomising excitatory
long-range outputs is the only perturbation that deterministically moves the
transition (r = 0.970 +/- 0.005 at K=0.63), while modulatory-output
randomisation only suppresses sub-critical activity.

Protocol lessons (each cost us a rerun): coupling must be the sin difference
form (pure drive never synchronises); keep K*dT < 0.2 for Euler stability;
single runs at the transition point are non-quantitative (real-graph
init variance +/-0.14 there) -- average over initial conditions.
"""
from __future__ import annotations

import numpy as np


def _sync_torch(A, n, K, dt, nsteps, seeds, device="cuda"):
    import torch

    results = []
    for seed in seeds:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        th = torch.rand(n, device=device, generator=g) * 2 * np.pi - np.pi
        for _ in range(nsteps):
            s, c = torch.sin(th), torch.cos(th)
            d = (c * torch.sparse.mm(A, s.unsqueeze(1)).squeeze(1)
                 - s * torch.sparse.mm(A, c.unsqueeze(1)).squeeze(1))
            th = th + dt * K * d
        results.append(float(torch.abs(torch.exp(1j * th).mean())))
    return results


def _sync_scipy(A, n, K, dt, nsteps, seeds):
    import scipy.sparse as sp

    Asp = A.tocsr()
    results = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        th = rng.uniform(-np.pi, np.pi, n)
        for _ in range(nsteps):
            s, c = np.sin(th), np.cos(th)
            # NB: `c * Asp @ s` parses as (c*Asp)@s = Asp@(c*s) -- column
            # broadcasting makes both terms identical and d identically zero.
            d = c * (Asp @ s) - s * (Asp @ c)
            th = th + dt * K * d
        results.append(float(np.abs(np.exp(1j * th).mean())))
    return results


def sync_curve(A, ks=(0.16, 0.63, 1.26), dt: float = 0.01, nsteps: int = 3000,
               n_init: int = 4, seed0: int = 0, device: str | None = None) -> dict:
    """Order parameter r(K) with initial-condition averaging.

    ``A``: CSR adjacency (row-normalised internally; K is then the effective
    coupling). Set ``device='cuda'`` for the torch path on large graphs.
    """
    from .io import degree_stats

    n = A.shape[0]
    out_deg = degree_stats(A)["out"]
    src = np.repeat(np.arange(n), np.diff(A.tocsr().indptr))
    dst = A.tocsr().indices
    w = 1.0 / out_deg[src]
    seeds = [seed0 * 1000 + i for i in range(n_init)]

    out = {}
    if device == "cuda":
        import torch

        idx = torch.tensor(np.vstack([src, dst]), dtype=torch.long, device=device)
        val = torch.tensor(w.astype(np.float32), device=device)
        Asp = torch.sparse_coo_tensor(idx, val, (n, n)).coalesce()
        for K in ks:
            vals = _sync_torch(Asp, n, float(K), dt, nsteps, seeds, device)
            out[float(K)] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                             "values": vals}
    else:
        import scipy.sparse as sp

        Asp = sp.csr_matrix((w, (src, dst)), shape=(n, n))
        for K in ks:
            vals = _sync_scipy(Asp, n, float(K), dt, nsteps, seeds)
            out[float(K)] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                             "values": vals}
    return out


def perturb_sync_comparison(pre, post, n, mask, ks=(0.63,), n_seed: int = 3,
                            n_init: int = 2, device: str | None = None) -> dict:
    """T116c protocol: real vs masked-subset target-shuffle vs complementary
    shuffle, with initial-condition averaging."""
    from .nullmodels import swap_targets

    out = {}
    curves = sync_curve(
        __import__("scipy.sparse", fromlist=["csr"]).csr_matrix(
            (np.ones(len(pre)), (pre, post)), shape=(n, n)),
        ks=ks, n_init=n_init, device=device)
    out["real"] = curves
    for tag, m in (("masked_shuf", mask), ("complement_shuf", ~mask)):
        acc = {float(k): [] for k in ks}
        for s in range(n_seed):
            _, dst_r = swap_targets(pre, post, mask=m, seed=100 + s)
            Ar = __import__("scipy.sparse", fromlist=["csr"]).csr_matrix(
                (np.ones(len(pre)), (pre, dst_r)), shape=(n, n))
            c = sync_curve(Ar, ks=ks, n_init=n_init, seed0=100 + s, device=device)
            for k in acc:
                acc[k].append(c[k]["mean"])
        out[tag] = {float(k): {"mean": float(np.mean(v)), "std": float(np.std(v)),
                               "values": v} for k, v in acc.items()}
    return out
