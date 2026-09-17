"""Smoke tests on a synthetic directed graph (100 nodes, two dense blocks)."""
import numpy as np
import pytest
import scipy.sparse as sp


@pytest.fixture
def small_graph():
    rng = np.random.default_rng(0)
    n = 100
    src = rng.integers(0, n, 800)
    dst = rng.integers(0, n, 800)
    keep = src != dst
    A = sp.csr_matrix((np.ones(keep.sum()), (src[keep], dst[keep])), shape=(n, n))
    A.sum_duplicates()
    A.data[:] = 1.0
    pre = np.repeat(np.arange(n), np.diff(A.indptr))
    post = A.indices
    return A, pre, post, n


def test_nullmodels_preserve_counts(small_graph):
    from connectome_kit.nullmodels import stub_matching_targets

    A, pre, post, n = small_graph
    src_r, dst_r = stub_matching_targets(pre, post, n, seed=1)
    assert len(src_r) == len(pre)
    # out-degree sequence preserved exactly (before duplicate collapsing)
    out_real = np.bincount(pre, minlength=n)
    out_rand = np.bincount(src_r, minlength=n)
    assert (out_real == out_rand).all()


def test_random_graph_like_close_size(small_graph):
    from connectome_kit.nullmodels import random_graph_like

    A, _, _, n = small_graph
    Ar = random_graph_like(A, seed=1)
    # duplicate collapsing in the random graph softens degree counts, but the
    # edge count should stay within a few percent of the real graph
    assert abs(Ar.nnz - A.nnz) / A.nnz < 0.05


def test_rings_smoke(small_graph):
    from connectome_kit.rings import sampled_ring_enrichment

    A, _, _, n = small_graph
    res = sampled_ring_enrichment(A, n_sample=50, n_rand=2, seed=0)
    for key in ("ring2_per_node", "ring3_per_node", "ring4_per_node"):
        assert res[key]["real"] >= 0
        assert np.isfinite(res[key]["z"])


def test_spectrum_smoke(small_graph):
    from connectome_kit.spectrum import component_profile, spectral_summary

    A, _, _, n = small_graph
    from connectome_kit.io import symmetrize_binary

    B = symmetrize_binary(A)
    prof = component_profile(B)
    assert prof["n_components"] >= 1
    summ = spectral_summary(B, k=10)
    assert summ["lambda_max"] <= 2.0 + 1e-6
    assert summ["lambda2"] > 0 or summ["n_zero_modes"] > 1


def test_chemo_enrichment(small_graph):
    from connectome_kit.chemo import config_model_enrichment, nt_transition_matrix
    import pandas as pd

    _, pre, post, n = small_graph
    rng = np.random.default_rng(3)
    nts = np.array(["acetylcholine", "gaba", "dopamine", "glutamate"])
    pre_nt = pd.Series(nts[rng.integers(0, 4, len(pre))])
    post_nt = pd.Series(nts[rng.integers(0, 4, len(post))])
    mat = nt_transition_matrix(pre_nt, post_nt)
    enr = config_model_enrichment(mat)
    assert np.isfinite(enr.values).all()
    # observed == enrichment x configuration expectation, by definition
    E = mat.values.sum()
    exp = np.outer(mat.values.sum(1, keepdims=True), mat.values.sum(0, keepdims=True)) / E
    assert np.allclose(enr.values * exp, mat.values)


def test_emergence_micro(small_graph):
    from connectome_kit.emergence import effective_information

    A, pre, post, n = small_graph
    res = effective_information(pre, post, np.ones(len(pre)), n)
    assert res["ei_norm"] >= 0
    assert res["determinism"] >= res["degeneracy"] - 1e-9


def test_emergence_coarse(small_graph):
    from connectome_kit.emergence import coarse_grained_ei

    _, pre, post, n = small_graph
    groups = (np.arange(n) % 4).astype(str)
    res = coarse_grained_ei(pre, post, np.ones(len(pre)), n, groups)
    assert res["K_groups"] == 4
    assert "emergence_delta" in res


def test_kuramoto_cpu_smoke(small_graph):
    from connectome_kit.kuramoto import sync_curve

    A, _, _, n = small_graph
    out = sync_curve(A, ks=(0.5, 5.0), dt=0.05, nsteps=300, n_init=2, device=None)
    assert out[5.0]["mean"] > out[0.5]["mean"]  # stronger coupling -> more sync
    assert out[5.0]["mean"] <= 1.0 + 1e-6


def test_communities_smoke(small_graph):
    from connectome_kit.communities import intermodule_control_tax, louvain_labels, module_chemical_signatures
    from connectome_kit.io import symmetrize_binary

    A, pre, post, n = small_graph
    B = symmetrize_binary(A)
    lab = louvain_labels(B)
    assert len(lab) == n
    roles = np.random.default_rng(0).choice(["E", "I", "M"], n)
    sig = module_chemical_signatures(lab, roles, top_k=5)
    assert len(sig) >= 1
    tax = intermodule_control_tax(lab, pre, post, edge_mask=None, top_k=5, min_edges=1)
    assert len(tax) >= 0


def test_wiring_smoke():
    from connectome_kit.wiring import distance_stats, long_range_role_fraction
    import pandas as pd

    rng = np.random.default_rng(2)
    ids = np.arange(50)
    pos = pd.DataFrame({"x": rng.uniform(0, 100, 50), "y": rng.uniform(0, 100, 50),
                        "z": rng.uniform(0, 100, 50)}, index=pd.Index(ids, name="root_id"))
    pre = rng.integers(0, 50, 200)
    post = rng.integers(0, 50, 200)
    roles = pd.Series(rng.choice(["E", "I", "M"], 50), index=pd.Index(ids, name="root_id"))
    stats = distance_stats(pos, pre, post, n_rand=1)
    assert stats["wire_cost_ratio"] > 0
    res = long_range_role_fraction(pos, pre, post, roles)
    assert "long_range_propensity" in res
