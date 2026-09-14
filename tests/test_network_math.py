import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src.advanced_analysis import bimodality_coefficient, correlation_tests, cronbach_alpha, domain_edge_enrichment
from src.analysis import eta_squared, statement_summary
from src.build_networks import build_statement_network, distance_from_similarity, pairwise_cosine
from src.data_prep import filter_respondents


def test_distance_from_similarity_handles_positive_and_nonpositive_values():
    assert np.isclose(distance_from_similarity(1.0), 1.0)
    assert np.isclose(distance_from_similarity(0.5), 2.0)
    assert np.isinf(distance_from_similarity(0.0))
    assert np.isinf(distance_from_similarity(-0.3))


def _brute_cosine(a, b):
    mask = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[mask], b[mask]
    return a @ b / np.sqrt((a @ a) * (b @ b))


def test_pairwise_cosine_matches_brute_force_with_missing_values():
    rng = np.random.default_rng(1)
    data = rng.integers(-2, 3, size=(6, 12)).astype(float)
    data[rng.random(data.shape) < 0.15] = np.nan
    df = pd.DataFrame(data)
    for center in (False, True):
        sim = pairwise_cosine(df, center=center).to_numpy()
        rows = data - np.nanmean(data, axis=1, keepdims=True) if center else data
        for i in range(6):
            for j in range(i + 1, 6):
                assert np.isclose(sim[i, j], _brute_cosine(rows[i], rows[j]))


def test_centring_removes_agreement_level():
    # Same pattern shifted up by one Likert point: raw cosine < 1, centred cosine = 1.
    base = np.array([-1.0, 0.0, 1.0, 0.0, -1.0, 1.0])
    df = pd.DataFrame([base, base + 1])
    assert pairwise_cosine(df, center=False).iat[0, 1] < 0.99
    assert np.isclose(pairwise_cosine(df, center=True).iat[0, 1], 1.0)


def test_negative_edges_have_no_path_distance():
    df = pd.DataFrame({"A": [-2, -1, 0, 1, 2] * 4, "B": [2, 1, 0, -1, -2] * 4, "C": [-2, -1, 0, 1, 2] * 4})
    G, _ = build_statement_network(df, corr_threshold=0.3)
    assert G["A"]["B"]["sign"] == -1 and np.isinf(G["A"]["B"]["distance"])
    assert G["A"]["C"]["sign"] == 1 and np.isclose(G["A"]["C"]["distance"], 1.0)


def test_filter_respondents_drops_sparse_rows():
    cols = [f"T{i:02d}" for i in range(20)] + [f"E{i:02d}" for i in range(20)]
    df = pd.DataFrame([[1.0] * 40, [np.nan] * 40, [1.0] * 10 + [np.nan] * 30], columns=cols)
    kept, dropped = filter_respondents(df, min_answered=30)
    assert list(kept.index) == [0]
    assert list(dropped["answered"]) == [0, 10]
    assert list(dropped["domains_answered"]) == ["none", "T"]


def test_eta_squared_is_size_weighted():
    values = pd.Series([0, 0, 0, 0, 10.0])
    groups = pd.Series([1, 1, 1, 1, 2])
    # Between-group SS = 4·(0−2)² + 1·(10−2)² = 80 = total SS → η² = 1.
    assert np.isclose(eta_squared(values, groups), 1.0)
    # Groups {0, 0, 10} and {0, 0}: SS_between = 3·(10/3 − 2)² + 2·(0 − 2)² = 40/3 → η² = 1/6.
    assert np.isclose(eta_squared(values, pd.Series([1, 2, 1, 2, 1])), 1 / 6)


def test_split_index_separates_split_from_lopsided():
    df = pd.DataFrame({"split": [-2, -1, 1, 2] * 5, "lopsided": [-2, -2, -2, 2] * 5})
    summ = statement_summary(df)
    assert summ.loc["split", "split"] == pytest.approx(0.5)
    assert summ.loc["lopsided", "split"] == pytest.approx(0.25)


def test_benjamini_hochberg_flags_only_real_correlations():
    rng = np.random.default_rng(0)
    base = rng.normal(size=200)
    df = pd.DataFrame({"A": base, "B": base + rng.normal(scale=0.3, size=200),
                       "C": rng.normal(size=200), "D": rng.normal(size=200)})
    tests = correlation_tests(df)
    flagged = {tuple(sorted((r.u, r.v))) for r in tests.itertuples() if r.fdr}
    assert flagged == {("A", "B")}


def test_domain_edge_enrichment_counts_within_domain_links():
    G = nx.Graph()
    G.add_nodes_from([f"{d}{i:02d}" for d in "TESV" for i in range(15)])
    G.add_edges_from([("T00", "T01"), ("T02", "E00")])
    enrich = domain_edge_enrichment(G)
    assert enrich["cross_share"] == pytest.approx(0.5)
    assert enrich["expected_cross_share"] == pytest.approx(1 - 420 / 1770)


def test_bimodality_coefficient_separates_one_and_two_modes():
    rng = np.random.default_rng(0)
    one = pd.Series(rng.normal(size=500))
    two = pd.Series(np.concatenate([rng.normal(-3, 0.5, 250), rng.normal(3, 0.5, 250)]))
    assert bimodality_coefficient(one) < 0.555 < bimodality_coefficient(two)


def test_cronbach_alpha_perfectly_consistent_items():
    x = pd.Series([1.0, 2, 3, 4, 5])
    assert np.isclose(cronbach_alpha(pd.DataFrame({"a": x, "b": x, "c": x})), 1.0)
