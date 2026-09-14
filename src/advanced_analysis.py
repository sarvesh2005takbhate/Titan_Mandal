"""advanced_analysis.py — Significance, robustness and cross-domain analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.analysis import detect_communities, eta_squared
from src.build_networks import build_respondent_network, build_statement_network, knn_graph, positive_subgraph
from src.data_prep import DOMAINS, domain_columns


def _ari(p1: dict, p2: dict) -> float:
    common = [n for n in p1 if n in p2]
    return adjusted_rand_score([p1[n] for n in common], [p2[n] for n in common])


# ═══════════════════════════════════════════════════════════════════════════════
#  Respondent network: is the community structure real?
# ═══════════════════════════════════════════════════════════════════════════════

def modularity_null_model(df: pd.DataFrame, observed_q: float, k: int = 8, runs: int = 50, seed: int = 0) -> dict:
    """
    Compare observed modularity with networks built from shuffled data.

    Each statement's answers are permuted independently across respondents: item
    distributions are kept, but any consistent opinion profile is destroyed. k-NN
    graphs are modular by construction, so this is the relevant baseline.
    """
    rng = np.random.default_rng(seed)
    qs, clustering, pc2_eta = [], [], []
    for i in range(runs):
        shuffled = shuffle_within_statements(df, rng)
        G, _ = build_respondent_network(shuffled, k=k)
        part, q = detect_communities(G, seed=i)
        qs.append(q)
        clustering.append(nx.average_clustering(G))
        # How much a partition of *structureless* data still "explains" its own PC2.
        scores = principal_components(shuffled, 2)["scores"]["PC2"]
        pc2_eta.append(eta_squared(scores, shuffled.index.to_series().map(part)))
    qs = np.array(qs)
    return {
        "observed_q": observed_q,
        "null_q": qs.tolist(),
        "null_mean": round(float(qs.mean()), 4),
        "null_std": round(float(qs.std()), 4),
        "z": round(float((observed_q - qs.mean()) / qs.std()), 2),
        "p": round(float((np.sum(qs >= observed_q) + 1) / (runs + 1)), 3),
        "null_clustering_mean": float(np.mean(clustering)),
        "null_clustering_std": float(np.std(clustering)),
        "null_pc2_eta_mean": float(np.mean(pc2_eta)),
    }


def shuffle_within_statements(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Permute each statement's answers across respondents independently."""
    return df.apply(lambda col: pd.Series(rng.permutation(col.to_numpy()), index=col.index))


def bimodality_coefficient(x: pd.Series) -> float:
    """Sarle's bimodality coefficient; values above 0.555 suggest two modes (camps)."""
    x = pd.Series(x).dropna()
    n = len(x)
    g, k = stats.skew(x), stats.kurtosis(x)
    return float((g ** 2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


def community_stability(G: nx.Graph, df: pd.DataFrame, partition: dict, k: int = 8,
                        runs: int = 20, subsample: float = 0.9, seed: int = 0) -> dict:
    """
    Two stability checks against the main partition:
      * seed stability — same graph, different Louvain seeds
      * data stability — rebuild the network from a random 90 % of respondents
    """
    rng = np.random.default_rng(seed)
    seed_ari = [_ari(partition, detect_communities(G, seed=s)[0]) for s in range(100, 100 + runs)]
    data_ari = []
    for s in range(runs):
        keep = rng.choice(df.index, size=int(subsample * len(df)), replace=False)
        Gs, _ = build_respondent_network(df.loc[keep], k=k)
        data_ari.append(_ari(partition, detect_communities(Gs, seed=s)[0]))
    return {
        "seed_ari_mean": round(float(np.mean(seed_ari)), 3),
        "subsample_ari_mean": round(float(np.mean(data_ari)), 3),
        "subsample_ari_std": round(float(np.std(data_ari)), 3),
    }


def agreement_bias_check(df: pd.DataFrame, k: int = 8) -> dict:
    """
    Raw vs row-centred cosine: how strongly does network position track a respondent's
    overall agreement level, and do the two versions give the same communities?
    """
    mean_answer = df.mean(axis=1)
    out = {}
    parts = {}
    for label, center in (("raw", False), ("centred", True)):
        G, sim = build_respondent_network(df, k=k, center=center)
        parts[label] = detect_communities(G)[0]
        vals = sim.to_numpy()[np.triu_indices(len(sim), 1)]
        out[label] = {
            "mean_similarity": round(float(np.nanmean(vals)), 3),
            "degree_vs_mean_answer_r": round(float(pd.Series(dict(G.degree())).corr(mean_answer)), 3),
        }
    out["partition_ari"] = round(_ari(parts["raw"], parts["centred"]), 3)
    return out


def respondent_k_sensitivity(df: pd.DataFrame, main_partition: dict, ks=(5, 6, 8, 10, 12)) -> pd.DataFrame:
    """Network structure and agreement with the main (k=8) partition across k."""
    rows = []
    for k in ks:
        G, _ = build_respondent_network(df, k=k)
        part, q = detect_communities(G)
        rows.append({
            "k": k,
            "edges": G.number_of_edges(),
            "clustering": round(nx.average_clustering(G, weight="weight"), 3),
            "modularity": q,
            "communities": len(set(part.values())),
            "ari_vs_k8": round(_ari(main_partition, part), 3),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
#  Statement network
# ═══════════════════════════════════════════════════════════════════════════════

def statement_threshold_sensitivity(df: pd.DataFrame, thresholds=(0.20, 0.25, 0.30, 0.35, 0.40, 0.50)) -> pd.DataFrame:
    """Edges, signs, fragmentation and expected false-positive edges per |r| threshold."""
    n_pairs = df.shape[1] * (df.shape[1] - 1) // 2
    n_obs = int(df.notna().sum().median())
    rows = []
    for t in thresholds:
        G, _ = build_statement_network(df, corr_threshold=t)
        tstat = t * np.sqrt((n_obs - 2) / (1 - t ** 2))
        p = 2 * stats.t.sf(tstat, n_obs - 2)
        rows.append({
            "threshold": t,
            "edges": G.number_of_edges(),
            "negative_edges": sum(1 for *_, d in G.edges(data=True) if d["sign"] < 0),
            "isolates": nx.number_of_isolates(G),
            "components": nx.number_connected_components(G),
            "p_value": p,
            "expected_false_edges": round(n_pairs * p, 1),
        })
    return pd.DataFrame(rows)


def negative_edges(G: nx.Graph) -> pd.DataFrame:
    """All negative (opposition) edges, strongest first."""
    rows = [{"u": u, "v": v, "r": d["correlation"]} for u, v, d in G.edges(data=True) if d["sign"] < 0]
    return pd.DataFrame(rows, columns=["u", "v", "r"]).sort_values("r")


def correlation_tests(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """p-values for every statement pair with Benjamini–Hochberg (FDR) and Bonferroni flags."""
    cols = df.columns
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            pair = df[[a, b]].dropna()
            r, p = stats.pearsonr(pair[a], pair[b])
            rows.append({"u": a, "v": b, "r": r, "p": p})
    out = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
    m = len(out)
    passed = out["p"] <= alpha * (np.arange(1, m + 1) / m)
    out["fdr"] = np.arange(m) < (passed[::-1].idxmax() + 1 if passed.any() else 0)
    out["bonferroni"] = out["p"] < alpha / m
    return out


def bootstrap_edge_ci(df: pd.DataFrame, edges: pd.DataFrame, runs: int = 1000, seed: int = 0) -> pd.DataFrame:
    """95 % bootstrap confidence interval (resampling respondents) for each listed correlation."""
    rng = np.random.default_rng(seed)
    samples = {(u, v): [] for u, v in zip(edges["u"], edges["v"])}
    for _ in range(runs):
        boot = df.loc[rng.choice(df.index, len(df), replace=True)]
        for u, v in samples:
            samples[(u, v)].append(boot[u].corr(boot[v]))
    out = edges.copy()
    out["ci_low"] = [np.percentile(samples[(u, v)], 2.5) for u, v in zip(edges["u"], edges["v"])]
    out["ci_high"] = [np.percentile(samples[(u, v)], 97.5) for u, v in zip(edges["u"], edges["v"])]
    return out


def domain_edge_enrichment(G: nx.Graph, domain_size: int = 15) -> dict:
    """Observed vs expected within-domain edges if edges ignored domain membership."""
    n = G.number_of_nodes()
    possible_within = len(DOMAINS) * domain_size * (domain_size - 1) / 2
    expected_share = possible_within / (n * (n - 1) / 2)
    within = sum(u[0] == v[0] for u, v in G.edges())
    edges = G.number_of_edges()
    return {
        "cross_share": 1 - within / edges,
        "expected_cross_share": 1 - expected_share,
        "within_enrichment": within / (edges * expected_share),
    }


def rewired_null(H: nx.Graph, runs: int = 30, seed: int = 0) -> dict:
    """Degree-preserving rewiring: clustering and (unweighted) modularity expected from degrees alone."""
    base = nx.Graph()
    base.add_nodes_from(H.nodes())
    base.add_edges_from(H.edges())
    q_obs = detect_communities(base, seed=seed)[1]
    clust, qs = [], []
    for s in range(runs):
        R = base.copy()
        nx.double_edge_swap(R, nswap=5 * R.number_of_edges(), max_tries=10 ** 6, seed=seed + s)
        clust.append(nx.average_clustering(R))
        qs.append(detect_communities(R, seed=s)[1])
    return {
        "clustering": nx.average_clustering(base),
        "null_clustering": float(np.mean(clust)),
        "q_unweighted": q_obs,
        "null_q_mean": float(np.mean(qs)),
        "null_q_std": float(np.std(qs)),
    }


def statement_robust_cores(df: pd.DataFrame, threshold: float, runs: int = 200,
                           min_coassign: float = 0.6, min_size: int = 3, seed: int = 0) -> dict:
    """
    Bootstrap consensus for statement clusters.

    Modularity has many near-optimal partitions, so a single Louvain run on one sample is
    not trustworthy. The positive-edge statement network is rebuilt on `runs` bootstrap
    resamples of respondents and clustered each time. A *robust core* is a connected group
    (≥ `min_size`) of statements that land in the same cluster in ≥ `min_coassign` of the
    resamples. Statements outside cores have no stable cluster.
    """
    rng = np.random.default_rng(seed)
    cols = list(df.columns)
    together = pd.DataFrame(0.0, index=cols, columns=cols)
    present = pd.DataFrame(0.0, index=cols, columns=cols)
    single_runs = []
    for s in range(runs):
        boot = df.loc[rng.choice(df.index, len(df), replace=True)].reset_index(drop=True)
        G, _ = build_statement_network(boot, threshold)
        H = positive_subgraph(G)
        H.remove_nodes_from(list(nx.isolates(H)))
        part, _ = detect_communities(H, seed=s)
        nodes = list(part)
        labels = np.array([part[n] for n in nodes])
        together.loc[nodes, nodes] += labels[:, None] == labels[None, :]
        present.loc[nodes, nodes] += 1
        single_runs.append(part)
    co = together / present.replace(0, np.nan)

    C = nx.Graph()
    C.add_edges_from((a, b) for i, a in enumerate(cols) for b in cols[i + 1:] if co.loc[a, b] >= min_coassign)
    cores = []
    for members in nx.connected_components(C):
        if len(members) >= min_size:
            members = sorted(members)
            rates = co.loc[members, members].to_numpy()[np.triu_indices(len(members), 1)]
            cores.append({"members": members, "coassign": float(np.nanmean(rates))})
    cores.sort(key=lambda c: -len(c["members"]))
    pair_ari = [_ari(single_runs[i], single_runs[i + 1]) for i in range(0, runs - 1, 2)]
    return {
        "cores": cores,
        "overall_coassign": float(np.nanmean(co.to_numpy()[np.triu_indices(len(cols), 1)])),
        "single_run_ari": float(np.mean(pair_ari)),
        "runs": runs,
        "min_coassign": min_coassign,
    }


def parallel_analysis(df: pd.DataFrame, n_components: int = 6, runs: int = 100, seed: int = 0) -> dict:
    """Horn's parallel analysis: compare explained variance with shuffled-data PCA (95th percentile)."""
    rng = np.random.default_rng(seed)
    observed = np.array(principal_components(df, n_components)["explained"])
    null = np.array([principal_components(shuffle_within_statements(df, rng), n_components)["explained"]
                     for _ in range(runs)])
    threshold = np.percentile(null, 95, axis=0)
    above = observed > threshold
    return {
        "observed": observed.tolist(),
        "null_95": threshold.tolist(),
        "n_above_noise": int(np.argmin(above)) if not above.all() else n_components,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Domain structure and latent dimensions
# ═══════════════════════════════════════════════════════════════════════════════

def domain_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Per-respondent mean answer per domain."""
    return pd.DataFrame({d: df[domain_columns(df, d)].mean(axis=1) for d in DOMAINS})


def cronbach_alpha(items: pd.DataFrame) -> float:
    """Internal consistency of a set of items (complete cases)."""
    X = items.dropna()
    k = X.shape[1]
    return float(k / (k - 1) * (1 - X.var().sum() / X.sum(axis=1).var()))


def domain_reliability(df: pd.DataFrame) -> pd.Series:
    return pd.Series({d: round(cronbach_alpha(df[domain_columns(df, d)]), 3) for d in DOMAINS})


def principal_components(df: pd.DataFrame, n_components: int = 2) -> dict:
    """
    PCA on complete-ish data (column-mean imputation for the few No Comments).
    Loadings are sign-aligned so that PC1 points towards overall agreement.
    """
    X = df.fillna(df.mean())
    Xc = X - X.mean()
    _, s, vt = np.linalg.svd(Xc.to_numpy(), full_matrices=False)
    explained = s ** 2 / np.sum(s ** 2)
    loadings = pd.DataFrame(vt[:n_components].T, index=df.columns,
                            columns=[f"PC{i + 1}" for i in range(n_components)])
    scores = pd.DataFrame(Xc.to_numpy() @ vt[:n_components].T, index=df.index, columns=loadings.columns)
    if loadings["PC1"].sum() < 0:
        loadings["PC1"] *= -1
        scores["PC1"] *= -1
    mean_answer = df.mean(axis=1)
    return {
        "explained": explained[:n_components].round(3).tolist(),
        "loadings": loadings,
        "scores": scores,
        "pc1_same_sign_share": round(float((loadings["PC1"] > 0).mean()), 2),
        "pc1_vs_mean_answer_r": round(float(scores["PC1"].corr(mean_answer)), 3),
    }


def item_pair_correlations(corr: pd.DataFrame, pairs) -> pd.DataFrame:
    """Correlations for hand-picked pairs of near-duplicate statements in different domains."""
    return pd.DataFrame([{"u": a, "v": b, "r": corr.loc[a, b]} for a, b in pairs])


# ═══════════════════════════════════════════════════════════════════════════════
#  Evidence for design choices (chosen method vs the obvious alternative)
# ═══════════════════════════════════════════════════════════════════════════════

def design_choice_evidence(df_all: pd.DataFrame, df: pd.DataFrame, G: nx.Graph, sim: pd.DataFrame,
                           partition: dict, summary: pd.DataFrame, seed: int = 0) -> dict:
    """Quantify what each methodological choice changes on this dataset."""
    ev = {}

    # Missing as Neutral would invent opinions.
    ev["neutral_share_true"] = float((df_all == 0).sum().sum() / df_all.notna().sum().sum())
    ev["neutral_share_if_blank_neutral"] = float((df_all.fillna(0) == 0).sum().sum() / df_all.size)

    # k-NN vs a global threshold giving the same number of edges.
    iu = np.triu_indices(len(sim), 1)
    values = sim.to_numpy()[iu]
    cut = np.sort(values[~np.isnan(values)])[::-1][G.number_of_edges() - 1]
    Gt = nx.Graph()
    Gt.add_nodes_from(sim.index)
    Gt.add_edges_from((sim.index[i], sim.index[j]) for i, j in zip(*iu) if sim.iat[i, j] >= cut)
    ev["threshold_isolates"] = nx.number_of_isolates(Gt)
    ev["threshold_components"] = nx.number_connected_components(Gt)
    ev["min_connected_k"] = next(k for k in range(1, 20) if nx.is_connected(knn_graph(sim, k)))

    # 1/similarity vs 1 − similarity as path cost.
    H = G.copy()
    for _, _, d in H.edges(data=True):
        d["alt"] = 1 - d["weight"]
    b_inv = nx.betweenness_centrality(G, weight="distance")
    b_alt = nx.betweenness_centrality(H, weight="alt")
    ev["betweenness_rank_rho_inv_vs_one_minus"] = float(stats.spearmanr([b_inv[n] for n in G], [b_alt[n] for n in G])[0])

    # Louvain vs other community algorithms.
    def as_partition(communities):
        return {n: i for i, c in enumerate(communities) for n in c}
    greedy = list(nx.community.greedy_modularity_communities(G, weight="weight"))
    lpa = list(nx.community.asyn_lpa_communities(G, weight="weight", seed=seed))
    ev["greedy_q"] = float(nx.community.modularity(G, greedy, weight="weight"))
    ev["greedy_n"] = len(greedy)
    ev["greedy_ari"] = float(_ari(partition, as_partition(greedy)))
    ev["lpa_n"] = len(lpa)

    # ARI vs NMI for unrelated partitions of the same sizes.
    rng = np.random.default_rng(seed)
    labels = np.array([partition[n] for n in G])
    shuffled = [rng.permutation(labels) for _ in range(200)]
    ev["random_nmi"] = float(np.mean([normalized_mutual_info_score(labels, p) for p in shuffled]))
    ev["random_ari"] = float(np.mean([adjusted_rand_score(labels, p) for p in shuffled]))

    # Centrality measures vs overall agreement level.
    mean_answer = df.mean(axis=1)
    eig = nx.eigenvector_centrality(G, weight="weight", max_iter=2000)
    ev["eigenvector_vs_agreement_r"] = float(pd.Series(eig).corr(mean_answer))
    ev["betweenness_vs_agreement_r"] = float(pd.Series(b_inv).corr(mean_answer))

    # Pearson vs Spearman for statements.
    ju = np.triu_indices(df.shape[1], 1)
    pear = df.corr(min_periods=10).to_numpy()[ju]
    spear = df.corr(method="spearman", min_periods=10).to_numpy()[ju]
    ev["pearson_spearman_r"] = float(np.corrcoef(pear, spear)[0, 1])

    # Variance vs split index.
    ev["variance_rank"] = summary["variance"].rank(ascending=False).astype(int).to_dict()
    return ev
