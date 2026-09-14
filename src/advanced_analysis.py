"""advanced_analysis.py — Significance, robustness and cross-domain analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.analysis import detect_communities
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
    qs = []
    for i in range(runs):
        shuffled = df.apply(lambda col: pd.Series(rng.permutation(col.to_numpy()), index=col.index))
        G, _ = build_respondent_network(shuffled, k=k)
        qs.append(detect_communities(G, seed=i)[1])
    qs = np.array(qs)
    return {
        "observed_q": observed_q,
        "null_q": qs.tolist(),
        "null_mean": round(float(qs.mean()), 4),
        "null_std": round(float(qs.std()), 4),
        "z": round(float((observed_q - qs.mean()) / qs.std()), 2),
        "p": round(float((np.sum(qs >= observed_q) + 1) / (runs + 1)), 3),
    }


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
            "similarities": vals[~np.isnan(vals)],
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


def cross_domain_share(G: nx.Graph) -> float:
    """Fraction of edges linking statements from different domains."""
    edges = list(G.edges())
    return sum(u[0] != v[0] for u, v in edges) / max(len(edges), 1)


def statement_communities(G: nx.Graph) -> tuple[dict, float, float, float]:
    """
    Louvain on the positive-edge graph (modularity needs non-negative weights),
    ignoring isolates. Returns partition, modularity, ARI vs domain labels and
    modularity of the domain labels themselves.
    """
    H = positive_subgraph(G)
    H.remove_nodes_from(list(nx.isolates(H)))
    part, q = detect_communities(H)
    ari = adjusted_rand_score([part[n] for n in H], [n[0] for n in H])
    domain_sets = [{n for n in H if n[0] == d} for d in DOMAINS]
    q_domain = nx.community.modularity(H, [s for s in domain_sets if s], weight="weight")
    return part, q, round(ari, 3), round(q_domain, 4)


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
