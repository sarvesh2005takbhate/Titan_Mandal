"""
analysis.py — Network analysis: descriptive stats, community detection,
centrality, polarization, and community interpretation.
"""

import numpy as np
import pandas as pd
import networkx as nx
from community import community_louvain        # python-louvain
from src.data_prep import get_category, CATEGORY_LABELS


# ═══════════════════════════════════════════════════════════════════════════════
#  Descriptive Network Statistics
# ═══════════════════════════════════════════════════════════════════════════════

def descriptive_stats(G: nx.Graph, name: str = "Network") -> dict:
    """Compute standard descriptive statistics for a network."""
    stats = {}
    stats["name"] = name
    stats["nodes"] = G.number_of_nodes()
    stats["edges"] = G.number_of_edges()
    stats["density"] = round(nx.density(G), 4)
    degrees = [d for _, d in G.degree()]
    stats["avg_degree"] = round(np.mean(degrees), 2)
    stats["clustering_coeff"] = round(nx.average_clustering(G, weight="weight"), 4)

    # Largest connected component for path-based metrics
    if nx.is_connected(G):
        lcc = G
    else:
        lcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    stats["lcc_nodes"] = lcc.number_of_nodes()
    stats["lcc_edges"] = lcc.number_of_edges()

    if lcc.number_of_nodes() > 1:
        stats["avg_path_length"] = round(nx.average_shortest_path_length(lcc), 4)
        stats["diameter"] = nx.diameter(lcc)
    else:
        stats["avg_path_length"] = None
        stats["diameter"] = None

    return stats


# ═══════════════════════════════════════════════════════════════════════════════
#  Community Detection (Louvain)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_communities(G: nx.Graph, resolution: float = 1.0) -> tuple[dict, float]:
    """
    Run Louvain community detection.

    Returns
    -------
    partition : dict  node → community_id
    modularity : float
    """
    partition = community_louvain.best_partition(G, weight="weight",
                                                  resolution=resolution,
                                                  random_state=42)
    modularity = community_louvain.modularity(partition, G, weight="weight")
    return partition, round(modularity, 4)


def community_sizes(partition: dict) -> pd.Series:
    """Return community sizes as a Series."""
    return pd.Series(partition).value_counts().sort_index().rename("size")


# ═══════════════════════════════════════════════════════════════════════════════
#  Community Interpretation
# ═══════════════════════════════════════════════════════════════════════════════

def community_opinion_profile(
    df_encoded: pd.DataFrame,
    partition: dict,
) -> pd.DataFrame:
    """
    For each community, compute mean opinion per category (T, E, S, V).

    Returns a DataFrame: index = community_id, columns = [T, E, S, V].
    """
    df = df_encoded.copy()
    df["community"] = df.index.map(partition)

    # Group columns by category
    cat_cols = {}
    for col in df.columns:
        if col == "community":
            continue
        cat = get_category(col)
        cat_cols.setdefault(cat, []).append(col)

    records = []
    for comm_id in sorted(df["community"].unique()):
        subset = df[df["community"] == comm_id]
        row = {"community": comm_id}
        for cat in ["T", "E", "S", "V"]:
            row[cat] = round(subset[cat_cols[cat]].mean().mean(), 3)
        row["n"] = len(subset)
        records.append(row)

    return pd.DataFrame(records).set_index("community")


def community_statement_means(
    df_encoded: pd.DataFrame,
    partition: dict,
) -> pd.DataFrame:
    """
    Mean response for every statement × community.
    Returns DataFrame: index=statement, columns=community_id.
    """
    df = df_encoded.copy()
    df["community"] = df.index.map(partition)
    stmt_cols = [c for c in df.columns if c != "community"]
    return df.groupby("community")[stmt_cols].mean().T


# ═══════════════════════════════════════════════════════════════════════════════
#  Centrality Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_eigenvector_centrality(G: nx.Graph, weight: str = "weight") -> dict:
    """Compute eigenvector centrality safely even if graph is disconnected."""
    eigen = {n: 0.0 for n in G.nodes()}
    if G.number_of_edges() == 0:
        return eigen
    try:
        return nx.eigenvector_centrality(G, weight=weight, max_iter=1000, tol=1e-06)
    except Exception:
        for cc in nx.connected_components(G):
            sub = G.subgraph(cc)
            if len(sub) > 1:
                try:
                    sub_eigen = nx.eigenvector_centrality(sub, weight=weight, max_iter=1000, tol=1e-06)
                    eigen.update(sub_eigen)
                except Exception:
                    pass
        return eigen


def centrality_analysis_n1(G: nx.Graph) -> pd.DataFrame:
    """Eigenvector, degree, and betweenness centrality for Network 1."""
    eigen = _safe_eigenvector_centrality(G, weight="weight")
    degree = dict(G.degree(weight="weight"))
    between = nx.betweenness_centrality(G, weight="weight")

    df = pd.DataFrame({
        "eigenvector": eigen,
        "weighted_degree": degree,
        "betweenness": between,
    })
    return df.sort_values("eigenvector", ascending=False)


def centrality_analysis_n2(G: nx.Graph) -> pd.DataFrame:
    """Betweenness and degree centrality for Network 2 (statements)."""
    between = nx.betweenness_centrality(G, weight="weight")
    degree = dict(G.degree(weight="weight"))
    eigen = _safe_eigenvector_centrality(G, weight="weight")

    df = pd.DataFrame({
        "betweenness": between,
        "weighted_degree": degree,
        "eigenvector": eigen,
    })
    df["category"] = [get_category(c) for c in df.index]
    return df.sort_values("betweenness", ascending=False)


# ═══════════════════════════════════════════════════════════════════════════════
#  Polarization Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def polarization_scores(df_encoded: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-statement variance (polarization proxy) and mean.
    High variance = polarizing; low variance = consensus.
    """
    var = df_encoded.var(skipna=True)
    mean = df_encoded.mean(skipna=True)
    count = df_encoded.count()  # non-NaN count per statement
    pol = pd.DataFrame({
        "variance": var,
        "mean": mean,
        "n_responses": count,
        "category": [get_category(c) for c in var.index],
    })
    pol = pol.sort_values("variance", ascending=False)
    return pol


def top_polarizing_consensus(pol: pd.DataFrame, n: int = 5):
    """Return top-n most polarizing and top-n most consensus statements."""
    top_polar = pol.head(n)
    top_consensus = pol.tail(n).sort_values("variance", ascending=True)
    return top_polar, top_consensus


# ═══════════════════════════════════════════════════════════════════════════════
#  Homophily / Assortativity Check
# ═══════════════════════════════════════════════════════════════════════════════

def within_vs_across_variance(
    df_encoded: pd.DataFrame,
    partition: dict,
) -> pd.DataFrame:
    """
    For each category (T/E/S/V), compute within-community vs across-community
    variance to assess homophily.
    """
    df = df_encoded.copy()
    df["community"] = df.index.map(partition)
    cat_cols = {}
    for col in df.columns:
        if col == "community":
            continue
        cat = get_category(col)
        cat_cols.setdefault(cat, []).append(col)

    records = []
    for cat in ["T", "E", "S", "V"]:
        cols = cat_cols[cat]
        # Overall variance
        overall_var = df[cols].values.flatten()
        overall_var = np.nanvar(overall_var)

        # Within-community variance (averaged across non-empty communities)
        within_vars = []
        for comm_id in sorted(df["community"].unique()):
            subset = df[df["community"] == comm_id][cols].values.flatten()
            valid = subset[~np.isnan(subset)]
            if len(valid) > 1:
                within_vars.append(np.var(valid))
        avg_within = np.mean(within_vars) if within_vars else overall_var

        records.append({
            "category": cat,
            "category_label": CATEGORY_LABELS[cat],
            "overall_variance": round(overall_var, 4),
            "avg_within_community_variance": round(avg_within, 4),
            "variance_reduction_pct": round((1 - avg_within / overall_var) * 100, 1) if overall_var > 0 else 0,
        })

    return pd.DataFrame(records)
