"""
build_networks.py — Construct the two complementary networks.

Network 1: Respondent Similarity Network (nodes = respondents, edges = cosine similarity).
Network 2: Statement Co-Endorsement Network (nodes = statements, edges = Pearson correlation).
"""

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from src.data_prep import get_category


# ═══════════════════════════════════════════════════════════════════════════════
#  Network 1 — Respondent Similarity
# ═══════════════════════════════════════════════════════════════════════════════

def _pairwise_cosine(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pairwise cosine similarity between rows, handling NaN
    by pairwise-complete observation (only columns both rows answered).
    Vectorized for high performance.
    """
    n = len(df)
    ids = df.index.tolist()
    mat = df.values.astype(float)

    nan_mask = np.isnan(mat)
    valid_mat = np.nan_to_num(mat, nan=0.0)
    presence = (~nan_mask).astype(float)

    shared_counts = presence @ presence.T
    dots = valid_mat @ valid_mat.T
    sq_norms = (valid_mat ** 2) @ presence.T

    norms_i = np.sqrt(sq_norms)
    norms_j = np.sqrt(sq_norms.T)
    denoms = norms_i * norms_j

    sim = np.full((n, n), np.nan)
    valid_pairs = (shared_counts >= 5) & (denoms > 0)
    sim[valid_pairs] = dots[valid_pairs] / denoms[valid_pairs]
    np.fill_diagonal(sim, 1.0)
    return pd.DataFrame(sim, index=ids, columns=ids)


def _pairwise_pearson(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pairwise Pearson correlation between rows (pairwise-complete).
    Vectorized for high performance.
    """
    n = len(df)
    ids = df.index.tolist()
    mat = df.values.astype(float)

    nan_mask = np.isnan(mat)
    valid_mat = np.nan_to_num(mat, nan=0.0)
    presence = (~nan_mask).astype(float)

    shared_counts = presence @ presence.T
    sum_x = valid_mat @ presence.T
    sum_y = sum_x.T

    mean_x = np.where(shared_counts >= 5, sum_x / np.maximum(shared_counts, 1), 0)
    mean_y = mean_x.T

    # sum((x - mean_x)*(y - mean_y)) = sum(x*y) - mean_y*sum(x) - mean_x*sum(y) + mean_x*mean_y*n
    dots = valid_mat @ valid_mat.T
    cov = dots - mean_y * sum_x - mean_x * sum_y + mean_x * mean_y * shared_counts

    sum_x2 = (valid_mat ** 2) @ presence.T
    var_x = sum_x2 - 2 * mean_x * sum_x + (mean_x ** 2) * shared_counts
    var_y = var_x.T

    std_x = np.sqrt(np.maximum(var_x, 0))
    std_y = np.sqrt(np.maximum(var_y, 0))
    denoms = std_x * std_y

    corr = np.full((n, n), np.nan)
    valid_pairs = (shared_counts >= 5) & (denoms > 0)
    corr[valid_pairs] = cov[valid_pairs] / denoms[valid_pairs]
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=ids, columns=ids)


def build_respondent_network(
    df: pd.DataFrame,
    k: int = 8,
    method: str = "knn",
    threshold: float | None = None,
) -> tuple[nx.Graph, pd.DataFrame, pd.DataFrame]:
    """
    Build the respondent-similarity network.

    Parameters
    ----------
    df : DataFrame  (respondents × statements, numeric with NaN)
    k : int         Number of nearest neighbors (for knn method)
    method : str    'knn' or 'threshold'
    threshold : float  Similarity cutoff (for threshold method)

    Returns
    -------
    G : networkx Graph  (undirected, weighted)
    sim_cos : DataFrame  Full cosine similarity matrix
    sim_pear : DataFrame Full Pearson matrix (robustness check)
    """
    print("  Computing pairwise cosine similarity …")
    sim_cos = _pairwise_cosine(df)
    print("  Computing pairwise Pearson correlation (robustness check) …")
    sim_pear = _pairwise_pearson(df)

    G = nx.Graph()
    ids = sim_cos.index.tolist()
    G.add_nodes_from(ids)

    if method == "knn":
        # For each node, connect to the k most similar neighbors
        for i, node_i in enumerate(ids):
            row = sim_cos.loc[node_i].drop(node_i).dropna()
            neighbors = row.nlargest(k)
            for node_j, w in neighbors.items():
                if w > 0:   # only positive similarity
                    if G.has_edge(node_i, node_j):
                        # keep max weight
                        G[node_i][node_j]["weight"] = max(G[node_i][node_j]["weight"], w)
                    else:
                        G.add_edge(node_i, node_j, weight=w)
    elif method == "threshold":
        assert threshold is not None
        for i, ni in enumerate(ids):
            for j in range(i + 1, len(ids)):
                nj = ids[j]
                w = sim_cos.iloc[i, j]
                if not np.isnan(w) and w >= threshold:
                    G.add_edge(ni, nj, weight=w)
    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"  Network 1 built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G, sim_cos, sim_pear


# ═══════════════════════════════════════════════════════════════════════════════
#  Network 2 — Statement Co-Endorsement
# ═══════════════════════════════════════════════════════════════════════════════

def build_statement_network(
    df: pd.DataFrame,
    corr_threshold: float = 0.3,
) -> tuple[nx.Graph, pd.DataFrame]:
    """
    Build the statement co-endorsement network.

    Nodes = 60 statements, edges = Pearson correlation between statement
    columns (across respondents), kept where |r| > corr_threshold.
    Edge attribute 'sign' = +1 or −1.

    Returns
    -------
    G : networkx Graph
    corr_matrix : DataFrame  full correlation matrix
    """
    print("  Computing statement-wise Pearson correlation …")
    # Use pandas .corr() which handles NaN with pairwise complete
    corr_matrix = df.corr(method="pearson", min_periods=10)

    G = nx.Graph()
    codes = corr_matrix.columns.tolist()
    for code in codes:
        G.add_node(code, category=get_category(code))

    for i, ci in enumerate(codes):
        for j in range(i + 1, len(codes)):
            cj = codes[j]
            r = corr_matrix.loc[ci, cj]
            if pd.notna(r) and abs(r) >= corr_threshold:
                G.add_edge(ci, cj, weight=abs(r), correlation=r,
                           sign=1 if r > 0 else -1)

    print(f"  Network 2 built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G, corr_matrix
