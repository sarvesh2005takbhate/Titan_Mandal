"""
build_networks.py — Construct the two complementary networks.

Network 1: Respondent Similarity Network (nodes = respondents, edges = k-NN on
           row-centred cosine similarity).
Network 2: Statement Association Network (nodes = statements, edges = signed
           Pearson correlation above a threshold).
"""

import warnings

import numpy as np
import pandas as pd
import networkx as nx
from src.data_prep import get_category

MIN_SHARED_ITEMS = 5


def distance_from_similarity(similarity: float) -> float:
    """Convert a positive similarity to a shortest-path distance (1 / similarity).

    Similarity is a strength, not a path cost. Non-positive similarities never
    connect nodes, so their distance is infinite.
    """
    if pd.isna(similarity) or similarity <= 0:
        return np.inf
    return 1.0 / float(similarity)


# ═══════════════════════════════════════════════════════════════════════════════
#  Network 1 — Respondent Similarity
# ═══════════════════════════════════════════════════════════════════════════════

def pairwise_cosine(df: pd.DataFrame, center: bool = True) -> pd.DataFrame:
    """
    Pairwise cosine similarity between rows using only statements both rows answered.

    center=True subtracts each respondent's own mean answer first. With ~80 % of
    all answers being Agree/Strongly Agree, raw cosine mostly measures *how much*
    someone agrees; centring compares *which* statements they favour.
    """
    mat = df.to_numpy(dtype=float)
    if center:
        mat = mat - np.nanmean(mat, axis=1, keepdims=True)
        # A respondent whose answers are all identical centres to the zero vector, whose
        # cosine similarity to everyone is undefined; such a respondent silently ends up
        # with no edges regardless of k. Surface it instead of letting it pass unnoticed.
        zero_variance = np.flatnonzero(np.sqrt(np.nansum(mat ** 2, axis=1)) == 0)
        if zero_variance.size:
            warnings.warn(
                "Respondent(s) " + ", ".join(str(df.index[i]) for i in zero_variance)
                + " have zero variance after centring (all answers identical or missing) and will "
                  "have no valid similarity to anyone; they become isolated nodes for any k.",
                RuntimeWarning, stacklevel=2,
            )

    presence = (~np.isnan(mat)).astype(float)
    values = np.nan_to_num(mat, nan=0.0)

    shared = presence @ presence.T
    dots = values @ values.T
    sq_norms = (values ** 2) @ presence.T          # |x_i|² over items j also answered
    denoms = np.sqrt(sq_norms) * np.sqrt(sq_norms.T)

    sim = np.full(dots.shape, np.nan)
    valid = (shared >= MIN_SHARED_ITEMS) & (denoms > 0)
    sim[valid] = dots[valid] / denoms[valid]
    np.fill_diagonal(sim, 1.0)
    return pd.DataFrame(sim, index=df.index, columns=df.index)


def knn_graph(sim: pd.DataFrame, k: int = 8) -> nx.Graph:
    """Connect every node to its k most similar (positive-similarity) neighbours."""
    G = nx.Graph()
    G.add_nodes_from(sim.index)
    for node in sim.index:
        row = sim.loc[node].drop(node).dropna()
        for nbr, w in row.nlargest(k).items():
            if w > 0:
                G.add_edge(node, nbr, weight=float(w), distance=distance_from_similarity(w))
    return G


def build_respondent_network(df: pd.DataFrame, k: int = 8, center: bool = True) -> tuple[nx.Graph, pd.DataFrame]:
    """Build the respondent k-NN similarity network. Returns (graph, similarity matrix)."""
    sim = pairwise_cosine(df, center=center)
    return knn_graph(sim, k=k), sim


# ═══════════════════════════════════════════════════════════════════════════════
#  Network 2 — Statement Association
# ═══════════════════════════════════════════════════════════════════════════════

def build_statement_network(df: pd.DataFrame, corr_threshold: float = 0.30) -> tuple[nx.Graph, pd.DataFrame]:
    """
    Nodes = statements, edges where |Pearson r| ≥ corr_threshold (pairwise complete).

    Edge attributes: weight = |r|, correlation = r, sign = ±1.
    Distance (1/r) is defined only for positive edges: a negative correlation means
    opposition and must not act as a shortcut in shortest-path metrics.
    """
    corr = df.corr(method="pearson", min_periods=10)
    G = nx.Graph()
    for code in corr.columns:
        G.add_node(code, category=get_category(code))

    codes = corr.columns
    for i, ci in enumerate(codes):
        for cj in codes[i + 1:]:
            r = corr.loc[ci, cj]
            if pd.notna(r) and abs(r) >= corr_threshold:
                G.add_edge(ci, cj, weight=abs(r), correlation=r, sign=int(np.sign(r)),
                           distance=distance_from_similarity(r))
    return G, corr


def positive_subgraph(G: nx.Graph) -> nx.Graph:
    """Keep all nodes but only positive (co-endorsement) edges."""
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))
    H.add_edges_from((u, v, d) for u, v, d in G.edges(data=True) if d.get("sign", 1) > 0)
    return H
