"""
analysis.py — Core network analysis: descriptive statistics, communities,
centrality, statement-level polarization and community interpretation.
"""

import numpy as np
import pandas as pd
import networkx as nx
from community import community_louvain        # python-louvain
from src.data_prep import DOMAINS, domain_columns, get_category


# ═══════════════════════════════════════════════════════════════════════════════
#  Descriptive Network Statistics
# ═══════════════════════════════════════════════════════════════════════════════

def descriptive_stats(G: nx.Graph) -> dict:
    """Standard statistics. Path metrics are computed on the largest component,
    in hops (interpretable) and in 1/similarity distance units."""
    degrees = [d for _, d in G.degree()]
    components = list(nx.connected_components(G))
    lcc = G.subgraph(max(components, key=len))
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": round(nx.density(G), 3),
        "avg_degree": round(float(np.mean(degrees)), 2),
        "clustering": round(nx.average_clustering(G, weight="weight"), 3),
        "clustering_unweighted": round(nx.average_clustering(G), 3),
        # A random (Erdős–Rényi) graph with the same density has expected clustering = density.
        "clustering_vs_random": round(nx.average_clustering(G) / nx.density(G), 2),
        "components": len(components),
        "isolates": sorted(nx.isolates(G)),
        "lcc_nodes": lcc.number_of_nodes(),
        "avg_path_hops": round(nx.average_shortest_path_length(lcc), 2),
        "diameter_hops": nx.diameter(lcc),
        "avg_path_distance": round(nx.average_shortest_path_length(lcc, weight="distance"), 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Community Detection (Louvain)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_communities(G: nx.Graph, seed: int | None = 42, resolution: float = 1.0) -> tuple[dict, float]:
    """Louvain partition (node → community, relabelled 1..n by size) and modularity."""
    part = community_louvain.best_partition(G, weight="weight", resolution=resolution, random_state=seed)
    order = pd.Series(part).value_counts().index          # largest community first
    relabel = {old: new for new, old in enumerate(order, start=1)}
    part = {n: relabel[c] for n, c in part.items()}
    return part, round(community_louvain.modularity(part, G, weight="weight"), 4)


# ═══════════════════════════════════════════════════════════════════════════════
#  Community Interpretation
# ═══════════════════════════════════════════════════════════════════════════════

def eta_squared(values: pd.Series, groups: pd.Series) -> float:
    """One-way ANOVA effect size: SS_between / SS_total (size-weighted)."""
    mask = values.notna() & groups.notna()
    v, g = values[mask], groups[mask]
    ss_total = ((v - v.mean()) ** 2).sum()
    if ss_total == 0:
        return 0.0
    ss_between = sum(len(x) * (x.mean() - v.mean()) ** 2 for _, x in v.groupby(g))
    return float(ss_between / ss_total)


def statement_eta_squared(df: pd.DataFrame, partition: dict) -> pd.Series:
    """How much of each statement's variance is explained by community membership."""
    groups = df.index.to_series().map(partition)
    return pd.Series({c: eta_squared(df[c], groups) for c in df.columns}).sort_values(ascending=False)


def domain_eta_squared(df: pd.DataFrame, partition: dict) -> pd.Series:
    """Mean statement-level η² per domain."""
    eta = statement_eta_squared(df, partition)
    return pd.Series({d: eta[domain_columns(df, d)].mean() for d in DOMAINS})


def community_deviation(df: pd.DataFrame, partition: dict, statements: list[str]) -> pd.DataFrame:
    """Community mean minus class mean for selected statements (rows = communities)."""
    groups = df.index.to_series().map(partition)
    means = df[statements].groupby(groups).mean()
    return means - df[statements].mean()


# ═══════════════════════════════════════════════════════════════════════════════
#  Centrality
# ═══════════════════════════════════════════════════════════════════════════════

def centrality_table(G: nx.Graph) -> pd.DataFrame:
    """Degree, weighted degree (strength) and distance-weighted betweenness."""
    return pd.DataFrame({
        "degree": dict(G.degree()),
        "strength": dict(G.degree(weight="weight")),
        "betweenness": nx.betweenness_centrality(G, weight="distance"),
    }).sort_values("betweenness", ascending=False)


# ═══════════════════════════════════════════════════════════════════════════════
#  Polarization and Consensus
# ═══════════════════════════════════════════════════════════════════════════════

def statement_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-statement distribution summary.

    Variance alone conflates two situations (a split class vs. a skewed but one-sided
    one), so polarization is measured by the split index = min(% agree, % disagree):
    it is high only when both camps are large.
    """
    n = df.notna().sum()
    out = pd.DataFrame({
        "mean": df.mean(),
        "variance": df.var(),
        "agree": (df > 0).sum() / n,
        "neutral": (df == 0).sum() / n,
        "disagree": (df < 0).sum() / n,
        "n": n,
    })
    out["split"] = out[["agree", "disagree"]].min(axis=1)
    out["category"] = [get_category(c) for c in out.index]
    return out


def polarizing_consensus_rejected(summary: pd.DataFrame, n: int = 5):
    """Most split, strongest positive consensus, and statements the class rejects (mean < 0)."""
    polar = summary.sort_values("split", ascending=False).head(n)
    consensus = summary[summary["mean"] > 0].sort_values("variance").head(n)
    rejected = summary[summary["mean"] < 0].sort_values("mean")
    return polar, consensus, rejected
