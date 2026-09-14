"""Advanced robustness and cross-domain analyses for the opinion network project."""

from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score

from src.data_prep import CATEGORY_LABELS, get_category


def respondent_k_sensitivity(df_encoded: pd.DataFrame, ks=(5, 6, 8, 10, 12)) -> pd.DataFrame:
    """Evaluate respondent-network structure across several k values."""
    from src.build_networks import build_respondent_network
    from src.analysis import detect_communities

    rows = []
    for k in ks:
        G, _, _ = build_respondent_network(df_encoded, k=k, method="knn")
        partition, modularity = detect_communities(G, seed=42)
        rows.append({
            "k": k,
            "edges": G.number_of_edges(),
            "density": round(nx.density(G), 4),
            "avg_degree": round(np.mean([d for _, d in G.degree()]), 2),
            "clustering": round(nx.average_clustering(G, weight="weight"), 4),
            "components": nx.number_connected_components(G),
            "lcc_nodes": max((len(c) for c in nx.connected_components(G)), default=0),
            "modularity": round(modularity, 4),
            "n_communities": len(set(partition.values())),
        })
    return pd.DataFrame(rows)


def statement_threshold_sensitivity(df_encoded: pd.DataFrame, thresholds=(0.20, 0.30, 0.40, 0.50)) -> pd.DataFrame:
    """Evaluate statement-network structure across Pearson thresholds."""
    from src.build_networks import build_statement_network

    rows = []
    for threshold in thresholds:
        G, _ = build_statement_network(df_encoded, corr_threshold=threshold)
        pos_edges = sum(1 for _, _, d in G.edges(data=True) if d.get("sign", 1) > 0)
        neg_edges = sum(1 for _, _, d in G.edges(data=True) if d.get("sign", 1) < 0)
        lcc = G if nx.is_connected(G) else G.subgraph(max(nx.connected_components(G), key=len)).copy()
        rows.append({
            "threshold": threshold,
            "edges": G.number_of_edges(),
            "density": round(nx.density(G), 4),
            "positive_edges": pos_edges,
            "negative_edges": neg_edges,
            "lcc_size": lcc.number_of_nodes(),
            "clustering": round(nx.average_clustering(G, weight="weight"), 4),
            "components": nx.number_connected_components(G),
        })
    return pd.DataFrame(rows)


def signed_network_summary(G: nx.Graph) -> dict:
    """Summarize the signed statement network by positive and negative association."""
    edges = list(G.edges(data=True))
    positive = [d for _, _, d in edges if d.get("sign", 1) > 0]
    negative = [d for _, _, d in edges if d.get("sign", 1) < 0]

    pos_df = pd.DataFrame([
        {"u": u, "v": v, "correlation": d.get("correlation", 0.0), "abs_correlation": abs(float(d.get("correlation", 0.0)))}
        for u, v, d in edges if d.get("sign", 1) > 0
    ])
    neg_df = pd.DataFrame([
        {"u": u, "v": v, "correlation": d.get("correlation", 0.0), "abs_correlation": abs(float(d.get("correlation", 0.0)))}
        for u, v, d in edges if d.get("sign", 1) < 0
    ])

    return {
        "positive_edges": len(positive),
        "negative_edges": len(negative),
        "fraction_positive": round(len(positive) / max(len(edges), 1), 4),
        "fraction_negative": round(len(negative) / max(len(edges), 1), 4),
        "strongest_positive": pos_df.sort_values("abs_correlation", ascending=False).head(10),
        "strongest_negative": neg_df.sort_values("abs_correlation", ascending=False).head(10),
    }


def domain_correlation_matrix(df_encoded: pd.DataFrame, domains=("T", "E", "S", "V")) -> pd.DataFrame:
    """Compute the 4x4 correlation matrix of domain mean scores."""
    scores = {}
    for domain in domains:
        cols = [c for c in df_encoded.columns if c.startswith(domain)]
        scores[domain] = df_encoded[cols].mean(axis=1, skipna=True)
    matrix = pd.DataFrame(index=list(domains), columns=list(domains), dtype=float)
    for d1 in domains:
        for d2 in domains:
            matrix.loc[d1, d2] = scores[d1].corr(scores[d2])
    return matrix


def domain_level_statement_summary(df_encoded: pd.DataFrame, corr_matrix: pd.DataFrame) -> pd.DataFrame:
    """Return a per-domain summary of association structure."""
    rows = []
    for domain in ["T", "E", "S", "V"]:
        cols = [c for c in corr_matrix.columns if c.startswith(domain)]
        sub = corr_matrix.loc[cols, cols].copy()
        upper = sub.where(~np.triu(np.ones(sub.shape), k=1).astype(bool))
        values = upper.stack().dropna().to_numpy()
        rows.append({
            "domain": domain,
            "within_domain_mean_abs_r": round(float(np.mean(np.abs(values))), 4),
            "within_domain_mean_r": round(float(np.mean(values)), 4),
            "n_within_edges": int(len(values)),
        })
    return pd.DataFrame(rows)


def within_vs_across_statement_edges(G: nx.Graph) -> pd.DataFrame:
    """Compare within-domain and cross-domain edges in the statement network."""
    domain_map = {n: n[0] for n in G.nodes()}
    all_edges = []
    for u, v, d in G.edges(data=True):
        all_edges.append({
            "u": u,
            "v": v,
            "abs_r": abs(float(d.get("correlation", 0.0))),
            "signed_r": float(d.get("correlation", 0.0)),
            "within": domain_map.get(u) == domain_map.get(v),
        })

    within = [e for e in all_edges if e["within"]]
    cross = [e for e in all_edges if not e["within"]]

    def summarize(edges):
        if not edges:
            return {"count": 0, "prop": 0.0, "mean_abs_r": np.nan, "mean_signed_r": np.nan}
        df = pd.DataFrame(edges)
        return {
            "count": len(df),
            "prop": round(len(df) / max(len(all_edges), 1), 4),
            "mean_abs_r": round(float(df["abs_r"].mean()), 4),
            "mean_signed_r": round(float(df["signed_r"].mean()), 4),
        }

    summary = {
        "within_domain": summarize(within),
        "cross_domain": summarize(cross),
    }
    return pd.DataFrame.from_dict(summary, orient="index")


def top_statement_correlations(corr_matrix: pd.DataFrame, n: int = 10):
    """Return top positive and negative statement correlations."""
    records = []
    for i, code_i in enumerate(corr_matrix.columns):
        for j in range(i + 1, len(corr_matrix.columns)):
            code_j = corr_matrix.columns[j]
            r = corr_matrix.loc[code_i, code_j]
            if pd.notna(r):
                records.append({
                    "code_i": code_i,
                    "code_j": code_j,
                    "corr": float(r),
                    "domain_i": code_i[0],
                    "domain_j": code_j[0],
                })
    pos = pd.DataFrame(records).sort_values("corr", ascending=False).head(n)
    neg = pd.DataFrame(records).sort_values("corr", ascending=True).head(n)
    return pos, neg


def community_stability_analysis(df_encoded: pd.DataFrame, k: int = 8, runs: int = 12) -> dict:
    """Assess Louvain stability across repeated runs."""
    from src.build_networks import build_respondent_network
    from src.analysis import detect_communities

    partitions = []
    modularities = []
    for seed in range(runs):
        G, _, _ = build_respondent_network(df_encoded, k=k, method="knn")
        partition, modularity = detect_communities(G, seed=seed)
        partitions.append(partition)
        modularities.append(modularity)

    pairwise = []
    for i in range(len(partitions)):
        for j in range(i + 1, len(partitions)):
            pairwise.append(normalized_mutual_info_score(list(partitions[i].values()), list(partitions[j].values())))
    stability = float(np.mean(pairwise)) if pairwise else 1.0
    return {
        "modularity_mean": round(float(np.mean(modularities)), 4),
        "modularity_std": round(float(np.std(modularities)), 4),
        "community_count_mean": round(float(np.mean([len(set(p.values())) for p in partitions])), 4),
        "community_count_std": round(float(np.std([len(set(p.values())) for p in partitions])), 4),
        "stability_nmi": round(stability, 4),
    }


def eta_squared_explained_variance(df_encoded: pd.DataFrame, partition: dict) -> pd.DataFrame:
    """Compute between-community explained variance by domain."""
    df = df_encoded.copy()
    df["community"] = df.index.map(partition)
    cat_cols = {cat: [] for cat in ["T", "E", "S", "V"]}
    for col in df.columns:
        if col == "community":
            continue
        cat_cols.setdefault(get_category(col), []).append(col)

    records = []
    for cat in ["T", "E", "S", "V"]:
        values = df[cat_cols[cat]].to_numpy().flatten()
        valid = values[~np.isnan(values)]
        total_var = float(np.var(valid)) if len(valid) > 1 else 0.0
        if total_var <= 0:
            eta2 = 0.0
            within_var = 0.0
            between_var = 0.0
        else:
            community_means = []
            within_values = []
            for comm_id in sorted(df["community"].dropna().unique()):
                subset = df[df["community"] == comm_id][cat_cols[cat]].to_numpy().flatten()
                valid_subset = subset[~np.isnan(subset)]
                if len(valid_subset) > 1:
                    community_means.append(np.mean(valid_subset))
                    within_values.append(np.var(valid_subset))
            if community_means:
                between_var = float(np.var(community_means, ddof=0))
                within_var = float(np.mean(within_values)) if within_values else 0.0
                eta2 = between_var / total_var
            else:
                between_var = 0.0
                within_var = total_var
                eta2 = 0.0
        records.append({
            "category": cat,
            "category_label": CATEGORY_LABELS[cat],
            "total_variance": round(total_var, 4),
            "within_community_variance": round(within_var, 4),
            "between_community_variance": round(between_var, 4),
            "eta_squared": round(float(eta2), 4),
        })
    return pd.DataFrame(records)


def polarization_vs_network_position(statement_graph: nx.Graph, pol: pd.DataFrame):
    """Compare statement variance with network centrality measures."""
    degree = dict(statement_graph.degree(weight="weight"))
    betweenness = nx.betweenness_centrality(statement_graph, weight="distance")
    combined = pol[["variance", "mean", "category"]].copy()
    combined["weighted_degree"] = [degree.get(code, 0.0) for code in combined.index]
    combined["betweenness"] = [betweenness.get(code, 0.0) for code in combined.index]
    return combined.sort_values("variance", ascending=False)


def statement_network_robustness_details(df_encoded: pd.DataFrame):
    """Return a compact table of statement network robustness measures."""
    from src.build_networks import build_statement_network

    results = []
    for threshold in (0.20, 0.30, 0.40, 0.50):
        G, _ = build_statement_network(df_encoded, corr_threshold=threshold)
        pos = sum(1 for _, _, d in G.edges(data=True) if d.get("sign", 1) > 0)
        neg = sum(1 for _, _, d in G.edges(data=True) if d.get("sign", 1) < 0)
        lcc = G if nx.is_connected(G) else G.subgraph(max(nx.connected_components(G), key=len)).copy()
        results.append({
            "threshold": threshold,
            "edges": G.number_of_edges(),
            "density": round(nx.density(G), 4),
            "positive_edges": pos,
            "negative_edges": neg,
            "lcc_size": lcc.number_of_nodes(),
            "clustering": round(nx.average_clustering(G, weight="weight"), 4),
            "components": nx.number_connected_components(G),
        })
    return pd.DataFrame(results)
