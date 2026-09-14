"""
visualize.py — Figures for the report.

  1. Data overview (response mix by domain, answers per respondent)
  2. Similarity distribution: raw vs row-centred cosine
  3. Respondent network coloured by community
  4. Statement network coloured by domain, negative edges highlighted
  5. Community profiles on the statements that distinguish them
  6. Polarization / consensus / rejection (diverging Likert bars)
  7. Robustness: null model, k-sensitivity, threshold sensitivity
  8. Domain structure: domain correlations and reliability
  9. Second principal component (clearest contested axis)

Figures are sized at print width (7.2 in) so fonts stay legible in the PDF;
titles are left to the report captions.
"""

import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from src.data_prep import CATEGORY_LABELS, DOMAINS

FIG_DIR = Path("output/figures")
WIDTH = 7.2

# ── Palette (validated categorical order) and chart chrome ───────────────────
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
DOMAIN_COLORS = dict(zip(DOMAINS, SERIES))
COMMUNITY_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h"]
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
LIKERT_COLORS = ["#c23b3a", "#f0a3a2", "#d3d1ca", "#86b6ef", "#2a78d6"]   # SD, D, N, A, SA
LIKERT_NAMES = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"]
DIVERGING = LinearSegmentedColormap.from_list("div", ["#c23b3a", "#f0efec", "#2a78d6"])
PERCENT = FuncFormatter(lambda x, _: f"{abs(x):.0%}")

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 220,
    "font.family": "sans-serif", "font.size": 8,
    "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 8, "axes.labelcolor": INK_2, "axes.edgecolor": AXIS,
    "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "legend.fontsize": 7.5,
})


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)


def _label(code, code_to_text, width=46):
    return f"{code}  {textwrap.shorten(code_to_text[code], width, placeholder='…')}"


def _likert_shares(series: pd.Series) -> list[float]:
    v = series.dropna()
    return [(v == x).mean() for x in (-2, -1, 0, 1, 2)]


def _diverging_bar(ax, y, shares, height=0.7):
    left = -(shares[0] + shares[1] + shares[2] / 2)
    for share, color in zip(shares, LIKERT_COLORS):
        ax.barh(y, share, left=left, color=color, height=height, edgecolor="white", linewidth=0.8)
        left += share


def _likert_legend(fig):
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in LIKERT_COLORS], labels=LIKERT_NAMES,
               ncol=5, loc="outside lower center", handlelength=1, columnspacing=1.2)


# ═══════════════════════════════════════════════════════════════════════════════

def plot_data_overview(df_all: pd.DataFrame, dropped: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.5), layout="constrained", gridspec_kw={"width_ratios": [1, 1]})

    ax = axes[0]
    kept = df_all.drop(dropped.index)
    for i, d in enumerate(DOMAINS):
        _diverging_bar(ax, i, _likert_shares(kept[[c for c in kept if c[0] == d]].stack()), height=0.62)
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.set_yticks(range(4), [CATEGORY_LABELS[d] for d in DOMAINS])
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PERCENT)
    ax.set_title("(a) Answer mix by domain")
    ax.set_xlabel("← disagree        agree →")

    ax = axes[1]
    answered = df_all.notna().sum(axis=1).sort_values(ascending=False)
    colors = [SERIES[1] if r in dropped.index else "#9ec5f4" for r in answered.index]
    ax.bar(range(len(answered)), answered.values, color=colors, width=0.85)
    ax.axhline(30, color=INK_2, lw=0.8)
    ax.text(2, 33, "inclusion cut-off = 30", fontsize=7, color=INK_2)
    ax.set_xticks([])
    ax.set_xlabel("Respondents (sorted)")
    ax.set_ylabel("Statements answered")
    ax.set_title(f"(b) Answers per respondent ({len(dropped)} excluded, orange)")
    _likert_legend(fig)
    _save(fig, "fig1_data_overview")


def plot_similarity_distribution(bias: dict):
    fig, ax = plt.subplots(figsize=(WIDTH, 1.9), layout="constrained")
    bins = np.linspace(-0.6, 1, 65)
    for label, color in (("raw", SERIES[1]), ("centred", SERIES[0])):
        vals = bias[label]["similarities"]
        ax.hist(vals, bins=bins, color=color, histtype="step", lw=1.8, label=f"{label} cosine (mean {vals.mean():.2f})")
        ax.axvline(vals.mean(), color=color, lw=1, ls=(0, (1, 2)))
    ax.set_xlabel("Pairwise respondent similarity")
    ax.set_ylabel("Pairs")
    ax.legend(loc="upper left")
    _save(fig, "fig2_similarity_distribution")


def plot_respondent_network(G: nx.Graph, partition: dict, centrality: pd.DataFrame, n_labels: int = 3):
    fig, ax = plt.subplots(figsize=(WIDTH, 3.7), layout="constrained")
    pos = nx.spring_layout(G, seed=42, weight="weight", k=1.8 / np.sqrt(len(G)), iterations=200)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=AXIS, alpha=0.5, width=0.4)
    bet = centrality["betweenness"]
    for c in sorted(set(partition.values())):
        nodes = [n for n in G if partition[n] == c]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, ax=ax, node_color=SERIES[(c - 1) % len(SERIES)],
                               node_shape=COMMUNITY_MARKERS[(c - 1) % len(COMMUNITY_MARKERS)],
                               node_size=[30 + 1200 * bet[n] for n in nodes], edgecolors="white", linewidths=0.6,
                               label=f"C{c} (n={len(nodes)})")
    for n, offset in zip(bet.head(n_labels).index, [(6, 6), (-26, 8), (6, -12)]):
        ax.annotate(f"#{n}", pos[n], xytext=offset, textcoords="offset points", fontsize=7.5, fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), markerscale=0.6, title="Community", title_fontsize=7.5)
    ax.axis("off")
    _save(fig, "fig3_respondent_network")


def plot_statement_network(G: nx.Graph):
    fig, ax = plt.subplots(figsize=(WIDTH, 3.9), layout="constrained")
    H = G.copy()
    isolates = sorted(nx.isolates(H))
    H.remove_nodes_from(isolates)
    pos = nx.kamada_kawai_layout(H, weight=None)
    pts = np.array(list(pos.values()))
    y_iso = pts[:, 1].min() - 0.2
    for x, n in zip(np.linspace(pts[:, 0].min() + 0.3, pts[:, 0].max() - 0.3, max(len(isolates), 1)), isolates):
        pos[n] = np.array([x, y_iso])
    pos_e = [(u, v) for u, v, d in H.edges(data=True) if d["sign"] > 0]
    neg_e = [(u, v) for u, v, d in H.edges(data=True) if d["sign"] < 0]
    nx.draw_networkx_edges(H, pos, edgelist=pos_e, ax=ax, edge_color=AXIS, alpha=0.6, width=0.5)
    nx.draw_networkx_edges(H, pos, edgelist=neg_e, ax=ax, edge_color=SERIES[7], width=2)
    for d in DOMAINS:
        nodes = [n for n in G if n[0] == d]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, ax=ax, node_color=DOMAIN_COLORS[d],
                               node_size=270, edgecolors="white", linewidths=0.8)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=5.3, font_color="white", font_weight="bold")
    if isolates:
        ax.text(pts[:, 0].min(), y_iso, "no edge:", fontsize=7, color=INK_2, ha="right", va="center")
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=DOMAIN_COLORS[d],
                      markersize=7, label=f"{d}: {CATEGORY_LABELS[d]}") for d in DOMAINS]
    handles += [Line2D([0], [0], color=AXIS, lw=1, label="positive r"),
                Line2D([0], [0], color=SERIES[7], lw=2, label="negative r")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.axis("off")
    _save(fig, "fig4_statement_network")


def plot_community_profiles(deviation: pd.DataFrame, eta: pd.Series, code_to_text: dict, sizes: pd.Series):
    data = deviation.T
    fig, ax = plt.subplots(figsize=(WIDTH, 0.23 * len(data) + 0.8), layout="constrained")
    lim = np.nanmax(np.abs(data.to_numpy()))
    im = ax.imshow(data.to_numpy(), cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
    for (i, j), v in np.ndenumerate(data.to_numpy()):
        ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=7, color="white" if abs(v) > 0.6 * lim else INK)
    ax.set_yticks(range(len(data)), [f"{_label(c, code_to_text, 42)} (η² {eta[c]:.2f})" for c in data.index])
    ax.set_xticks(range(data.shape[1]), [f"C{c} (n={sizes[c]})" for c in data.columns])
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.01, aspect=15)
    cb.set_label("community − class mean", fontsize=7)
    cb.outline.set_visible(False)
    _save(fig, "fig5_community_profiles")


def plot_polarization(df: pd.DataFrame, groups: dict[str, list[str]], code_to_text: dict):
    rows, labels = [], []
    for title, codes in groups.items():
        rows.append(title)
        labels.append("")
        rows += codes
        labels += [_label(c, code_to_text, 50) for c in codes]
    fig, ax = plt.subplots(figsize=(WIDTH, 0.2 * len(rows) + 0.7), layout="constrained")
    for i, c in enumerate(rows):
        if c in groups:
            ax.text(-1.0, i + 0.1, c, fontsize=7.5, fontweight="bold", va="center")
            continue
        shares = _likert_shares(df[c])
        _diverging_bar(ax, i, shares)
        ax.text(1.03, i, f"{shares[0] + shares[1]:.0%} / {shares[3] + shares[4]:.0%}", va="center", fontsize=7, color=INK_2)
    ax.text(1.03, -1, "disagree / agree", fontsize=7, color=INK_2, va="center")
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.set_yticks(range(len(rows)), labels)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(len(rows) - 0.5, -1.5)
    ax.xaxis.set_major_formatter(PERCENT)
    ax.spines["left"].set_visible(False)
    _likert_legend(fig)
    _save(fig, "fig6_polarization")


def plot_robustness(null: dict, k_sens: pd.DataFrame, thr_sens: pd.DataFrame, chosen_threshold: float):
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.3), layout="constrained")

    ax = axes[0]
    ax.hist(null["null_q"], bins=12, color="#9ec5f4")
    ax.axvline(null["observed_q"], color=SERIES[1], lw=2)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
    ax.text(0.02, 0.97, f"observed Q = {null['observed_q']:.3f} (orange)\nz = {null['z']:.1f}, p = {null['p']:.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=7)
    ax.set_xlabel("Modularity Q")
    ax.set_ylabel("Shuffled datasets")
    ax.set_title("(a) Null model")

    ax = axes[1]
    ax.plot(k_sens["k"], k_sens["modularity"], marker="o", ms=4, color=SERIES[0], lw=1.8, label="modularity Q")
    others = k_sens[k_sens["ari_vs_k8"] < 1]
    ax.plot(others["k"], others["ari_vs_k8"], marker="s", ms=5, color=SERIES[1], lw=0, label="ARI vs k = 8")
    ax.set_ylim(0, 1)
    ax.set_xlabel("k")
    ax.set_title("(b) Respondent network vs k")
    ax.legend(loc="upper right")
    ax.grid(axis="y")

    ax = axes[2]
    ax.plot(thr_sens["threshold"], thr_sens["edges"], marker="o", ms=4, color=SERIES[0], lw=1.8, label="edges")
    ax.plot(thr_sens["threshold"], thr_sens["expected_false_edges"].clip(lower=0.1), marker="s", ms=4,
            color=SERIES[1], lw=1.8, label="expected chance edges")
    ax.set_yscale("log")
    ax.axvline(chosen_threshold, color=AXIS, lw=1)
    ax.set_xlabel("|r| threshold")
    ax.set_title("(c) Statement network vs threshold")
    ax.set_ylim(0.05, 50000)
    ax.legend(loc="upper right")
    ax.grid(axis="y")
    _save(fig, "fig7_robustness")


def plot_domain_structure(corr: pd.DataFrame, alpha: pd.Series):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.2), layout="constrained", gridspec_kw={"width_ratios": [1, 1.1]})
    ax = axes[0]
    m = corr.to_numpy().copy()
    np.fill_diagonal(m, np.nan)
    ax.imshow(m, cmap=LinearSegmentedColormap.from_list("seq", ["#e6f0fc", "#104281"]), vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(corr.to_numpy()):
        if i != j:
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color="white" if v > 0.55 else INK)
    ax.set_xticks(range(4), DOMAINS)
    ax.set_yticks(range(4), [CATEGORY_LABELS[d] for d in DOMAINS])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("(a) Correlation of domain scores")

    ax = axes[1]
    ax.barh(range(4), alpha.values, color=SERIES[0], height=0.55)
    ax.set_yticks(range(4), [CATEGORY_LABELS[d] for d in DOMAINS])
    ax.axvline(0.7, color=INK_2, lw=0.8)
    ax.text(0.72, -0.62, "0.70 acceptable", fontsize=7, color=INK_2, va="center")
    for i, v in enumerate(alpha.values):
        ax.text(v - 0.015, i, f"{v:.2f}", va="center", ha="right", fontsize=7.5, color="white", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(3.5, -0.9)
    ax.tick_params(axis="y", length=0)
    ax.set_title("(b) Cronbach's α")
    _save(fig, "fig8_domain_structure")


def plot_pc2(loadings: pd.Series, code_to_text: dict, n: int = 6):
    top = pd.concat([loadings.nsmallest(n), loadings.nlargest(n).iloc[::-1]])
    fig, ax = plt.subplots(figsize=(WIDTH, 2.5), layout="constrained")
    colors = [SERIES[0] if v < 0 else SERIES[7] for v in top.values]
    ax.barh(range(len(top)), top.values, color=colors, height=0.65)
    ax.set_yticks(range(len(top)), [_label(c, code_to_text, 55) for c in top.index])
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("PC2 loading   (blue: open / flexible learning pole · red: regulation / structure pole)")
    _save(fig, "fig9_pc2_loadings")
