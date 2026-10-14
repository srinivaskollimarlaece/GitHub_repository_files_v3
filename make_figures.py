"""Publication figures for the paper, redrawn from results/*.csv (no re-simulation).
Usage:  python make_figures.py   (expects results/ next to this file; writes figures/)"""
import os, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "figures"); os.makedirs(OUT, exist_ok=True)

# ---- style: validated categorical slots 1-3 + neutrals ---------------------------------
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#1f2328", "#52514e", "#e6e6e3"
GREY1, GREY2, GREY3 = "#b9b8b2", "#8f8e88", "#6b6a65"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.edgecolor": INK2, "axes.linewidth": 0.7, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300, "savefig.bbox": "tight",
    "lines.linewidth": 1.8, "legend.frameon": False})

PROP, BASE = "Proposed (full)", "Base SQP (Alnakhli 2024)"
SERIES = [  # method, label, colour, marker, linestyle
    (PROP, "Proposed (full)", BLUE, "o", "-"),
    ("SE-aware + alternating", "SE-aware + alternating", AQUA, "D", "-"),
    (BASE, "Base SQP [2]", ORANGE, "s", "--"),
    ("Equal MIS bandwidth", "Equal MIS bandwidth", GREY3, "^", ":"),
    ("Random Selection", "Random selection", GREY1, "v", "-."),
]
s = pd.read_csv(os.path.join(RES, "summary_stats.csv")); main = s[s.exp == "main"]
DENS = [0.5, 0.8, 1.0]

def series_panels(metric, ylabel, fname, skip=()):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    sel = main[~main.method.isin(skip) & main.method.isin([x[0] for x in SERIES])]
    top = (sel[f"{metric}_mean"] + sel[f"{metric}_ci95"]).max() * 1.06
    for ax, d, tag in zip(axes, DENS, "abc"):
        for m, lab, c, mk, ls in SERIES:
            if m in skip: continue
            g = main[(main.density == d) & (main.method == m)].sort_values("N")
            ax.errorbar(g.N, g[f"{metric}_mean"], yerr=g[f"{metric}_ci95"], color=c, marker=mk, ms=4.5,
                        ls=ls, lw=1.8 if m == PROP else 1.4, capsize=2, elinewidth=0.8,
                        mec="white", mew=0.6, zorder=3 if m == PROP else 2)
        ax.set_title(f"({tag}) density $\\rho$ = {d}"); ax.set_xlabel("Number of UAVs")
        ax.set_xticks(range(3, 9)); ax.set_ylim(0, top)
    axes[0].set_ylabel(ylabel)
    hs = [Line2D([], [], color=c, marker=mk, ls=ls, ms=4.5, mec="white", mew=0.6, label=lab)
          for m, lab, c, mk, ls in SERIES if m not in skip]
    fig.legend(handles=hs, loc="upper center", ncol=len(hs), bbox_to_anchor=(0.5, 1.07), handlelength=2.4)
    fig.tight_layout(w_pad=0.8); fig.savefig(os.path.join(OUT, fname)); plt.close(fig)

series_panels("DR", "Sum data rate (Mbps)", "fig_data_rate.png")
series_panels("EE", "Energy efficiency (Gbit/J)", "fig_energy_efficiency.png", skip=("Random Selection",))

# ---- combined data rate + EE (2 x 3 panels) ------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.9), sharex=True)
for row, (metric, ylabel, skip) in enumerate((("DR", "Sum data rate (Mbps)", ()),
                                              ("EE", "Energy efficiency (Gbit/J)", ("Random Selection",)))):
    sel = main[~main.method.isin(skip) & main.method.isin([x[0] for x in SERIES])]
    top = (sel[f"{metric}_mean"] + sel[f"{metric}_ci95"]).max() * 1.06
    for col, d in enumerate(DENS):
        ax = axes[row, col]
        for m, lab, c, mk, ls in SERIES:
            if m in skip: continue
            g = main[(main.density == d) & (main.method == m)].sort_values("N")
            ax.errorbar(g.N, g[f"{metric}_mean"], yerr=g[f"{metric}_ci95"], color=c, marker=mk, ms=4.2,
                        ls=ls, lw=1.8 if m == PROP else 1.3, capsize=2, elinewidth=0.8, mec="white", mew=0.6,
                        zorder=3 if m == PROP else 2)
        ax.set_ylim(0, top); ax.set_xticks(range(3, 9))
        ax.set_title(f"({'abcdef'[3*row+col]}) {'data rate' if row == 0 else 'EE'}, $\\rho$ = {d}")
        if col: ax.tick_params(labelleft=False)
        else: ax.set_ylabel(ylabel)
        if row: ax.set_xlabel("Number of UAVs")
hs = [Line2D([], [], color=c, marker=mk, ls=ls, ms=4.5, mec="white", mew=0.6, label=lab) for m, lab, c, mk, ls in SERIES]
fig.legend(handles=hs, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.035), handlelength=2.4)
fig.tight_layout(w_pad=0.6, h_pad=1.0); fig.savefig(os.path.join(OUT, "fig_rate_ee.png")); plt.close(fig)

# ---- ablation ---------------------------------------------------------------------------
abl = [(BASE, "Base\nSQP [2]", ORANGE), ("SE-aware only", "SE-aware\nonly", GREY1),
       ("SE-aware + alternating", "SE-aware\n+ alt.", AQUA), ("SE-aware + power-aware", "SE-aware\n+ power-aware", GREY3),
       (PROP, "Proposed\n(full)", BLUE)]
g = main[(main.N == 6) & (main.density == 0.8)].set_index("method")
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
for ax, met, lab, fmt in zip(axes, ("DR", "EE"), ("Sum data rate (Mbps)", "Energy efficiency (Gbit/J)"), ("{:.1f}", "{:.3f}")):
    x = np.arange(len(abl))
    vals = [(g.at[m, f"{met}_mean"], g.at[m, f"{met}_ci95"], c) for m, _, c in abl]
    top = max(v + e for v, e, _ in vals)
    for i, (v, e, c) in enumerate(vals):
        ax.bar(i, v, 0.62, color=c, edgecolor="white", linewidth=1, zorder=2)
        ax.errorbar(i, v, yerr=e, color=INK, capsize=2.5, elinewidth=0.8, zorder=3)
        ax.text(i, v + e + 0.025 * top, fmt.format(v), ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l, _ in abl], fontsize=7.5); ax.set_ylabel(lab)
    ax.grid(axis="x", visible=False); ax.margins(y=0.12)
axes[0].set_title("(a) data rate"); axes[1].set_title("(b) energy efficiency")
fig.tight_layout(w_pad=1.5); fig.savefig(os.path.join(OUT, "fig_ablation.png")); plt.close(fig)

# ---- optimality gap ---------------------------------------------------------------------
gdf = pd.read_csv(os.path.join(RES, "optimality_gap.csv")); gm = gdf.groupby("N")
mu, ci = 100 * gm.mean(), 100 * 1.96 * gm.std() / np.sqrt(gm.count())
fig, ax = plt.subplots(figsize=(4.6, 2.8))
for i, (c, lab, col) in enumerate((("base", "Base SQP [2]", ORANGE), ("se_aware", "SE-aware at $P_{t,\\max}$", AQUA),
                                   ("power_aware", "SE-aware + power-aware", BLUE))):
    xs = mu.index + (i - 1) * 0.26
    ax.bar(xs, mu[c], 0.24, color=col, edgecolor="white", linewidth=1, label=lab, zorder=2)
    ax.errorbar(xs, mu[c], yerr=ci[c], fmt="none", color=INK, capsize=2, elinewidth=0.8, zorder=3)
    for xx, v, e in zip(xs, mu[c], ci[c]): ax.text(xx, v + e + 2, f"{v:.0f}", ha="center", fontsize=7.2, color=INK)
ax.axhline(100, color=INK2, lw=0.8, ls=(0, (4, 3)), label="near-optimal benchmark", zorder=1)
ax.set_xticks(mu.index); ax.set_xlabel("Number of UAVs"); ax.set_ylabel("Share of benchmark data rate (%)")
ax.set_ylim(0, 112); ax.set_yticks(range(0, 101, 20)); ax.grid(axis="x", visible=False)
ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.24), fontsize=7.5, handlelength=1.6)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_optimality_gap.png")); plt.close(fig)

# ---- fairness floor trade-off -----------------------------------------------------------
kk = s[(s.N == 6) & (s.density == 0.8) & (s.channel == "LoS (base)") & (s.method == PROP)
       & s.exp.isin(["main", "kappa=0.0", "kappa=0.5", "kappa=1.5", "kappa=2.0"])].copy()
kk["kappa"] = kk.exp.map(lambda e: 1.0 if e == "main" else float(e.split("=")[1])); kk = kk.sort_values("kappa")
b = main[(main.N == 6) & (main.density == 0.8) & (main.method == BASE)].iloc[0]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
axes[0].plot(kk.kappa, kk.DR_mean, color=BLUE, marker="o", ms=4.5, mec="white", label="Proposed")
axes[0].axhline(b.DR_mean, color=ORANGE, ls="--", lw=1.4, label="Base SQP [2]")
axes[0].set_xlabel("Rate-floor factor $\\kappa$"); axes[0].set_ylabel("Sum data rate (Mbps)"); axes[0].set_ylim(0, 72)
axes[0].set_title("(a) data rate"); axes[0].legend(loc="upper right")
axes[1].plot(kk.kappa, kk.MinRate_mean, color=BLUE, marker="o", ms=4.5, mec="white")
axes[1].axhline(b.MinRate_mean, color=ORANGE, ls="--", lw=1.4)
axes[1].set_xlabel("Rate-floor factor $\\kappa$"); axes[1].set_ylabel("Worst-link rate (Mbps)"); axes[1].set_ylim(0, 0.3)
axes[1].set_title("(b) worst-link rate")
for ax in axes: ax.set_xticks([0, 0.5, 1, 1.5, 2])
fig.tight_layout(w_pad=1.5); fig.savefig(os.path.join(OUT, "fig_fairness_floor.png")); plt.close(fig)

# ---- convergence (recomputed; ~30 s) ----------------------------------------------------
sys.path.insert(0, HERE)
import uav_simulation as U   # the simulation module (same folder)
conv = []
for t in range(30):
    rng = np.random.default_rng(777 + t); sc = U.make_scenario(5, 0.8, rng)
    P0 = np.where(sc["active"], U.PMAX, 0.0); R = U.dir_se(P0, sc).sum(1); S = sc["S"]
    fb = U.stage1_constraints(sc, R); rmin = ((S.T @ U.bw_base(sc, R)) * R).min()
    Pa, Xa, _ = U.power_aware(sc, fb, rmin, rng)
    if Pa is None: continue
    B = S.T @ Xa; dem = U.BETA * B[:, None] * U.dir_se(Pa, sc); P = U.min_power(sc, B, dem)
    if P is None: continue
    _, _, h = U.alternating(sc, Xa, P, dem); conv.append(np.array(h) / h[0])
L = max(len(c) for c in conv); C = np.array([np.r_[c, np.full(L - len(c), c[-1])] for c in conv])
fig, ax = plt.subplots(figsize=(4.4, 2.7))
for c in C: ax.plot(range(L), c, color=GREY1, lw=0.7, alpha=0.8, zorder=1)
ax.plot(range(L), C.mean(0), color=AQUA, marker="o", ms=4.5, mec="white", lw=2, zorder=3, label=f"mean of {len(C)} networks")
ax.set_xlabel("Alternating iteration"); ax.set_ylabel("Total power / power after Stage 2")
ax.set_xticks(range(L)); ax.legend(loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_convergence.png")); plt.close(fig)
print("figures written to", OUT, "| mean power reduction:", f"{100*(1-C.mean(0)[-1]):.1f}%")
