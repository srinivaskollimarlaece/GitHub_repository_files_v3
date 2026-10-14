import itertools, numpy as np, networkx as nx
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, MUTED, ACC, ACC2, WARN = "#1f2328", "#8f8e88", "#2a78d6", "#14805a", "#eb6834"
NEUTRAL_FC, ACC_FC, ACC2_FC = "#f6f6f4", "#eaf2fc", "#e6f5ee"
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix"})

# ---------------------------------------------------------------- Fig. 1: system model
rng = np.random.default_rng(7)
M = 6
xy = np.array([[10, 70], [40, 92], [78, 80], [92, 40], [55, 18], [18, 30]], float)
z = np.full(M, 1.0)
pairs = list(itertools.combinations(range(M), 2))
links = [(0, 1), (0, 5), (1, 2), (2, 3), (3, 4), (4, 5), (1, 4), (2, 5), (0, 3)]
G = nx.Graph(); G.add_edges_from(links)
L = nx.line_graph(G)
mis = [sorted(tuple(sorted(e)) for e in c) for c in nx.find_cliques(nx.complement(L))]
mis.sort(key=lambda c: (-len(c), c))
colors = ["#2a78d6"] * 8

fig = plt.figure(figsize=(11, 4.4))
# (a) network
ax = fig.add_axes([0.02, 0.06, 0.40, 0.84]); ax.set_xlim(-5, 108); ax.set_ylim(-8, 110); ax.axis("off")
ax.set_title("(a) Multi-UAV network G(V, E)", fontsize=11, color=INK, loc="left")
ax.add_patch(FancyBboxPatch((-2, -6), 108, 112, boxstyle="round,pad=0,rounding_size=4",
                            fc=NEUTRAL_FC, ec="#d8d7d2", lw=1))
show = mis[:1]
lk_col = {}
for qi, q in enumerate(show):
    for e in q: lk_col.setdefault(e, colors[qi])
for (i, j) in links:
    c = lk_col.get(tuple(sorted((i, j))), MUTED)
    ax.plot(*zip(xy[i], xy[j]), color=c, lw=2.4, zorder=1, solid_capstyle="round")
for v in range(M):
    x, y = xy[v]
    ax.add_patch(plt.Circle((x, y), 5.2, fc="white", ec=INK, lw=1.6, zorder=3))
    for dx, dy in ((-4.2, 4.2), (4.2, 4.2), (-4.2, -4.2), (4.2, -4.2)):
        ax.add_patch(plt.Circle((x + dx, y + dy), 1.7, fc="none", ec=INK, lw=1, zorder=3))
    ax.text(x, y, f"v{v+1}", ha="center", va="center", fontsize=9, color=INK, zorder=4, weight="bold")
ax.text(2, -3, "UAVs at altitude h; horizontal spacing U[1,100] m; vertical spacing 50 m",
        fontsize=7.5, color=MUTED)
# (b) MIS / bandwidth
ax2 = fig.add_axes([0.46, 0.06, 0.52, 0.84]); ax2.axis("off"); ax2.set_xlim(0, 100); ax2.set_ylim(0, 100)
ax2.set_title("(b) Link graph Ĝ → maximal independent sets → shared bands", fontsize=11, color=INK, loc="left")
y = 88
for qi, q in enumerate(mis[:5]):
    c = colors[0] if qi == 0 else MUTED
    ax2.add_patch(FancyBboxPatch((1, y - 6), 50, 11, boxstyle="round,pad=0,rounding_size=1.8",
                                 fc="white", ec=c, lw=1.6))
    ax2.text(4, y - 0.5, f"MIS q{qi+1}:  " + ",  ".join(f"(v{a+1},v{b+1})" for a, b in q),
             fontsize=8.6, va="center", color=INK)
    ax2.add_patch(FancyArrowPatch((52, y - 0.5), (60, y - 0.5), arrowstyle="-|>", mutation_scale=10, color=MUTED))
    y -= 15
ax2.text(4, y + 4, f"... {len(mis)} MISs in total. Links in one MIS share no UAV,\nso they can reuse the same band (q1 highlighted in (a)).",
         fontsize=7.8, color=MUTED, va="top")
# band bar
widths = np.array([0.30, 0.24, 0.18, 0.16, 0.12])
x0 = 62; total_h = 66; yb = 94
ax2.text(x0, 97, "B_tot = 10 MHz split into X_q", fontsize=8.6, color=INK)
for qi, w in enumerate(widths):
    h = w * total_h
    c = colors[0] if qi == 0 else MUTED
    ax2.add_patch(plt.Rectangle((x0, yb - h - 1), 14, h, fc=c, alpha=0.85 if qi == 0 else 0.3+0.1*(qi%2), ec="white", lw=1.5))
    ax2.text(x0 + 16, yb - h / 2 - 1, f"X_q{qi+1}", fontsize=8.4, va="center", color=INK)
    yb -= h
ax2.text(x0, yb - 3, "Base [2]:  max Σ_q |q|·X_q\nProposed:  max Σ_q (Σ_k s_qk R_k)·X_q",
         fontsize=8.2, color=INK, va="top", linespacing=1.6)
fig.savefig("fig_system_model.png", dpi=300); plt.close(fig)

# ---------------------------------------------------------------- Fig. 2: block diagram
fig, ax = plt.subplots(figsize=(11, 6)); ax.set_xlim(0, 110); ax.set_ylim(0, 62); ax.axis("off")

def box(x, y, w, h, title, body, ec="#52514e", fc=NEUTRAL_FC, tc=INK, fs=8.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6", fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h - 2.4, title, ha="center", va="top", fontsize=9.6, weight="bold", color=tc)
    ax.text(x + w / 2, y + h - 6.4, body, ha="center", va="top", fontsize=fs, color=INK, linespacing=1.35)

def arrow(p, q, c=INK, txt=None, off=(0, 1.3), cs="arc3"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13, color=c, lw=1.5, connectionstyle=cs))
    if txt: ax.text((p[0]+q[0])/2+off[0], (p[1]+q[1])/2+off[1], txt, ha="center", fontsize=7.8, color=c)

# top row
box(1, 40, 24, 19, "Inputs", "UAV positions, graph G\nB_tot, Pt_max, β\nA2A path-loss model")
box(28, 40, 24, 19, "Graph model", "link graph Ĝ, MISs\nMIS matrix [s_qk]\nR_k, g_k at Pt_max")
box(55, 40, 25, 19, "Stage 1 (proposed)", "SE-aware LP (P1)\nmax Σ_q X_q Σ_k s_qk R_k\nB_k ≥ τ B_tot g_k", ec=ACC, fc=ACC_FC, tc=ACC)
box(83, 40, 26, 19, "Stage 1b (proposed)", "power-aware search (P3)\nmulti-start coordinate\nsearch over Pt levels;\nexact LP per candidate,\nevery link ≥ r_min", ec=ACC, fc=ACC_FC, tc=ACC, fs=7.6)
arrow((25, 49.5), (28, 49.5)); arrow((52, 49.5), (55, 49.5)); arrow((80, 49.5), (83, 49.5))
# bottom row
box(78, 6, 31, 22, "Stage 2", "QoS targets c_kd = β B_k SE_kd(Pt)\nminimum-power LP\n(M-matrix solution for one link per UAV)")
arrow((96, 40), (96, 28), txt="B_k, Pt", off=(4, 0))
ax.add_patch(FancyBboxPatch((30, 3), 40, 26, boxstyle="round,pad=0,rounding_size=1.6", fc=ACC2_FC, ec=ACC2, lw=1.6))
ax.text(50, 27.6, "Stage 3 (proposed): alternating loop", ha="center", va="top", fontsize=9.6, weight="bold", color=ACC2)
box(32, 6, 17, 17, "(a) bandwidth", "fix Pt;\nLP (P2): re-allocate\nX_q to maximise\nheadroom t", ec=ACC2, fc="white", fs=7.8)
box(51, 6, 17, 17, "(b) power", "fix B;\nminimum-power\nLP", ec=ACC2, fc="white", fs=7.8)
arrow((49, 17), (51, 17), c=ACC2); arrow((51, 10), (49, 10), c=ACC2)
arrow((78, 17), (70, 17), txt="Pt", c=INK)
box(2, 6, 20, 22, "Output", "bandwidths B*\npowers Pt*\ndata rate, EE")
arrow((30, 17), (22, 17), txt="Σ Pt drop\n< 0.01 %", off=(0, 1.6), c=ACC2)
ax.text(2, 38.2, "Prop. 1: (P1) is an LP over the base feasible set, so its rate at Pt_max is never lower.", fontsize=7.8, color=ACC, va="top")
ax.text(2, 35.6, "Prop. 2: Stage 1b never lowers the data rate and keeps every link ≥ r_min.", fontsize=7.8, color=ACC, va="top")
ax.text(2, 33.0, "Prop. 3: Σ Pt never increases in Stage 3 and every rate target stays met, so the loop converges.", fontsize=7.8, color=ACC2, va="top")
fig.savefig("fig_block_diagram.png", dpi=300, bbox_inches="tight"); plt.close(fig)
