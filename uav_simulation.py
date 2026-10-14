# =============================================================================
# Power-Aware, SE-Aware Bandwidth Allocation with Alternating Power Control
# for Multi-UAV 6G Networks  —  complete simulation (Google Colab ready), v2
#
# Base paper: M. A. Alnakhli, E. M. Mohamed, M. M. Fouda, "Bandwidth Allocation
# and Power Control Optimization for Multi-UAVs Enabled 6G Network",
# IEEE Access, vol. 12, pp. 67405-67415, 2024.
# Code: https://github.com/srinivaskollimarlaece/SE-aware-Bandwidth-Allocation-Alternating-Bandwidth-Power-Optimization-for-Multi-UAV-6G-Networks
#
# HOW TO RUN IN COLAB: Runtime -> Run all.  QUICK = True gives a 2-3 minute test;
# QUICK = False reproduces the paper (about 30-45 minutes).  results.zip is
# written at the end (Files panel on the left -> right-click -> Download).
# =============================================================================
QUICK        = False
TRIALS       = 20 if QUICK else 200     # Monte Carlo trials per configuration
UAV_COUNTS   = [3, 4, 5, 6, 7, 8]
LARGE_COUNTS = [9, 10]                  # scalability study
DENSITIES    = [0.5, 0.8, 1.0]
BETA         = 0.8                      # QoS: each link keeps >= BETA of its stage-1 rate
MAX_ALT_IT   = 20                       # alternating-stage iterations
N_STARTS     = 20                       # random starts of the power-aware search
SEED         = 2026
OUT          = "results"

import os, time, itertools, zipfile, warnings
import numpy as np, pandas as pd, networkx as nx
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import linprog, minimize
from scipy.stats import wilcoxon
warnings.filterwarnings("ignore")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- parameters (base Table 1)
BTOT, PMAX = 10e6, 1.0
PL0, D_V   = 46.4, 50.0
RHO0 = 10 ** ((-147 + 10*np.log10(BTOT) + 10 - 30) / 10)
LEVELS = np.array([1, .5, .25, .1, .05, .02, .01, .003, .001])     # power levels (x Pmax)
CHANNELS = {"LoS (base)":           dict(alpha=2.03, shadow=0.0),
            "LoS + 4 dB shadowing": dict(alpha=2.03, shadow=4.0),
            "Mixed LoS/NLoS":       dict(alpha=2.03, shadow=6.0, nlos_alpha=3.0, p_los=0.7)}

# ---------------------------------------------------------------- scenario (Sec. V of base paper)
def make_scenario(M, density, rng, alpha=2.03, shadow=0.0, nlos_alpha=None, p_los=1.0):
    dH = np.zeros((M, M)); iu = np.triu_indices(M, 1)
    dH[iu] = rng.uniform(1, 100, len(iu[0])); dH = dH + dH.T
    d = np.sqrt(dH**2 + D_V**2); np.fill_diagonal(d, 1.0)
    a = np.full((M, M), alpha)
    if nlos_alpha is not None:
        los = rng.uniform(size=(M, M)) < p_los; los = np.triu(los, 1); los = los | los.T
        a = np.where(los, alpha, nlos_alpha)
    pl = PL0 + 10*a*np.log10(d)
    if shadow > 0:
        sh = rng.normal(0, shadow, (M, M)); sh = np.triu(sh, 1); pl = pl + sh + sh.T
    g = 10 ** (-pl / 10)
    pairs = list(itertools.combinations(range(M), 2))
    n_e = max(1, int(round(density * len(pairs))))
    links = [pairs[i] for i in sorted(rng.choice(len(pairs), n_e, replace=False))]
    G = nx.Graph(); G.add_nodes_from(range(M)); G.add_edges_from(links)
    active = np.array([G.degree(v) > 0 for v in range(M)])
    L = nx.line_graph(G)
    mis = [set(c) for c in nx.find_cliques(nx.complement(L))] if len(links) > 1 else [set(L.nodes)]
    pos = {e: k for k, e in enumerate(links)}
    S = np.zeros((len(mis), len(links)))
    for q, c in enumerate(mis):
        for e in c: S[q, pos[tuple(sorted(e))]] = 1
    T = np.array([[i for i, j in links], [j for i, j in links]])          # tx of dir 0 / dir 1
    return dict(M=M, g=g, links=links, S=S, active=active,
                tx=np.c_[T[0], T[1]], rx=np.c_[T[1], T[0]])

def dir_se(P, sc):
    """log2(1+SINR) for both directions of every link, Eq. (4)-(5); shape (K,2)."""
    g, act, t, r = sc["g"], sc["active"], sc["tx"], sc["rx"]
    Pa = P * act
    tot = Pa @ g                                     # total received power at each UAV
    I = tot[r] - Pa[t] * g[t, r] - Pa[r] * g[r, r]   # exclude own tx and receiver itself
    return np.log2(1 + P[t] * g[t, r] / (np.maximum(I, 0) + RHO0))

def jain(v): return v.sum()**2 / (len(v) * (v**2).sum())

# ---------------------------------------------------------------- stage 1
def fairness_tau(S, gk):
    Q, K = S.shape
    r = linprog(np.r_[np.zeros(Q), -1], A_ub=np.hstack([-S.T, (BTOT*gk)[:, None]]),
                b_ub=np.zeros(K), A_eq=np.r_[np.ones(Q), 0][None], b_eq=[BTOT],
                bounds=[(0, None)]*Q + [(0, 1)], method="highs")
    return min(1.0, r.x[-1]) * (1 - 1e-9)

def lp_bandwidth(S, w, floor_bw, R=None, rmin=0.0):
    """max sum_q X_q w_q  s.t. sum X = Btot, S^T X >= floor_bw, (S^T X)*R >= rmin. None if infeasible."""
    Q, K = S.shape; A = [-S.T]; b = [-floor_bw]
    if rmin > 0: A.append(-(S.T * R[:, None])); b.append(-rmin * np.ones(K))
    r = linprog(-w, A_ub=np.vstack(A), b_ub=np.concatenate(b), A_eq=np.ones((1, Q)),
                b_eq=[BTOT], bounds=[(0, None)]*Q, method="highs")
    return r.x if r.status == 0 else None

def stage1_constraints(sc, R):
    gk = R / R.sum(); tau = fairness_tau(sc["S"], gk); return tau * BTOT * gk

def bw_base(sc, R):      # base paper (11): maximise total bandwidth (SE not in objective)
    return lp_bandwidth(sc["S"], sc["S"].sum(1), stage1_constraints(sc, R))

def bw_se_aware(sc, R):  # proposed (P1): maximise data rate sum_k B_k R_k
    return lp_bandwidth(sc["S"], sc["S"] @ R, stage1_constraints(sc, R))

def bw_equal(sc, R): return np.full(sc["S"].shape[0], BTOT / sc["S"].shape[0])

def bw_sqp_random(sc, R, rng):
    S = sc["S"]; Q = S.shape[0]; fb = stage1_constraints(sc, R) / BTOT
    res = minimize(lambda x: -S.sum(1) @ x, rng.dirichlet(np.ones(Q)), jac=lambda x: -S.sum(1),
                   method="SLSQP", bounds=[(0, 1)]*Q,
                   constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1},
                                {"type": "ineq", "fun": lambda x: S.T @ x - fb}],
                   options=dict(maxiter=300))
    return np.clip(res.x, 0, None) * BTOT

# ---------------------------------------------------------------- stage 1b (proposed): power-aware refinement
def rate_at(sc, P, floor_bw, rmin):
    """(P3) inner LP: best SE-aware allocation for powers P with per-link rate floor rmin."""
    R = dir_se(P, sc).sum(1); X = lp_bandwidth(sc["S"], sc["S"] @ R, floor_bw, R, rmin)
    if X is None: return -1.0, None
    return float(((sc["S"].T @ X) * R).sum()), X

def power_aware(sc, floor_bw, rmin, rng, starts=N_STARTS, sweeps=4):
    """Multi-start coordinate search over discrete power levels; each candidate is
    scored by the exact LP (P3).  The rate never decreases and every link keeps >= rmin."""
    act = np.where(sc["active"])[0]; M = sc["M"]; n_eval = 0
    P0 = np.where(sc["active"], PMAX, 0.0); cands = [P0]
    for _ in range(starts):
        Q = np.zeros(M); Q[act] = PMAX * 10 ** rng.uniform(-3, 0, len(act)); cands.append(Q)
    vals = []
    for Q in cands: vals.append(rate_at(sc, Q, floor_bw, rmin)[0]); n_eval += 1
    best_v, best_P = vals[0], P0
    for o in np.argsort(vals)[::-1][:3]:
        if vals[o] < 0: continue
        P, cur = cands[o].copy(), vals[o]
        for _ in range(sweeps):
            improved = False
            for i in act:
                for l in LEVELS:
                    Q = P.copy(); Q[i] = l * PMAX; v = rate_at(sc, Q, floor_bw, rmin)[0]; n_eval += 1
                    if v > cur * (1 + 1e-6): P, cur, improved = Q, v, True
            if not improved: break
        if cur > best_v: best_v, best_P = cur, P
    if best_v < 0: return None, None, n_eval                 # no feasible point found
    return best_P, rate_at(sc, best_P, floor_bw, rmin)[1], n_eval

def near_optimal(sc, floor_bw, rmin, rng, n=300):
    """Benchmark: best of n random power vectors + Nelder-Mead polish, each scored by (P3)."""
    act = np.where(sc["active"])[0]
    def val(p):
        P = np.zeros(sc["M"]); P[act] = np.clip(p, 1e-4, PMAX); return rate_at(sc, P, floor_bw, rmin)[0]
    cands = [np.full(len(act), PMAX)] + [rng.uniform(0, PMAX, len(act)) for _ in range(n)]
    vs = [val(c) for c in cands]; i = int(np.argmax(vs))
    res = minimize(lambda p: -val(p), cands[i], method="Nelder-Mead", options=dict(maxiter=400))
    return max(vs[i], -res.fun)

# ---------------------------------------------------------------- stage 2 and 3
def min_power(sc, B, dem, Pcap=None):
    """min sum Pt s.t. per-direction rate >= dem, 0<=Pt<=Pmax (LP; = M-matrix solution
    when each UAV has one link)."""
    g, act, M = sc["g"], sc["active"], sc["M"]; rows = []
    for k, (i, j) in enumerate(sc["links"]):
        for c, (t, r) in enumerate(((i, j), (j, i))):
            if dem[k, c] <= 0: continue
            if B[k] <= 1.0: return None
            phi = 2 ** min(dem[k, c] / B[k], 60) - 1
            row = np.zeros(M); m = act.copy(); m[t] = m[r] = False
            row[t] = -g[t, r] / RHO0; row[m] = phi * g[m, r] / RHO0
            rows.append(row / phi)
    if not rows: return np.zeros(M)
    r = linprog(np.ones(M), A_ub=np.array(rows), b_ub=-np.ones(len(rows)),
                bounds=[(0, PMAX) if a else (0, 0) for a in act], method="highs")
    return r.x if r.status == 0 else None

def bw_headroom(sc, P, dem):
    S = sc["S"]; Q, K = S.shape; se = dir_se(P, sc); rows = []
    for k in range(K):
        for c in range(2):
            if dem[k, c] > 0:
                row = np.zeros(Q + 1); row[:Q] = -S[:, k]*se[k, c]/dem[k, c]; row[Q] = 1; rows.append(row)
    r = linprog(np.r_[np.zeros(Q), -1], A_ub=np.array(rows), b_ub=np.zeros(len(rows)),
                A_eq=np.r_[np.ones(Q), 0][None], b_eq=[BTOT], bounds=[(0, None)]*(Q + 1), method="highs")
    return r.x[:Q]

def alternating(sc, X, P, dem):
    hist = [P.sum()]
    for _ in range(MAX_ALT_IT):
        Xn = bw_headroom(sc, P, dem); Pn = min_power(sc, sc["S"].T @ Xn, dem)
        if Pn is None or Pn.sum() >= P.sum() * (1 - 1e-4): break
        X, P = Xn, Pn; hist.append(P.sum())
    return X, P, hist

# ---------------------------------------------------------------- one trial
PROP  = "Proposed (full)"
BASE  = "Base SQP (Alnakhli 2024)"
def run_trial(sc, rng, kappa=1.0):
    out = {}; S = sc["S"]
    P0 = np.where(sc["active"], PMAX, 0.0); se0 = dir_se(P0, sc); R = se0.sum(1)
    fb = stage1_constraints(sc, R)

    def record(name, X, Pstage1, pc=True, alt=False, t_extra=0.0):
        t0 = time.perf_counter(); B = S.T @ X; se1 = dir_se(Pstage1, sc)
        dem = BETA * B[:, None] * se1                      # QoS targets from the stage-1 rates
        P = min_power(sc, B, dem) if pc else Pstage1
        if P is None: P = Pstage1
        it = 0
        if alt and pc: X, P, h = alternating(sc, X, P, dem); B = S.T @ X; it = len(h) - 1
        el = (time.perf_counter() - t0) * 1e3 + t_extra
        rl = (B[:, None] * dir_se(P, sc)).sum(1); dr = rl.sum(); pw = P.sum()
        out[name] = dict(DR=dr/1e6, EE=dr/pw/1e9 if pw > 0 else np.nan, Ptot=pw, Jain=jain(rl),
                         MinRate=rl.min()/1e6, alt_iters=it, t_ms=el)

    t0 = time.perf_counter(); Xb = bw_base(sc, R); tb = (time.perf_counter()-t0)*1e3
    t0 = time.perf_counter(); Xs = bw_se_aware(sc, R); ts = (time.perf_counter()-t0)*1e3
    rmin = kappa * ((S.T @ Xb) * R).min()                # QoS floor: worst link of the base scheme
    t0 = time.perf_counter(); Pa, Xa, ne = power_aware(sc, fb, rmin, rng); tp = (time.perf_counter()-t0)*1e3
    feasible = Pa is not None
    if not feasible: Pa, Xa = P0, Xs                      # fall back to SE-aware at Pmax

    record(BASE, Xb, P0, t_extra=tb)
    record("Base SQP-WoPC", Xb, P0, pc=False)
    record("Base SLSQP random start", bw_sqp_random(sc, R, rng), P0)
    record("Equal MIS bandwidth", bw_equal(sc, R), P0)
    record("SE-aware only", Xs, P0, t_extra=ts)
    record("SE-aware + alternating", Xs, P0, alt=True, t_extra=ts)
    record("SE-aware + power-aware", Xa, Pa, t_extra=ts + tp)
    record(PROP, Xa, Pa, alt=True, t_extra=ts + tp)
    record("Proposed-WoPC", Xa, Pa, pc=False)
    Br = rng.uniform(0, 1, len(R)); Br *= BTOT / Br.sum(); Pr = rng.uniform(0, PMAX, sc["M"]) * sc["active"]
    rl = (Br[:, None] * dir_se(Pr, sc)).sum(1); dr = rl.sum()
    out["Random Selection"] = dict(DR=dr/1e6, EE=dr/Pr.sum()/1e9, Ptot=Pr.sum(), Jain=jain(rl),
                                   MinRate=rl.min()/1e6, alt_iters=0, t_ms=0)
    out[PROP]["feasible"] = float(feasible); out[PROP]["n_eval"] = ne
    out[PROP]["t_power_aware_ms"] = tp
    return out

def run_config(M, dens, trials, chan="LoS (base)", tag="main", kappa=1.0):
    rows = []
    for t in range(trials):
        rng = np.random.default_rng(SEED + 100000*M + int(1000*dens) + t)
        sc = make_scenario(M, dens, rng, **CHANNELS[chan])
        for meth, v in run_trial(sc, rng, kappa).items():
            rows.append(dict(exp=tag, channel=chan, N=M, density=dens, trial=t, method=meth, **v))
    return rows

# ---------------------------------------------------------------- statistics
def summarise(df):
    res = []
    for key, g in df.groupby(["exp", "channel", "N", "density"]):
        ref = g[g.method == PROP].sort_values("trial")
        for m, gm in g.groupby("method"):
            gm = gm.sort_values("trial"); row = dict(zip(["exp", "channel", "N", "density"], key), method=m)
            for met in ("DR", "EE", "Jain", "MinRate", "Ptot", "t_ms", "alt_iters"):
                x = gm[met].dropna().values
                row[f"{met}_mean"] = x.mean(); row[f"{met}_std"] = x.std(ddof=1)
                row[f"{met}_ci95"] = 1.96*x.std(ddof=1)/np.sqrt(len(x)); row[f"{met}_median"] = np.median(x)
            if m == PROP and "feasible" in gm: row["feasible_%"] = 100 * gm["feasible"].mean()
            if m != PROP:
                for met in ("DR", "EE"):
                    a, b = ref[met].values, gm[met].values
                    row[f"gain_{met}_%"] = 100*(a.mean()/b.mean() - 1)
                    if np.allclose(a, b): row[f"p_{met}"] = np.nan; continue
                    try: p = wilcoxon(a, b, alternative="greater").pvalue
                    except ValueError: p = 1.0
                    row[f"p_{met}"] = 1.0 if np.isnan(p) else p
            res.append(row)
    s = pd.DataFrame(res)
    for met in ("DR", "EE"):
        s[f"p_{met}_holm"] = np.nan
        for _, g in s.groupby(["exp", "channel", "N", "density"]):
            idx = g[f"p_{met}"].dropna().sort_values().index; m = len(idx); run = 0
            for r_, i in enumerate(idx):
                run = max(run, min(1, (m - r_) * s.at[i, f"p_{met}"])); s.at[i, f"p_{met}_holm"] = run
    return s

if __name__ == "__main__":
    T0 = time.time(); all_rows = []
    for d in DENSITIES:
        for M in UAV_COUNTS:
            all_rows += run_config(M, d, TRIALS); print(f"main  density={d}  N={M}  ({time.time()-T0:.0f}s)")
    for M in LARGE_COUNTS:
        all_rows += run_config(M, 0.5, max(10, TRIALS//4), tag="scale"); print(f"scale N={M}  ({time.time()-T0:.0f}s)")
    for ch in list(CHANNELS)[1:]:
        for M in (4, 6, 8):
            all_rows += run_config(M, 0.8, TRIALS, chan=ch, tag="channel"); print(f"channel {ch} N={M}  ({time.time()-T0:.0f}s)")
    for kap in (0.0, 0.5, 1.5, 2.0):                      # fairness floor sweep (kappa = 1 is the main run)
        all_rows += run_config(6, 0.8, max(10, TRIALS//4), tag=f"kappa={kap}", kappa=kap)
    print(f"fairness sweep done ({time.time()-T0:.0f}s)")
    raw = pd.DataFrame(all_rows); raw.to_csv(f"{OUT}/raw_all_trials.csv", index=False)
    summ = summarise(raw); summ.to_csv(f"{OUT}/summary_stats.csv", index=False)

    # optimality gap vs near-optimal benchmark (N = 3, 4, 5; density 0.8)
    gap = []
    for M in (3, 4, 5):
        for t in range(10 if QUICK else 30):
            rng = np.random.default_rng(SEED + 7000*M + t); sc = make_scenario(M, 0.8, rng)
            P0 = np.where(sc["active"], PMAX, 0.0); R = dir_se(P0, sc).sum(1); S = sc["S"]
            fb = stage1_constraints(sc, R); Xb = bw_base(sc, R); Xs = bw_se_aware(sc, R)
            rmin = ((S.T @ Xb) * R).min()
            Pa, Xa, _ = power_aware(sc, fb, rmin, rng)
            ra = rate_at(sc, Pa, fb, rmin)[0] if Pa is not None else np.nan
            opt = max(near_optimal(sc, fb, rmin, rng), ra if ra == ra else -1)
            gap.append(dict(N=M, trial=t, base=((S.T @ Xb) * R).sum()/opt, se_aware=((S.T @ Xs) * R).sum()/opt,
                            power_aware=ra/opt))
        print(f"gap N={M} ({time.time()-T0:.0f}s)")
    gdf = pd.DataFrame(gap); gdf.to_csv(f"{OUT}/optimality_gap.csv", index=False)

    # beta sweep
    beta_rows = []
    for b in (0.5, 0.6, 0.7, 0.8, 0.9):
        BETA = b
        for r in run_config(6, 0.8, max(10, TRIALS//4), tag=f"beta={b}"):
            if r["method"] in (PROP, BASE): r["beta"] = b; beta_rows.append(r)
    BETA = 0.8
    bdf = pd.DataFrame(beta_rows); bdf.to_csv(f"{OUT}/beta_sweep_raw.csv", index=False)

    # convergence of the alternating stage (N = 5, density 0.8)
    conv = []
    for t in range(30):
        rng = np.random.default_rng(777 + t); sc = make_scenario(5, 0.8, rng)
        P0 = np.where(sc["active"], PMAX, 0.0); se0 = dir_se(P0, sc); R = se0.sum(1); S = sc["S"]
        fb = stage1_constraints(sc, R); rmin = ((S.T @ bw_base(sc, R)) * R).min()
        Pa, Xa, _ = power_aware(sc, fb, rmin, rng)
        if Pa is None: continue
        B = S.T @ Xa; dem = BETA * B[:, None] * dir_se(Pa, sc); P = min_power(sc, B, dem)
        if P is None: continue
        _, _, h = alternating(sc, Xa, P, dem); conv.append(np.array(h) / h[0])

    # ------------------------------------------------------------ figures
    main = summ[summ.exp == "main"]
    show = [PROP, "SE-aware + alternating", BASE, "Equal MIS bandwidth", "Random Selection"]
    style = {show[0]: ("o", "-"), show[1]: ("D", "-"), show[2]: ("s", "--"), show[3]: ("^", ":"), show[4]: ("x", "-.")}
    for met, lab, log in (("DR", "Sum data rate (Mbps)", False), ("EE", "Energy efficiency (Gbit/J)", True)):
        for d in DENSITIES:
            fig, ax = plt.subplots(figsize=(5.2, 3.8))
            for m in show:
                g = main[(main.density == d) & (main.method == m)].sort_values("N"); mk, ls = style[m]
                ax.errorbar(g.N, g[f"{met}_mean"], yerr=g[f"{met}_ci95"], marker=mk, ls=ls, capsize=3, ms=5, label=m)
            ax.set_xlabel("Number of UAVs"); ax.set_ylabel(lab); ax.grid(alpha=.3)
            if log: ax.set_yscale("log")
            ax.set_title(f"Network density = {d}  (mean ± 95% CI, {TRIALS} trials)", fontsize=9)
            ax.legend(fontsize=6.5); fig.tight_layout(); fig.savefig(f"{OUT}/fig_{met}_density_{d}.png", dpi=300)
            plt.close(fig)
    abl = [BASE, "SE-aware only", "SE-aware + alternating", "SE-aware + power-aware", PROP]
    g = main[(main.N == 6) & (main.density == 0.8)].set_index("method").loc[abl]
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.4))
    for a, met, lab in zip(ax, ("DR", "EE"), ("Sum data rate (Mbps)", "EE (Gbit/J)")):
        a.bar(range(5), g[f"{met}_mean"], yerr=g[f"{met}_ci95"], capsize=3,
              color=["#888", "#9ab", "#69c", "#3a7bbf", "#1f4e8c"])
        a.set_xticks(range(5)); a.set_xticklabels(["Base", "SE-aware", "+Alt.", "+Power-aware", "Full"], fontsize=7.5)
        a.set_ylabel(lab); a.grid(axis="y", alpha=.3)
    fig.suptitle("Ablation (N = 6, density = 0.8)", fontsize=9); fig.tight_layout()
    fig.savefig(f"{OUT}/fig_ablation.png", dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    L = max(len(c) for c in conv); C = np.array([np.r_[c, np.full(L - len(c), c[-1])] for c in conv])
    for c in C: ax.plot(c, color="#ccc", lw=.8)
    ax.plot(C.mean(0), "o-", color="#247", label="mean")
    ax.set_xlabel("Alternating iteration"); ax.set_ylabel("Total power / initial")
    ax.set_title("Monotone convergence of the alternating stage", fontsize=9)
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/fig_convergence.png", dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    gm = gdf.groupby("N")[["base", "se_aware", "power_aware"]]
    mu, ci = gm.mean(), 1.96 * gm.std() / np.sqrt(gm.count())
    for i, (c, lab, col) in enumerate((("base", "Base SQP [2]", "#999"), ("se_aware", "SE-aware (Pmax)", "#69c"),
                                       ("power_aware", "SE-aware + power-aware", "#1f4e8c"))):
        ax.bar(mu.index + (i - 1) * 0.27, 100 * mu[c], 0.27, yerr=100 * ci[c], capsize=3, color=col, label=lab)
    ax.set_xticks(mu.index); ax.set_xlabel("Number of UAVs"); ax.set_ylabel("% of near-optimal data rate")
    ax.set_ylim(0, 128); ax.set_yticks(range(0, 101, 20)); ax.grid(axis="y", alpha=.3)
    ax.legend(fontsize=6.8, loc="upper center", ncol=3, frameon=False)
    ax.set_title("Optimality gap (density 0.8)", fontsize=9); fig.tight_layout()
    fig.savefig(f"{OUT}/fig_optimality_gap.png", dpi=300); plt.close(fig)

    # ------------------------------------------------------------ printed tables
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    cols = ["N", "density", "method", "DR_mean", "DR_ci95", "EE_mean", "EE_ci95", "Jain_mean", "MinRate_mean",
            "gain_DR_%", "gain_EE_%", "p_DR_holm", "p_EE_holm"]
    print("\n=== A: main ==="); print(main[main.method.isin([PROP, BASE, "SE-aware + alternating"])][cols].round(4).to_string(index=False))
    print("\n=== B: ablation ==="); print(main[main.N.isin([4, 6, 8]) & (main.density == 0.8)][cols].round(4).to_string(index=False))
    print("\n=== C: scale/channel ==="); print(summ[summ.exp.isin(["scale", "channel"]) & summ.method.isin([PROP, BASE])][["exp", "channel"] + cols].round(4).to_string(index=False))
    print("\n=== D: runtime (ms) ==="); print(raw[raw.exp.isin(["main", "scale"])].groupby(["N", "method"])["t_ms"].mean().unstack()[[BASE, "SE-aware + alternating", PROP]].round(1).to_string())
    print("\n=== E: feasibility % ==="); print(summ[summ.method == PROP][["exp", "channel", "N", "density", "feasible_%"]].to_string(index=False))
    print("\n=== F: fairness floor kappa (N=6, 0.8) ===")
    kk = summ[(summ.N == 6) & (summ.density == 0.8) & summ.exp.isin(["main", "kappa=0.0", "kappa=0.5", "kappa=1.5", "kappa=2.0"])
              & (summ.channel == "LoS (base)") & (summ.method == PROP)]
    print(kk[["exp", "DR_mean", "EE_mean", "Jain_mean", "MinRate_mean", "feasible_%"]].round(4).to_string(index=False))
    print("\n=== G: optimality gap ==="); print(gdf.groupby("N")[["base", "se_aware", "power_aware"]].agg(["mean", "min"]).round(3).to_string())
    print("\n=== H: beta ==="); print(bdf.groupby(["beta", "method"])[["DR", "EE", "Jain"]].mean().round(4).to_string())
    print(f"\nTotal time {time.time()-T0:.0f} s")
    with zipfile.ZipFile("results.zip", "w") as z:
        for f in os.listdir(OUT): z.write(os.path.join(OUT, f))
    try:
        from google.colab import files; files.download("results.zip")
    except Exception:
        print("results.zip written -> Files panel (left) -> right-click -> Download")
