# Power-Aware and SE-Aware Bandwidth Allocation with Alternating Power Control for Multi-UAV 6G Networks

Simulation code for the paper *"Power-Aware and Spectral-Efficiency-Aware Graph-Based Bandwidth Allocation with Alternating Power Control for Multi-UAV 6G Networks"* by K. Srinivasa Rao, P. Bharat Kumar, S. Kannappan and G. Jyothi.

The system model follows M. A. Alnakhli, E. M. Mohamed and M. M. Fouda, "Bandwidth allocation and power control optimization for multi-UAVs enabled 6G network," *IEEE Access*, vol. 12, pp. 67405–67415, 2024.

## Method

1. **Stage 1, SE-aware LP:** maximises the data rate Σ_k B_k R_k over the maximal independent sets (MISs) of the link graph. It is an exact linear program.
2. **Stage 1b, power-aware refinement:** a multi-start coordinate search over UAV power levels. Each candidate is scored by the exact LP, with every link kept at or above the base scheme's worst-link rate.
3. **Stage 2, minimum-power control:** finds the lowest powers that meet the QoS targets (LP / M-matrix).
4. **Stage 3, alternating loop:** re-allocates bandwidth at fixed power, then minimises power at fixed bandwidth, and repeats. Total power never increases.

## Files

| File | Content |
|---|---|
| `UAV_power_SE_aware.ipynb` | Google Colab notebook (Runtime → Run all) |
| `uav_simulation.py` | The same code as a Python script |
| `draw_diagrams.py` | Draws the system-model and block diagrams (Figs. 1–2) |
| `make_figures.py` | Redraws every result figure from `results/*.csv` (no re-simulation) |
| `results/` | Raw trials, summary statistics, optimality gap, β sweep (CSV) |
| `figures/` | All figures used in the paper |

## How to run

```
pip install numpy scipy pandas networkx matplotlib
python uav_simulation.py
```

`QUICK = True` runs a 20-trial test in about 10 minutes. `QUICK = False` reproduces the paper (200 trials, about 80 minutes on a 2-core CPU). The seed is fixed (`SEED = 2026`), so the results are reproducible.
