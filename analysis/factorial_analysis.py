import numpy as np
from scipy import stats

# ============================================================
# PART 1: Reconstructed two-way ANOVA for continuous metrics
# (steps, path length, energy), from verified cell means/SDs/n.
# This is a standard, legitimate technique for a balanced design
# (equal n per cell) when only summary statistics are available,
# not raw per-episode values. All numbers below are the verified,
# single-matched-batch (commit bd73c88) figures.
# ============================================================

n = 30  # per cell, confirmed equal across all four conditions

metrics = {
    "steps":  {"full": (67.03, 13.61), "no_ga": (62.23, 16.67), "no_repulsion": (61.83, 12.15), "floor": (67.67, 22.41)},
    "path":   {"full": (437.51, 74.90), "no_ga": (423.01, 68.73), "no_repulsion": (416.86, 64.43), "floor": (438.70, 86.19)},
    "energy": {"full": (55.68, 7.57),  "no_ga": (53.58, 6.77),  "no_repulsion": (52.70, 6.83),  "floor": (55.34, 8.65)},
}

def two_way_anova_from_summary(cell_stats, n):
    """
    cell_stats: dict with keys full, no_ga, no_repulsion, floor -> (mean, sd)
    Factor A = allocator (GA vs Greedy); Factor B = repulsion (Yes vs No)
    GA     = {full, no_repulsion}
    Greedy = {no_ga, floor}
    Rep=Y  = {full, no_ga}
    Rep=N  = {no_repulsion, floor}
    """
    m_full, s_full = cell_stats["full"]
    m_noga, s_noga = cell_stats["no_ga"]
    m_norep, s_norep = cell_stats["no_repulsion"]
    m_floor, s_floor = cell_stats["floor"]

    grand_mean = (m_full + m_noga + m_norep + m_floor) / 4.0

    A_ga = (m_full + m_norep) / 2.0
    A_greedy = (m_noga + m_floor) / 2.0
    B_rep = (m_full + m_noga) / 2.0
    B_norep = (m_norep + m_floor) / 2.0

    SS_A = (2 * n) * ((A_ga - grand_mean) ** 2 + (A_greedy - grand_mean) ** 2)
    SS_B = (2 * n) * ((B_rep - grand_mean) ** 2 + (B_norep - grand_mean) ** 2)

    SS_cells = n * sum((m - grand_mean) ** 2 for m in [m_full, m_noga, m_norep, m_floor])
    SS_AB = SS_cells - SS_A - SS_B

    SS_error = (n - 1) * (s_full**2 + s_noga**2 + s_norep**2 + s_floor**2)

    df_A, df_B, df_AB = 1, 1, 1
    df_error = 4 * (n - 1)

    MS_A, MS_B, MS_AB = SS_A / df_A, SS_B / df_B, SS_AB / df_AB
    MS_error = SS_error / df_error

    F_A = MS_A / MS_error
    F_B = MS_B / MS_error
    F_AB = MS_AB / MS_error

    p_A = 1 - stats.f.cdf(F_A, df_A, df_error)
    p_B = 1 - stats.f.cdf(F_B, df_B, df_error)
    p_AB = 1 - stats.f.cdf(F_AB, df_AB, df_error)

    SS_total = SS_A + SS_B + SS_AB + SS_error
    eta2_A = SS_A / SS_total
    eta2_B = SS_B / SS_total
    eta2_AB = SS_AB / SS_total

    return {
        "grand_mean": grand_mean,
        "A_ga": A_ga, "A_greedy": A_greedy, "B_rep": B_rep, "B_norep": B_norep,
        "F_A": F_A, "p_A": p_A, "eta2_A": eta2_A,
        "F_B": F_B, "p_B": p_B, "eta2_B": eta2_B,
        "F_AB": F_AB, "p_AB": p_AB, "eta2_AB": eta2_AB,
        "df_error": df_error,
    }

print("="*70)
print("RECONSTRUCTED TWO-WAY ANOVA (balanced design, n=30 per cell)")
print("="*70)
for metric_name, cell_stats in metrics.items():
    r = two_way_anova_from_summary(cell_stats, n)
    print(f"\n--- {metric_name.upper()} ---")
    print(f"Allocator main effect (GA mean={r['A_ga']:.2f} vs Greedy mean={r['A_greedy']:.2f}):")
    print(f"  F(1,{r['df_error']}) = {r['F_A']:.3f}, p = {r['p_A']:.4f}, eta^2 = {r['eta2_A']:.4f}")
    print(f"Repulsion main effect (Rep mean={r['B_rep']:.2f} vs NoRep mean={r['B_norep']:.2f}):")
    print(f"  F(1,{r['df_error']}) = {r['F_B']:.3f}, p = {r['p_B']:.4f}, eta^2 = {r['eta2_B']:.4f}")
    print(f"Allocator x Repulsion interaction:")
    print(f"  F(1,{r['df_error']}) = {r['F_AB']:.3f}, p = {r['p_AB']:.4f}, eta^2 = {r['eta2_AB']:.4f}")

# ============================================================
# PART 2: Collision outcome analysis, using exact per-episode
# raw counts (Full and No-Repulsion confirmed from raw per-episode
# breakdown; No-GA / Floor are deterministic all-zero, confirmed
# by their reported 0.00 +/- 0.00).
# ============================================================

print("\n" + "="*70)
print("COLLISION OUTCOME ANALYSIS (exact per-episode data)")
print("="*70)

collision_episodes = {
    "full": 2,           # confirmed: run_14, run_16
    "no_ga": 0,           # confirmed deterministic zero
    "no_repulsion": 16,   # confirmed from raw breakdown
    "floor": 0,           # confirmed deterministic zero
}

def wilson_ci(successes, n, z=1.96):
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2)))) / denom
    return max(0, centre - half), min(1, centre + half)

for variant, k in collision_episodes.items():
    p = k / n
    lo, hi = wilson_ci(k, n)
    print(f"{variant:15s}: {k}/{n} episodes with >=1 collision "
          f"({p*100:.1f}%, 95% CI [{lo*100:.1f}%, {hi*100:.1f}%])")

table_full_vs_norep = [[collision_episodes["full"], n - collision_episodes["full"]],
                        [collision_episodes["no_repulsion"], n - collision_episodes["no_repulsion"]]]
odds_ratio, p_fisher = stats.fisher_exact(table_full_vs_norep)
print(f"\nFisher's exact test, Full vs No-Repulsion (repulsion effect under GA allocation):")
print(f"  Odds ratio = {odds_ratio:.3f}, p = {p_fisher:.6f}")

rr = (collision_episodes["no_repulsion"] / n) / (collision_episodes["full"] / n)
print(f"  Relative risk (No-Repulsion vs Full) = {rr:.2f}x")

print("\nNote: a repulsion effect under GREEDY allocation cannot be tested,")
print("since both No-GA and Floor recorded zero collisions in all 30 episodes")
print("(no variance to test). Repulsion's protective effect is specific to the")
print("GA-allocation condition in this scenario.")
