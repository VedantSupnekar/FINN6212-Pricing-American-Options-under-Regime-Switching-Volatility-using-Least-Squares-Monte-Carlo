"""
Early Exercise Boundary Analysis by Regime.

Extracts the (time_step, stock_price, regime) triples where the LSMC
algorithm decides to exercise, then visualises the exercise boundary
separated by the prevailing volatility regime.

Expected result: the critical stock price (exercise boundary) is higher
in the high-vol regime — the option carries more time value, so the
holder demands a deeper ITM price before giving it up.

Usage
-----
    python -m analysis.early_exercise      # from project root
    python analysis/early_exercise.py      # also works
"""

import sys
import os

# Allow running from the analysis/ subdirectory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from simulate_gbm import simulate_regime_switching_gbm
from lsmc import lsmc_american_put


# ════════════════════════════════════════════════════════════════════════
#  Extract exercise events from LSMC output
# ════════════════════════════════════════════════════════════════════════

def extract_exercise_events(S, regime, exercise_matrix, params):
    """
    Extract the first exercise event for each path.

    Parameters
    ----------
    S               : ndarray (N+1, n_paths) – stock prices
    regime          : ndarray (N+1, n_paths) – regime state (0 or 1)
    exercise_matrix : ndarray (N+1, n_paths) – 1 at exercise, 0 otherwise
    params          : dict – simulation params (needs 'N', 'dt')

    Returns
    -------
    events : dict with keys
        time_step  : ndarray – time step index of exercise
        time_years : ndarray – exercise time in years
        stock_price: ndarray – stock price at exercise
        regime     : ndarray – regime at exercise (0=Low, 1=High)
        is_early   : ndarray – True if exercised before maturity
    """
    N = params["N"]
    dt = params["dt"]
    n_paths = S.shape[1]

    steps, prices, regimes = [], [], []

    for p in range(n_paths):
        ex_times = np.where(exercise_matrix[:, p] == 1)[0]
        if len(ex_times) > 0:
            t_ex = ex_times[0]
            steps.append(t_ex)
            prices.append(S[t_ex, p])
            regimes.append(regime[t_ex, p])

    steps = np.array(steps)
    prices = np.array(prices)
    regimes = np.array(regimes)

    return {
        "time_step": steps,
        "time_years": steps * dt,
        "stock_price": prices,
        "regime": regimes,
        "is_early": steps < N,
    }


# ════════════════════════════════════════════════════════════════════════
#  Compute exercise boundary (lower envelope by time step)
# ════════════════════════════════════════════════════════════════════════

def compute_exercise_boundary(events, N, by_regime=True):
    """
    Estimate the exercise boundary as the maximum stock price at which
    exercise occurs at each time step (i.e. the critical S* boundary).

    For a put option the holder exercises when S ≤ S*(t), so S* is the
    upper envelope of exercise prices at each time step.

    Parameters
    ----------
    events    : dict – output of extract_exercise_events
    N         : int  – number of time steps
    by_regime : bool – if True, compute separate boundaries per regime

    Returns
    -------
    boundary : dict
        If by_regime=True:
            {0: (steps_0, S_star_0), 1: (steps_1, S_star_1)}
        If by_regime=False:
            {"all": (steps, S_star)}
    """
    result = {}
    regime_keys = [0, 1] if by_regime else ["all"]

    for rk in regime_keys:
        if rk == "all":
            mask = np.ones(len(events["time_step"]), dtype=bool)
        else:
            mask = events["regime"] == rk

        if mask.sum() == 0:
            result[rk] = (np.array([]), np.array([]))
            continue

        ts = events["time_step"][mask]
        sp = events["stock_price"][mask]

        # For each time step, find the max stock price at which exercise occurs
        unique_steps = np.unique(ts)
        boundary_steps = []
        boundary_prices = []

        for step in unique_steps:
            step_mask = ts == step
            if step_mask.sum() >= 5:  # require enough observations
                boundary_steps.append(step)
                boundary_prices.append(np.max(sp[step_mask]))

        result[rk] = (np.array(boundary_steps), np.array(boundary_prices))

    return result


# ════════════════════════════════════════════════════════════════════════
#  Plotting
# ════════════════════════════════════════════════════════════════════════

def plot_exercise_boundary(events, boundary, params, K, save_path=None):
    """
    Create a publication-quality exercise boundary plot.

    Panel 1: Scatter of exercise events coloured by regime.
    Panel 2: Exercise boundary curves (S* vs time) by regime.
    Panel 3: Histogram of exercise times by regime.
    """
    dt = params["dt"]
    N = params["N"]
    T = params["T"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 14), height_ratios=[3, 2, 1.5])

    regime_colors = {0: "#2196F3", 1: "#F44336"}  # blue = calm, red = turbulent
    regime_labels = {0: "Low-Vol Regime (σ₁)", 1: "High-Vol Regime (σ₂)"}

    # ── Panel 1: Scatter plot of exercise events ─────────────────────
    ax1 = axes[0]
    for r_val in [0, 1]:
        mask = (events["regime"] == r_val) & events["is_early"]
        if mask.sum() > 0:
            ax1.scatter(
                events["time_years"][mask],
                events["stock_price"][mask],
                c=regime_colors[r_val],
                alpha=0.08,
                s=6,
                label=regime_labels[r_val],
                rasterized=True,
            )

    ax1.axhline(K, color="grey", linestyle="--", linewidth=1, alpha=0.7, label=f"Strike K = {K:.0f}")
    ax1.set_xlabel("Time (years)")
    ax1.set_ylabel("Stock Price at Exercise")
    ax1.set_title("Early Exercise Events by Regime", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, T)

    # ── Panel 2: Exercise boundary curves ────────────────────────────
    ax2 = axes[1]
    for r_val in [0, 1]:
        steps, prices = boundary[r_val]
        if len(steps) > 0:
            times = steps * dt
            ax2.plot(
                times, prices,
                color=regime_colors[r_val],
                linewidth=2.5,
                label=regime_labels[r_val],
                marker="o",
                markersize=3,
                alpha=0.85,
            )

    ax2.axhline(K, color="grey", linestyle="--", linewidth=1, alpha=0.7, label=f"Strike K = {K:.0f}")
    ax2.set_xlabel("Time (years)")
    ax2.set_ylabel("Critical Stock Price S*(t)")
    ax2.set_title("Exercise Boundary S*(t) by Regime", fontsize=14, fontweight="bold")
    ax2.legend(loc="lower right", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, T)

    # ── Panel 3: Histogram of exercise times ─────────────────────────
    ax3 = axes[2]
    bins = np.linspace(0, T, 30)
    for r_val in [0, 1]:
        mask = events["regime"] == r_val
        if mask.sum() > 0:
            ax3.hist(
                events["time_years"][mask],
                bins=bins,
                color=regime_colors[r_val],
                alpha=0.5,
                label=regime_labels[r_val],
                edgecolor="white",
                linewidth=0.5,
            )

    ax3.set_xlabel("Time (years)")
    ax3.set_ylabel("Number of Exercises")
    ax3.set_title("Distribution of Exercise Times by Regime", fontsize=14, fontweight="bold")
    ax3.legend(loc="upper right", fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, T)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"    Figure saved to {save_path}")

    plt.show()


def print_exercise_statistics(events, params, K):
    """Print a formatted summary of exercise behaviour by regime."""
    N = params["N"]
    dt = params["dt"]
    n_total = len(events["time_step"])

    print(f"\n  {'─' * 60}")
    print(f"  Exercise Statistics (K = {K:.0f})")
    print(f"  {'─' * 60}")
    print(f"    Total exercised paths: {n_total:,}")
    print(f"    Early exercises (t < T):  {events['is_early'].sum():,}")
    print(f"    At-maturity exercises:    {(~events['is_early']).sum():,}")

    for r_val, label in [(0, "Low-Vol"), (1, "High-Vol")]:
        mask = events["regime"] == r_val
        n = mask.sum()
        if n == 0:
            print(f"\n    {label} regime: no exercises")
            continue

        early_mask = mask & events["is_early"]
        prices = events["stock_price"][mask]
        times = events["time_years"][mask]

        print(f"\n    ── {label} Regime (state {r_val}) ──")
        print(f"      Exercises in this regime:       {n:,} ({n/n_total*100:.1f}%)")
        print(f"        of which early (t < T):       {early_mask.sum():,}")
        print(f"      Mean exercise time (years):     {times.mean():.3f}")
        print(f"      Mean exercise stock price:      {prices.mean():.2f}")
        print(f"      Max stock price at exercise:    {prices.max():.2f}")
        print(f"      Min stock price at exercise:    {prices.min():.2f}")

    print(f"\n  {'─' * 60}")


# ════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("  Early Exercise Boundary Analysis (Regime-Switching American Put)")
    print("=" * 70)

    # ── Parameters ───────────────────────────────────────────────────
    S0 = 36.0
    K = 40.0
    r = 0.06
    T = 1.0
    dt = 0.02
    n_paths = 100_000
    sigma_low = 0.15
    sigma_high = 0.40
    q12 = 0.05
    q21 = 0.10

    print(f"\n  Parameters:")
    print(f"    S0 = {S0}, K = {K}, r = {r}, T = {T}")
    print(f"    σ_low = {sigma_low}, σ_high = {sigma_high}")
    print(f"    q12 = {q12}, q21 = {q21}")
    print(f"    Paths = {n_paths:,}, Steps = {int(T/dt)}")

    # ── Simulate under Q ─────────────────────────────────────────────
    print("\n  Simulating Q-measure paths...")
    S, regime, params = simulate_regime_switching_gbm(
        S0=S0, r=r, T=T, dt=dt, n_paths=n_paths,
        sigma_low=sigma_low, sigma_high=sigma_high,
        q12=q12, q21=q21,
        measure="Q",
        seed=42,
        initial_regime="stationary",
    )

    # ── Run LSMC ─────────────────────────────────────────────────────
    print("  Running LSMC...")
    price, stderr, ex_matrix = lsmc_american_put(
        S, regime, params, K=K, use_regime_feature=True
    )
    print(f"    American Put Price = {price:.4f}  (SE: {stderr:.4f})")

    # ── Extract exercise events ──────────────────────────────────────
    print("\n  Extracting exercise events...")
    events = extract_exercise_events(S, regime, ex_matrix, params)

    # ── Compute exercise boundary ────────────────────────────────────
    boundary = compute_exercise_boundary(events, params["N"], by_regime=True)

    # ── Print statistics ─────────────────────────────────────────────
    print_exercise_statistics(events, params, K)

    # ── Plot ──────────────────────────────────────────────────────────
    print("\n  Generating plots...")
    save_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "report", "figures", "early_exercise_boundary.png"
    )
    plot_exercise_boundary(events, boundary, params, K, save_path=save_path)

    print("\n" + "=" * 70)
