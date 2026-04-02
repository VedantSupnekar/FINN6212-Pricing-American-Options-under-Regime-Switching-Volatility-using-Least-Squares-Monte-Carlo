import numpy as np
import matplotlib.pyplot as plt


def simulate_regime_switching_gbm(
    S0=36.0,
    r=0.06,
    T=1.0,
    dt=0.02,
    n_paths=100,
    sigma_low=0.15,
    sigma_high=0.40,
    p12=0.05,
    p21=0.10,
    seed=42,
    initial_regime=0,
):
    """
    Simulate stock-price paths under a 2-state Markov regime-switching GBM.

    Parameters
    ----------
    S0           : float – initial stock price
    r            : float – risk-free rate
    T            : float – time to maturity (years)
    dt           : float – time-step size
    n_paths      : int   – number of Monte-Carlo paths
    sigma_low    : float – volatility in the "calm" regime (state 0)
    sigma_high   : float – volatility in the "turbulent" regime (state 1)
    p12          : float – per-step prob of switching Low → High
    p21          : float – per-step prob of switching High → Low
    seed         : int | None – random seed (None for no seeding)
    initial_regime : int or "stationary"
        If int (0 or 1): all paths start in that regime.
        If "stationary": each path's initial regime is drawn from the
                         stationary distribution π = [p21, p12]/(p12+p21).

    Returns
    -------
    S       : ndarray (N+1, n_paths) – simulated stock prices
    regime  : ndarray (N+1, n_paths) – regime at each step (0=Low, 1=High)
    params  : dict – all input parameters for reference
    """
    N = int(T / dt)
    sigmas = np.array([sigma_low, sigma_high])

    transition_matrix = np.array([
        [1 - p12, p12],
        [p21,     1 - p21]
    ])

    if seed is not None:
        np.random.seed(seed)

    # ── Regime simulation ───────────────────────────────────────────
    regime = np.zeros((N + 1, n_paths), dtype=int)

    if initial_regime == "stationary":
        pi_stat = np.array([p21, p12]) / (p12 + p21)
        regime[0] = (np.random.uniform(size=n_paths) > pi_stat[0]).astype(int)
    else:
        regime[0] = initial_regime

    for t in range(1, N + 1):
        u = np.random.uniform(size=n_paths)
        for p in range(n_paths):
            current = regime[t - 1, p]
            if u[p] < transition_matrix[current, 0]:
                regime[t, p] = 0
            else:
                regime[t, p] = 1

    sigma_path = sigmas[regime]

    # ── Price simulation ────────────────────────────────────────────
    S = np.zeros((N + 1, n_paths))
    S[0] = S0

    for t in range(1, N + 1):
        Z = np.random.standard_normal(n_paths)
        sig = sigma_path[t, :]
        S[t] = S[t - 1] * np.exp((r - 0.5 * sig**2) * dt + sig * np.sqrt(dt) * Z)

    params = dict(
        S0=S0, r=r, T=T, dt=dt, N=N, n_paths=n_paths,
        sigma_low=sigma_low, sigma_high=sigma_high,
        p12=p12, p21=p21, seed=seed,
        transition_matrix=transition_matrix,
    )

    return S, regime, params


# ════════════════════════════════════════════════════════════════════════
#  When run directly: simulate & plot (same behaviour as before)
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    S, regime, params = simulate_regime_switching_gbm()
    N = params["N"]
    n_paths = params["n_paths"]

    # ── Console output ──────────────────────────────────────────────
    print("Final prices of the first 5 paths:", S[-1, :5])
    print("\nRegime at final step for first 5 paths:",
          ["Low" if r == 0 else "High" for r in regime[-1, :5]])

    # ── Plot ────────────────────────────────────────────────────────
    highlight_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    n_highlight = min(5, n_paths)

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    ax1 = axes[0]
    for p_idx in range(n_paths):
        ax1.plot(S[:, p_idx], color="lightgrey", alpha=0.5, linewidth=0.5)
    for p_idx in range(n_highlight):
        ax1.plot(S[:, p_idx], color=highlight_colors[p_idx],
                 alpha=0.9, linewidth=1.4, label=f"Path {p_idx + 1}")
    ax1.set_title("Regime-Switching GBM Paths  (paths 1-5 highlighted)")
    ax1.set_ylabel("Asset Price")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    for p_idx in range(n_highlight):
        ax2.step(range(N + 1), regime[:, p_idx] + p_idx * 1.2,
                 where="mid", linewidth=1.4, color=highlight_colors[p_idx],
                 label=f"Path {p_idx + 1}")
    ax2.set_yticks([])
    ax2.set_title("Regime State over Time  (low = Low-Vol, high = High-Vol)")
    ax2.set_xlabel("Time Steps")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
