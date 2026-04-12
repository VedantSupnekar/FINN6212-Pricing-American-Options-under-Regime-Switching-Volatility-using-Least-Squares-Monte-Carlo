"""
Least-Squares Monte Carlo (LSMC) pricing of American put options
under a 2-state Markov regime-switching GBM.

Extends the Longstaff & Schwartz (2001) algorithm by adding a regime
indicator as an extra regression feature:
    Standard L&S basis:  [1, S_t, S_t²]
    Regime-aware basis:  [1, S_t, S_t², 1_{regime=1}]

Usage
-----
    python lsmc.py                  # uses default parameters
"""

import numpy as np
from simulate_gbm import simulate_regime_switching_gbm


def lsmc_american_put(
    S,
    regime,
    params,
    K=40.0,
    use_regime_feature=True,
):
    """
    Price an American put option via LSMC backward induction.

    Parameters
    ----------
    S       : ndarray (N+1, n_paths) – simulated stock prices
    regime  : ndarray (N+1, n_paths) – regime at each step (0 or 1)
    params  : dict – simulation parameters (must contain 'r', 'dt', 'N', 'n_paths')
    K       : float – strike price
    use_regime_feature : bool – if True, include regime indicator in the
                                regression basis (the project's extension)

    Returns
    -------
    price   : float – estimated American put option price
    stderr  : float – Monte-Carlo standard error
    exercise_matrix : ndarray (N+1, n_paths) – 1 where exercise occurs, 0 otherwise
    """
    r = params["r"]
    dt = params["dt"]
    N = params["N"]
    n_paths = params["n_paths"]
    discount = np.exp(-r * dt)

    # ── Step 1: Cash-flow matrix ────────────────────────────────────
    # cashflow[t, p] = cash flow received at time t on path p
    # (will be updated backwards; initially only maturity is set)
    cashflow = np.zeros((N + 1, n_paths))
    exercise = np.zeros((N + 1, n_paths), dtype=int)

    # At maturity: exercise value = max(K - S_T, 0)
    cashflow[N] = np.maximum(K - S[N], 0.0)
    exercise[N] = (cashflow[N] > 0).astype(int)

    # ── Step 2: Backward induction ──────────────────────────────────
    for t in range(N - 1, 0, -1):  # t = N-1, N-2, ..., 1  (skip t=0)
        # Intrinsic value at time t
        intrinsic = np.maximum(K - S[t], 0.0)

        # Only consider in-the-money paths for regression
        itm = intrinsic > 0
        n_itm = np.sum(itm)

        # Need enough ITM paths for a stable regression
        min_paths = 5 if not use_regime_feature else 6
        if n_itm < min_paths:
            continue

        # ── Build regression features ───────────────────────────────
        # Normalize stock prices by K for numerical stability
        S_itm = S[t, itm] / K
        ones = np.ones(n_itm)
        X = np.column_stack([ones, S_itm, S_itm ** 2])

        if use_regime_feature:
            regime_indicator = regime[t, itm].astype(float)
            # Only add regime column if there's variation (otherwise rank-deficient)
            if regime_indicator.min() != regime_indicator.max():
                X = np.column_stack([X, regime_indicator])

        # ── Discounted future cash flows (the "Y" for regression) ───
        # Sum all future cash flows on each path, discounted back to t
        # We only need the NEXT exercise/cash-flow time for each path.
        # Since we update in place, cashflow already reflects optimal
        # future decisions for t+1 .. N.
        # Find the FIRST future time with a non-zero cash flow for each path
        future_cf = np.zeros(n_paths)
        for s in range(t + 1, N + 1):
            mask = (cashflow[s] > 0) & (future_cf == 0)
            future_cf[mask] = cashflow[s, mask] * np.exp(-r * (s - t) * dt)

        Y = future_cf[itm]  # discounted continuation value for ITM paths

        # ── OLS regression ──────────────────────────────────────────
        # C_hat = X @ beta  (estimated continuation value)
        try:
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
                C_hat = X @ beta
            # If regression produced NaN (near-singular), skip this step
            if np.any(np.isnan(C_hat)):
                continue
        except np.linalg.LinAlgError:
            continue

        # ── Exercise decision ───────────────────────────────────────
        # Exercise now if intrinsic > estimated continuation
        exercise_now = intrinsic[itm] > C_hat

        # Update cash-flow matrix:
        #   - paths that exercise at t: set cashflow[t] = intrinsic,
        #     and zero out all future cash flows on those paths
        #   - paths that continue: leave as is
        itm_indices = np.where(itm)[0]
        exercise_indices = itm_indices[exercise_now]

        cashflow[t, exercise_indices] = intrinsic[exercise_indices]
        exercise[t, exercise_indices] = 1
        # Zero out future cash flows on paths that exercise at t
        for s in range(t + 1, N + 1):
            cashflow[s, exercise_indices] = 0.0
            exercise[s, exercise_indices] = 0

    # ── Step 3: Discount to t=0 & compute price ────────────────────
    # For each path, find the single exercise time and discount back
    path_values = np.zeros(n_paths)
    for p in range(n_paths):
        ex_times = np.where(exercise[:, p] == 1)[0]
        if len(ex_times) > 0:
            t_ex = ex_times[0]  # earliest exercise time
            path_values[p] = cashflow[t_ex, p] * np.exp(-r * t_ex * dt)

    price = np.mean(path_values)
    stderr = np.std(path_values) / np.sqrt(n_paths)

    return price, stderr, exercise


# ════════════════════════════════════════════════════════════════════════
#  Main: run LSMC and print results
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    K = 40.0
    n_paths = 100_000

    print("=" * 65)
    print("  American Put Pricing via LSMC (Regime-Switching GBM)")
    print("=" * 65)

    # ── Run with regime-aware features ──────────────────────────────
    S, regime, params = simulate_regime_switching_gbm(n_paths=n_paths, seed=42)

    price_regime, se_regime, ex_regime = lsmc_american_put(
        S, regime, params, K=K, use_regime_feature=True
    )

    print(f"\n  Parameters:")
    print(f"    S0 = {params['S0']},  K = {K},  r = {params['r']},  T = {params['T']}")
    print(f"    σ_low = {params['sigma_low']},  σ_high = {params['sigma_high']}")
    print(f"    p12 = {params['p12']},  p21 = {params['p21']}")
    print(f"    Paths = {n_paths:,},  Steps = {params['N']}")

    print(f"\n  ── With regime indicator feature ──")
    print(f"    American Put Price = {price_regime:.4f}")
    print(f"    Std Error          = {se_regime:.4f}")
    print(f"    95% CI             = [{price_regime - 1.96*se_regime:.4f}, "
          f"{price_regime + 1.96*se_regime:.4f}]")

    # ── Run WITHOUT regime feature (standard L&S baseline) ──────────
    price_std, se_std, ex_std = lsmc_american_put(
        S, regime, params, K=K, use_regime_feature=False
    )

    print(f"\n  ── Without regime feature (standard L&S) ──")
    print(f"    American Put Price = {price_std:.4f}")
    print(f"    Std Error          = {se_std:.4f}")
    print(f"    95% CI             = [{price_std - 1.96*se_std:.4f}, "
          f"{price_std + 1.96*se_std:.4f}]")

    diff = price_regime - price_std
    print(f"\n  ── Comparison ──")
    print(f"    Price difference (regime - standard) = {diff:+.4f}")
    print(f"    Mispricing from ignoring regime info = {abs(diff):.4f}")

    # ── Exercise statistics ─────────────────────────────────────────
    ex_times_regime = []
    for p in range(n_paths):
        ex = np.where(ex_regime[:, p] == 1)[0]
        if len(ex) > 0:
            ex_times_regime.append(ex[0])

    if ex_times_regime:
        ex_times_regime = np.array(ex_times_regime)
        print(f"\n  ── Exercise Statistics (regime-aware) ──")
        print(f"    Paths exercised early (before T) = "
              f"{np.sum(ex_times_regime < params['N']):,} / {n_paths:,}")
        print(f"    Mean exercise time step = {np.mean(ex_times_regime):.1f} / {params['N']}")

    print("\n" + "=" * 65)
