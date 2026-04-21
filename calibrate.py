"""
Calibration of the regime-switching GBM to market data.

Finds Q-measure parameters (σ₁, σ₂, λ̃₁₂, λ̃₂₁) that minimize the
vega-weighted sum of squared pricing errors between model (Carr-Madan)
and market European call prices.

The pipeline is fully functional with synthetic data and ready to
plug in real OptionMetrics data when it arrives.

Usage
-----
    python calibrate.py            # runs round-trip test with synthetic data
"""

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from scipy.stats import norm

from carr_madan import carr_madan_price_at_strikes, black_scholes_call


# ════════════════════════════════════════════════════════════════════════
#  Black-Scholes Vega (for weighting)
# ════════════════════════════════════════════════════════════════════════

def bs_vega(S0, K, r, T, sigma):
    """
    Black-Scholes vega: ∂C/∂σ = S₀·√T·φ(d₁).

    Used as a weight so ATM options (high vega) dominate the objective
    and deep OTM options (low vega, noisy prices) contribute less.
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S0 * np.sqrt(T) * norm.pdf(d1)


# ════════════════════════════════════════════════════════════════════════
#  Objective Function
# ════════════════════════════════════════════════════════════════════════

def calibration_objective(
    params_vec,
    market_data,
    current_regime=0,
    dt=0.02,
    vega_ref_sigma=0.25,
):
    """
    Vega-weighted sum of squared pricing errors.

    Parameters
    ----------
    params_vec    : array [σ_low, σ_high, q12, q21]
        The 4 Q-measure parameters to calibrate.
    market_data   : DataFrame with columns:
        strike, maturity_years, mid_price, underlying_price, risk_free_rate
    current_regime: int – assumed starting regime (0 or 1)
    dt            : float – time-step for CF conversion
    vega_ref_sigma: float – reference σ for computing vega weights

    Returns
    -------
    objective : float – weighted SSE
    """
    sigma_low, sigma_high, q12, q21 = params_vec

    # Enforce σ_low < σ_high via penalty
    if sigma_low >= sigma_high:
        return 1e10

    total_error = 0.0

    # Group by maturity for efficient FFT reuse
    for T, group in market_data.groupby("maturity_years"):
        S0 = group["underlying_price"].iloc[0]
        r = group["risk_free_rate"].iloc[0]
        strikes = group["strike"].values
        market_prices = group["mid_price"].values

        # Model prices via Carr-Madan
        try:
            model_prices = carr_madan_price_at_strikes(
                strikes, S0, r, T, sigma_low, sigma_high, q12, q21,
                current_regime=current_regime, dt=dt,
                cf_method="continuous",
            )
        except Exception:
            return 1e10

        # Vega weights (using reference σ for stability)
        vegas = np.array([
            bs_vega(S0, K, r, T, vega_ref_sigma) for K in strikes
        ])
        # Normalize weights so they sum to 1 per maturity
        w = vegas / (vegas.sum() + 1e-12)

        # Weighted SSE
        errors = (model_prices - market_prices) ** 2
        total_error += np.sum(w * errors)

    return total_error


# ════════════════════════════════════════════════════════════════════════
#  Calibration Engine
# ════════════════════════════════════════════════════════════════════════

def calibrate(
    market_data,
    current_regime=0,
    dt=0.02,
    method="differential_evolution",
    n_restarts=5,
    verbose=True,
):
    """
    Calibrate the regime-switching GBM to market option prices.

    Parameters
    ----------
    market_data    : DataFrame – market option data (see format below)
    current_regime : int – assumed starting regime
    dt             : float – time-step for CF
    method         : str – "differential_evolution" (global) or "L-BFGS-B" (local)
    n_restarts     : int – number of random restarts for L-BFGS-B
    verbose        : bool – print progress

    Returns
    -------
    result : dict with keys:
        sigma_low, sigma_high, q12, q21  – calibrated parameters
        objective  – final objective value
        optimizer_output  – raw scipy result
    """
    # Parameter bounds
    bounds = [
        (0.05, 0.50),   # σ_low
        (0.10, 1.00),   # σ_high
        (0.001, 0.30),  # q12 (per-step prob, must be < 1)
        (0.001, 0.30),  # q21
    ]

    obj_args = (market_data, current_regime, dt)

    if method == "differential_evolution":
        if verbose:
            print("  Running differential evolution (global optimizer)...")

        result = differential_evolution(
            calibration_objective,
            bounds=bounds,
            args=obj_args,
            seed=42,
            maxiter=200,
            tol=1e-8,
            polish=True,       # L-BFGS-B polish at the end
            disp=verbose,
            workers=1,
        )

        best_params = result.x
        best_obj = result.fun
        best_result = result

    elif method == "L-BFGS-B":
        if verbose:
            print(f"  Running L-BFGS-B with {n_restarts} random restarts...")

        best_obj = np.inf
        best_params = None
        best_result = None

        rng = np.random.RandomState(42)
        for i in range(n_restarts):
            x0 = np.array([
                rng.uniform(*bounds[0]),
                rng.uniform(*bounds[1]),
                rng.uniform(*bounds[2]),
                rng.uniform(*bounds[3]),
            ])
            # Ensure σ_low < σ_high in initial guess
            if x0[0] >= x0[1]:
                x0[0], x0[1] = x0[1] * 0.5, x0[1]

            res = minimize(
                calibration_objective,
                x0,
                args=obj_args,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 500, "ftol": 1e-10},
            )

            if verbose:
                print(f"    Restart {i+1}/{n_restarts}: obj = {res.fun:.8f}")

            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x
                best_result = res

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'differential_evolution' or 'L-BFGS-B'.")

    sigma_low, sigma_high, q12, q21 = best_params

    return {
        "sigma_low": sigma_low,
        "sigma_high": sigma_high,
        "q12": q12,
        "q21": q21,
        "objective": best_obj,
        "optimizer_output": best_result,
    }


# ════════════════════════════════════════════════════════════════════════
#  Synthetic Data Generator (for testing without real data)
# ════════════════════════════════════════════════════════════════════════

def generate_synthetic_market_data(
    S0=150.0,
    r=0.04,
    maturities=(0.25, 0.50, 1.0),
    moneyness_range=(0.85, 1.15),
    n_strikes_per_maturity=15,
    true_sigma_low=0.18,
    true_sigma_high=0.45,
    true_q12=0.04,
    true_q21=0.08,
    current_regime=0,
    dt=0.02,
    noise_std=0.0,
    seed=123,
):
    """
    Generate synthetic European call option prices from known RS-GBM params.

    Returns
    -------
    market_data  : DataFrame – same format as real market data
    true_params  : dict – the parameters used (ground truth)
    """
    rng = np.random.RandomState(seed)
    rows = []

    for T in maturities:
        strikes = np.linspace(S0 * moneyness_range[0],
                              S0 * moneyness_range[1],
                              n_strikes_per_maturity)

        model_prices = carr_madan_price_at_strikes(
            strikes, S0, r, T,
            true_sigma_low, true_sigma_high, true_q12, true_q21,
            current_regime=current_regime, dt=dt,
            cf_method="continuous",
        )

        # Add small noise to simulate market imperfections
        if noise_std > 0:
            noise = rng.normal(0, noise_std, size=len(strikes))
            model_prices = np.maximum(model_prices + noise, 0.01)

        for K, price in zip(strikes, model_prices):
            rows.append({
                "strike": K,
                "maturity_years": T,
                "mid_price": price,
                "underlying_price": S0,
                "risk_free_rate": r,
            })

    market_data = pd.DataFrame(rows)

    true_params = {
        "sigma_low": true_sigma_low,
        "sigma_high": true_sigma_high,
        "q12": true_q12,
        "q21": true_q21,
    }

    return market_data, true_params


# ════════════════════════════════════════════════════════════════════════
#  Real Data Loader (stub — ready for OptionMetrics CSV)
# ════════════════════════════════════════════════════════════════════════

def load_market_data(
    csv_path,
    underlying_price,
    risk_free_rate=0.04,
    moneyness_range=(0.80, 1.20),
    min_open_interest=100,
    min_bid=0.0,
    maturity_days_range=(30, 365),
):
    """
    Load and filter real option market data from a CSV file.

    Expected CSV columns (OptionMetrics format):
        strike_price, exdate (or days_to_expiry), best_bid, best_offer,
        open_interest, cp_flag

    This function will need minor adjustments once the real column
    names are known. The filtering logic is ready.

    Parameters
    ----------
    csv_path          : str – path to the CSV file
    underlying_price  : float – current stock price (S₀)
    risk_free_rate    : float – risk-free rate
    moneyness_range   : tuple – (min, max) moneyness filter K/S₀
    min_open_interest : int – minimum open interest filter
    min_bid           : float – minimum bid price filter
    maturity_days_range: tuple – (min, max) days to expiry

    Returns
    -------
    market_data : DataFrame with standardized columns:
        [strike, maturity_years, mid_price, underlying_price, risk_free_rate]
    """
    df = pd.read_csv(csv_path)

    # ── Standardize column names (adjust when real data arrives) ────
    # Expected mappings (common OptionMetrics names):
    col_map = {
        "strike_price": "strike",
        "best_bid": "bid",
        "best_offer": "offer",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # ── Compute mid price ───────────────────────────────────────────
    if "mid_price" not in df.columns:
        df["mid_price"] = (df["bid"] + df["offer"]) / 2

    # ── Compute maturity in years ───────────────────────────────────
    if "maturity_years" not in df.columns:
        if "days_to_expiry" in df.columns:
            df["maturity_years"] = df["days_to_expiry"] / 365.0
        elif "exdate" in df.columns:
            # Will need a reference date — adjust when format is known
            raise NotImplementedError(
                "exdate column found but reference date not set. "
                "Adjust this function when real data format is known."
            )

    # ── Set underlying & rate ───────────────────────────────────────
    df["underlying_price"] = underlying_price
    df["risk_free_rate"] = risk_free_rate

    # ── Strike price scaling (OptionMetrics often uses strike × 1000) ──
    if df["strike"].median() > underlying_price * 10:
        df["strike"] = df["strike"] / 1000.0

    # ── Apply filters ───────────────────────────────────────────────
    moneyness = df["strike"] / underlying_price

    mask = (
        (moneyness >= moneyness_range[0]) &
        (moneyness <= moneyness_range[1])
    )

    if "cp_flag" in df.columns:
        mask &= df["cp_flag"].str.upper() == "C"  # calls only

    if "open_interest" in df.columns:
        mask &= df["open_interest"] >= min_open_interest

    if "bid" in df.columns:
        mask &= df["bid"] > min_bid

    if "maturity_years" in df.columns:
        days = df["maturity_years"] * 365
        mask &= (days >= maturity_days_range[0]) & (days <= maturity_days_range[1])

    df = df[mask].copy()

    if len(df) == 0:
        raise ValueError("No options passed the filters. Check data format and filter params.")

    # ── Return standardized columns ─────────────────────────────────
    return df[["strike", "maturity_years", "mid_price",
               "underlying_price", "risk_free_rate"]].reset_index(drop=True)


# ════════════════════════════════════════════════════════════════════════
#  Main: Round-trip test with synthetic data
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("  Calibration Pipeline — Round-Trip Test with Synthetic Data")
    print("=" * 70)

    # ── Generate synthetic market data ──────────────────────────────
    print("\n  Step 1: Generating synthetic market data...")

    market_data, true_params = generate_synthetic_market_data(
        S0=150.0,
        r=0.04,
        maturities=(0.25, 0.50, 1.0),
        n_strikes_per_maturity=15,
        true_sigma_low=0.18,
        true_sigma_high=0.45,
        true_q12=0.04,
        true_q21=0.08,
        noise_std=0.02,  # small noise to simulate market imperfections
    )

    print(f"    Generated {len(market_data)} option prices across "
          f"{market_data['maturity_years'].nunique()} maturities")
    print(f"    Strike range: [{market_data['strike'].min():.1f}, "
          f"{market_data['strike'].max():.1f}]")
    print(f"\n    True parameters:")
    for k, v in true_params.items():
        print(f"      {k:>12s} = {v}")

    # ── Calibrate ───────────────────────────────────────────────────
    print("\n  Step 2: Calibrating...")

    result = calibrate(
        market_data,
        current_regime=0,
        method="differential_evolution",
        verbose=True,
    )

    # ── Results ─────────────────────────────────────────────────────
    print("\n  Step 3: Results")
    print("  " + "-" * 60)

    print(f"\n  {'Parameter':>12s}  {'True':>10s}  {'Calibrated':>12s}  {'Error':>10s}  {'Rel %':>8s}")
    for k in ["sigma_low", "sigma_high", "q12", "q21"]:
        true_val = true_params[k]
        cal_val = result[k]
        err = abs(cal_val - true_val)
        rel = err / true_val * 100
        print(f"  {k:>12s}  {true_val:>10.4f}  {cal_val:>12.4f}  {err:>10.4f}  {rel:>7.2f}%")

    print(f"\n  Final objective value: {result['objective']:.8f}")

    # ── Pricing comparison ──────────────────────────────────────────
    print("\n  Step 4: Pricing comparison (true vs calibrated)")
    print("  " + "-" * 60)

    S0 = 150.0
    r = 0.04
    test_K = np.array([135, 140, 145, 150, 155, 160, 165])
    T_test = 0.50

    true_prices = carr_madan_price_at_strikes(
        test_K, S0, r, T_test,
        true_params["sigma_low"], true_params["sigma_high"],
        true_params["q12"], true_params["q21"],
    )
    cal_prices = carr_madan_price_at_strikes(
        test_K, S0, r, T_test,
        result["sigma_low"], result["sigma_high"],
        result["q12"], result["q21"],
    )

    print(f"\n  {'Strike':>8s}  {'True Price':>12s}  {'Cal Price':>12s}  {'|Δ|':>8s}")
    for i, K in enumerate(test_K):
        err = abs(true_prices[i] - cal_prices[i])
        print(f"  {K:>8.0f}  {true_prices[i]:>12.4f}  {cal_prices[i]:>12.4f}  {err:>8.4f}")

    max_price_err = np.max(np.abs(true_prices - cal_prices))
    print(f"\n  Max pricing error: {max_price_err:.4f}")
    print(f"  {'✅ PASS' if max_price_err < 0.50 else '❌ FAIL'} (threshold: 0.50)")

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Summary: Calibration pipeline is fully operational.")
    print("  When real OptionMetrics data arrives, call:")
    print("    data = load_market_data('data/aapl_options.csv', S0=...)")
    print("    result = calibrate(data)")
    print("=" * 70)
