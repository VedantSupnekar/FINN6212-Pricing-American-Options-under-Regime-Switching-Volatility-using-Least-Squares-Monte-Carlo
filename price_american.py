"""
Pricing American Put Options under Calibrated Regime-Switching Volatility.

This script demonstrates the end-to-end pipeline:
1. Generates synthetic market data (acting as a placeholder until real data arrives).
2. Calibrates the Q-measure parameters (σ₁, σ₂, q₁₂, q₂₁) to the market data.
3. Simulates stock price paths under the Q-measure using the calibrated parameters.
4. Prices an American put option using the LSMC algorithm, starting from each regime.
"""

import numpy as np

from calibrate import generate_synthetic_market_data, calibrate
from simulate_gbm import simulate_regime_switching_gbm
from lsmc import lsmc_american_put

def main():
    print("=" * 70)
    print("  End-to-End Pipeline: Calibration to American Option Pricing")
    print("=" * 70)

    # ── Step 1: Obtain "Market" Data ──────────────────────────────────────
    print("\n  Step 1: Generating synthetic market data...")
    # Using default synthetic parameters
    market_data, true_params = generate_synthetic_market_data(
        S0=150.0,
        r=0.04,
        maturities=(0.25, 0.50, 1.0),
        n_strikes_per_maturity=15,
        noise_std=0.01,
        seed=123
    )
    print(f"    Generated {len(market_data)} option prices.")

    # ── Step 2: Calibrate to Market Data ──────────────────────────────────
    print("\n  Step 2: Calibrating Q-measure parameters to market data...")
    calib_result = calibrate(
        market_data,
        method="differential_evolution",
        verbose=False
    )
    
    sigma_low = calib_result["sigma_low"]
    sigma_high = calib_result["sigma_high"]
    q12 = calib_result["q12"]
    q21 = calib_result["q21"]

    print(f"    Calibrated σ_low  = {sigma_low:.4f}")
    print(f"    Calibrated σ_high = {sigma_high:.4f}")
    print(f"    Calibrated q_12   = {q12:.4f}")
    print(f"    Calibrated q_21   = {q21:.4f}")

    # ── Step 3 & 4: Simulate under Q and Price via LSMC ───────────────────
    print("\n  Step 3 & 4: Simulating Q-measure paths and running LSMC...")
    
    S0 = 150.0
    K = 150.0  # ATM put
    r = 0.04
    T = 1.0
    dt = 0.02
    n_paths = 50_000  # adjust for speed vs accuracy

    print(f"\n    Option Details:")
    print(f"      S0 = {S0}, K = {K}, r = {r}, T = {T}, dt = {dt}, Paths = {n_paths:,}")

    for initial_regime in [0, 1]:
        regime_name = "Low-Vol (0)" if initial_regime == 0 else "High-Vol (1)"
        print(f"\n    ── Starting in {regime_name} Regime ──")
        
        # Simulate paths under Q
        S, regime, params = simulate_regime_switching_gbm(
            S0=S0, r=r, T=T, dt=dt, n_paths=n_paths,
            sigma_low=sigma_low, sigma_high=sigma_high,
            p12=q12, p21=q21,  # P parameters are ignored if measure="Q" and q params are passed
            q12=q12, q21=q21,
            measure="Q",
            seed=42,
            initial_regime=initial_regime
        )

        # Price American Put via LSMC
        price, stderr, ex_matrix = lsmc_american_put(
            S, regime, params, K=K, use_regime_feature=True
        )

        print(f"      American Put Price = {price:.4f}  (SE: {stderr:.4f})")
        print(f"      95% CI             = [{price - 1.96*stderr:.4f}, {price + 1.96*stderr:.4f}]")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
