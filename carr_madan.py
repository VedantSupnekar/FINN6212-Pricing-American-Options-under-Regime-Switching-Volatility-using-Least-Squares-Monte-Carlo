"""
Carr-Madan FFT European option pricing under regime-switching GBM.

Uses the characteristic function from characteristic_function.py to price
European calls across a grid of strikes via the Fast Fourier Transform.

Key formula (Carr & Madan, 1999):
    C(k) = (e^{-αk} / π) · Re[ ∫₀^∞ e^{-ivk} · ψ(v) dv ]

where k = ln(K), and:
    ψ(v) = e^{-rT} · φ(v - (α+1)i) / [α² + α - v² + i(2α+1)v]

The integral is evaluated via FFT with Simpson's rule weights.

References:
    Carr, P. & Madan, D.B. (1999), "Option valuation using the fast Fourier
    transform", Journal of Computational Finance, 2(4), 61-73.
"""

import numpy as np
from characteristic_function import characteristic_function


def carr_madan_fft(
    S0,
    r,
    T,
    sigma_low,
    sigma_high,
    q12,
    q21,
    current_regime=0,
    dt=0.02,
    cf_method="continuous",
    alpha=1.5,
    N=4096,
    eta=0.25,
):
    """
    Price European calls on a grid of strikes using FFT.

    Parameters
    ----------
    S0, r, T, sigma_low, sigma_high, q12, q21, current_regime, dt
        Model parameters passed through to characteristic_function().
    cf_method   : str – "continuous" or "discrete" (CF evaluation method)
    alpha       : float – damping parameter (must satisfy α > 0 and the
                  integrability condition; typically 1.0–2.0)
    N           : int – FFT grid size (power of 2)
    eta         : float – integration step size in Fourier space

    Returns
    -------
    strikes     : ndarray – strike prices (positive, sorted)
    call_prices : ndarray – corresponding European call prices
    log_strikes : ndarray – log-strike grid (k values)
    """
    # ── FFT grid setup ──────────────────────────────────────────────
    # Fourier-space grid:  v_j = j · η,   j = 0, 1, ..., N-1
    # Log-strike grid:     k_n = -b + n · Δk,  n = 0, 1, ..., N-1
    # where  Δk = 2π / (N · η)  and  b = N · Δk / 2
    dk = 2 * np.pi / (N * eta)
    b = N * dk / 2

    # Fourier-space grid points
    v = np.arange(N) * eta  # v_0 = 0, v_1 = η, ...

    # Log-strike grid points
    log_strikes = -b + np.arange(N) * dk

    # ── Evaluate ψ(v) on the grid ───────────────────────────────────
    # ψ(v) = e^{-rT} · φ(v - (α+1)i) / [α² + α - v² + i(2α+1)v]
    #
    # Note: we need φ evaluated at u = v - (α+1)i  (complex argument)
    u_complex = v - (alpha + 1) * 1j

    # Evaluate CF at all grid points
    phi_values = characteristic_function(
        u_complex, S0, r, T, sigma_low, sigma_high, q12, q21,
        current_regime=current_regime, dt=dt, method=cf_method,
    )

    # Denominator: α² + α - v² + i(2α+1)v
    denom = alpha**2 + alpha - v**2 + 1j * (2 * alpha + 1) * v

    # ψ(v) = e^{-rT} · φ(u_complex) / denom
    psi = np.exp(-r * T) * phi_values / denom

    # ── Simpson's rule weights ──────────────────────────────────────
    # w_j = η/3 · [1, 4, 2, 4, 2, ..., 4, 1]  (Simpson's 1/3 rule)
    simpson = np.ones(N)
    simpson[1::2] = 4  # odd indices
    simpson[2::2] = 2  # even indices
    simpson[0] = 1
    simpson[-1] = 1    # last point
    simpson *= eta / 3

    # ── FFT integrand ───────────────────────────────────────────────
    # The integral is:
    #   C(k_n) = (e^{-α·k_n} / π) · Re[ Σ_j e^{-iv_j·k_n} · ψ(v_j) · w_j ]
    #
    # Using the FFT trick:  k_n = -b + n·Δk, so
    #   e^{-iv_j k_n} = e^{iv_j b} · e^{-i·2π·j·n/N}  (the FFT kernel)
    #
    # Therefore the integrand for FFT is:
    #   x_j = e^{iv_j b} · ψ(v_j) · w_j
    x = np.exp(1j * v * b) * psi * simpson

    # ── Run FFT ─────────────────────────────────────────────────────
    fft_result = np.fft.fft(x)

    # ── Extract call prices ─────────────────────────────────────────
    # C(k_n) = (e^{-α·k_n} / π) · Re[fft_result_n]
    call_prices = (np.exp(-alpha * log_strikes) / np.pi) * fft_result.real

    # ── Convert log-strikes to strikes ──────────────────────────────
    strikes = np.exp(log_strikes)

    return strikes, call_prices, log_strikes


def carr_madan_price_at_strikes(
    K_targets,
    S0, r, T, sigma_low, sigma_high, q12, q21,
    current_regime=0, dt=0.02, cf_method="continuous",
    alpha=1.5, N=4096, eta=0.25,
):
    """
    Price European calls at specific strike values via interpolation.

    Parameters
    ----------
    K_targets   : float or array-like – desired strike prices

    Returns
    -------
    prices      : float or ndarray – European call prices at K_targets
    """
    K_targets = np.atleast_1d(np.asarray(K_targets, dtype=float))

    strikes, call_prices, _ = carr_madan_fft(
        S0, r, T, sigma_low, sigma_high, q12, q21,
        current_regime=current_regime, dt=dt, cf_method=cf_method,
        alpha=alpha, N=N, eta=eta,
    )

    # Interpolate onto requested strikes
    prices = np.interp(K_targets, strikes, call_prices)

    return prices if len(prices) > 1 else prices[0]


# ════════════════════════════════════════════════════════════════════════
#  Black-Scholes analytical price (for validation)
# ════════════════════════════════════════════════════════════════════════

def black_scholes_call(S0, K, r, T, sigma):
    """Analytical Black-Scholes European call price."""
    from scipy.stats import norm
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_put(S0, K, r, T, sigma):
    """Analytical Black-Scholes European put price."""
    from scipy.stats import norm
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)


# ════════════════════════════════════════════════════════════════════════
#  Validation
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from simulate_gbm import simulate_regime_switching_gbm

    S0 = 36.0
    r = 0.06
    T = 1.0
    dt = 0.02
    sigma_low = 0.15
    sigma_high = 0.40
    q12 = 0.05
    q21 = 0.10
    regime = 0

    print("=" * 70)
    print("  Carr-Madan FFT European Option Pricing — Validation")
    print("=" * 70)

    # ── TEST 1: Black-Scholes limit (σ₁ = σ₂ = σ) ──────────────────
    print("\n  TEST 1: Black-Scholes limit (σ₁ = σ₂ = 0.20)")
    print("  " + "-" * 60)

    sigma_bs = 0.20
    test_strikes = np.array([30, 32, 34, 36, 38, 40, 42, 44, 46])

    cm_prices = carr_madan_price_at_strikes(
        test_strikes, S0, r, T, sigma_bs, sigma_bs, q12, q21,
        current_regime=0, dt=dt, cf_method="continuous",
    )
    bs_prices = np.array([black_scholes_call(S0, K, r, T, sigma_bs) for K in test_strikes])

    print(f"\n  {'Strike':>8s}  {'CM Price':>10s}  {'BS Price':>10s}  {'Error':>10s}  {'Rel %':>8s}")
    for i, K in enumerate(test_strikes):
        err = abs(cm_prices[i] - bs_prices[i])
        rel = err / bs_prices[i] * 100 if bs_prices[i] > 0.01 else 0
        print(f"  {K:>8.0f}  {cm_prices[i]:>10.4f}  {bs_prices[i]:>10.4f}  {err:>10.6f}  {rel:>7.3f}%")

    max_err = np.max(np.abs(cm_prices - bs_prices))
    print(f"\n  Max absolute error: {max_err:.6f}")
    print(f"  {'✅ PASS' if max_err < 0.01 else '❌ FAIL'} (threshold: 0.01)")

    # ── TEST 2: Regime-switching prices ──────────────────────────────
    print("\n\n  TEST 2: Regime-switching call prices (σ₁=0.15, σ₂=0.40)")
    print("  " + "-" * 60)

    cm_rs_prices = carr_madan_price_at_strikes(
        test_strikes, S0, r, T, sigma_low, sigma_high, q12, q21,
        current_regime=regime, dt=dt, cf_method="continuous",
    )

    print(f"\n  {'Strike':>8s}  {'CM (RS)':>10s}  {'CM (BS σ=0.20)':>14s}  {'Diff':>10s}")
    for i, K in enumerate(test_strikes):
        diff = cm_rs_prices[i] - cm_prices[i]
        print(f"  {K:>8.0f}  {cm_rs_prices[i]:>10.4f}  {cm_prices[i]:>14.4f}  {diff:>+10.4f}")

    # ── TEST 3: Monte Carlo cross-check ─────────────────────────────
    print("\n\n  TEST 3: Cross-check against Monte Carlo European calls")
    print("  " + "-" * 60)

    n_mc = 500_000
    print(f"  Running Monte Carlo ({n_mc:,} paths)...")
    S, reg, params = simulate_regime_switching_gbm(
        S0=S0, r=r, T=T, dt=dt, n_paths=n_mc,
        sigma_low=sigma_low, sigma_high=sigma_high,
        q12=q12, q21=q21, measure="Q",
        seed=42, initial_regime=regime,
    )
    S_T = S[-1, :]

    mc_strikes = np.array([32, 34, 36, 38, 40, 42])
    print(f"\n  {'Strike':>8s}  {'CM (RS)':>10s}  {'MC Price':>10s}  {'MC SE':>8s}  {'|Δ|':>8s}")
    for K in mc_strikes:
        # MC European call: e^{-rT} · E[max(S_T - K, 0)]
        payoffs = np.maximum(S_T - K, 0) * np.exp(-r * T)
        mc_price = np.mean(payoffs)
        mc_se = np.std(payoffs) / np.sqrt(n_mc)

        cm_price = carr_madan_price_at_strikes(
            K, S0, r, T, sigma_low, sigma_high, q12, q21,
            current_regime=regime, dt=dt, cf_method="continuous",
        )
        err = abs(cm_price - mc_price)
        print(f"  {K:>8.0f}  {cm_price:>10.4f}  {mc_price:>10.4f}  {mc_se:>8.4f}  {err:>8.4f}")

    # ── TEST 4: Put-call parity ─────────────────────────────────────
    print("\n\n  TEST 4: Put-call parity verification")
    print("  " + "-" * 60)
    print("  C - P = S₀ - K·e^{-rT}  (for European options)")

    # Get call prices from Carr-Madan
    parity_strikes = np.array([34, 36, 38, 40, 42])
    cm_calls = carr_madan_price_at_strikes(
        parity_strikes, S0, r, T, sigma_low, sigma_high, q12, q21,
        current_regime=regime, dt=dt,
    )

    # Compute put from parity: P = C - S₀ + K·e^{-rT}
    cm_puts_parity = cm_calls - S0 + parity_strikes * np.exp(-r * T)

    # MC puts for comparison
    print(f"\n  {'Strike':>8s}  {'Call (CM)':>10s}  {'Put (parity)':>12s}  {'Put (MC)':>10s}  {'|Δ|':>8s}")
    for i, K in enumerate(parity_strikes):
        put_payoffs = np.maximum(K - S_T, 0) * np.exp(-r * T)
        mc_put = np.mean(put_payoffs)
        err = abs(cm_puts_parity[i] - mc_put)
        print(f"  {K:>8.0f}  {cm_calls[i]:>10.4f}  {cm_puts_parity[i]:>12.4f}  {mc_put:>10.4f}  {err:>8.4f}")

    print("\n" + "=" * 70)
    print("  All Carr-Madan validation tests complete.")
    print("=" * 70)
