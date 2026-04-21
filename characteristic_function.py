"""
Characteristic function for the regime-switching GBM under the Q-measure.

Implements both:
  1. Continuous-time CF via 2×2 matrix exponential (analytical)
  2. Discrete-time CF via matrix power (matches Monte Carlo simulation)

The key result (Buffington & Elliott 2002, Duan-Popova-Ritchken 2002):

    φ_j(u, T) = exp(iu·x + iu·r·T) · [exp(M(u)·T) · 1]_j

where x = ln(S0), j = current regime, and:

    M(u) = G̃ + diag(a₀(u), a₁(u))

    G̃ = Q-measure generator matrix
    a_k(u) = -(iu + u²)/2 · σ_k²
"""

import numpy as np
from scipy.linalg import expm
from simulate_gbm import simulate_regime_switching_gbm


# ════════════════════════════════════════════════════════════════════════
#  Core: Characteristic Function
# ════════════════════════════════════════════════════════════════════════

def characteristic_function(
    u,
    S0,
    r,
    T,
    sigma_low,
    sigma_high,
    q12,
    q21,
    current_regime=0,
    dt=0.02,
    method="continuous",
):
    """
    Compute φ_j(u) = E^Q[exp(iu·ln(S_T)) | s_0 = j, S_0].

    Parameters
    ----------
    u             : float or ndarray – CF argument (can be complex-valued)
    S0            : float – initial stock price
    r             : float – risk-free rate
    T             : float – time to maturity
    sigma_low     : float – volatility in regime 0 (calm)
    sigma_high    : float – volatility in regime 1 (turbulent)
    q12           : float – Q-measure per-step prob Low→High
    q21           : float – Q-measure per-step prob High→Low
    current_regime: int (0 or 1) – starting regime
    dt            : float – time-step size (used for discrete method and
                    for converting discrete q's to continuous λ's)
    method        : str – "continuous" or "discrete"

    Returns
    -------
    phi : complex or ndarray of complex – characteristic function value(s)
    """
    x = np.log(S0)
    sigmas = np.array([sigma_low, sigma_high])

    if method == "continuous":
        return _cf_continuous(u, x, r, T, sigmas, q12, q21, dt, current_regime)
    elif method == "discrete":
        return _cf_discrete(u, x, r, T, sigmas, q12, q21, dt, current_regime)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'continuous' or 'discrete'.")


def _cf_continuous(u, x, r, T, sigmas, q12, q21, dt, regime):
    """Continuous-time CF via vectorized 2×2 closed-form matrix exponential."""
    # Convert discrete-time transition probs to continuous-time intensities
    lam12 = -np.log(1 - q12) / dt
    lam21 = -np.log(1 - q21) / dt

    u = np.atleast_1d(np.asarray(u, dtype=complex))

    # α(u) = -(iu + u²) / 2   (vectorized over all u)
    alpha = -(1j * u + u**2) / 2.0

    # M(u) entries:  m11 = -λ12 + a0(u),  m12 = λ12,  m21 = λ21,  m22 = -λ21 + a1(u)
    a0 = alpha * sigmas[0]**2
    a1 = alpha * sigmas[1]**2
    m11 = -lam12 + a0
    m22 = -lam21 + a1
    m12 = lam12   # scalar, broadcast
    m21 = lam21   # scalar, broadcast

    # Eigenvalues of M(u):  η± = (tr ± √(tr² - 4det)) / 2
    tr = m11 + m22
    det = m11 * m22 - m12 * m21
    disc = tr**2 - 4.0 * det
    sqrt_disc = np.sqrt(disc + 0j)

    eta_p = (tr + sqrt_disc) / 2.0
    eta_m = (tr - sqrt_disc) / 2.0
    diff = eta_p - eta_m

    # exp(M·T)·1 using Sylvester formula:
    # exp(MT) = [(η₊e^{η₋T} - η₋e^{η₊T})·I + (e^{η₊T} - e^{η₋T})·M] / (η₊ - η₋)
    # We only need [exp(MT)·1]_j, so expand:
    #   f₀ = [η₊e^{η₋T}-η₋e^{η₊T} + (e^{η₊T}-e^{η₋T})·(m11+m12)] / diff
    #   f₁ = [η₊e^{η₋T}-η₋e^{η₊T} + (e^{η₊T}-e^{η₋T})·(m21+m22)] / diff
    with np.errstate(over="ignore", invalid="ignore"):
        exp_p = np.exp(eta_p * T)
        exp_m = np.exp(eta_m * T)

        coeff_I = eta_p * exp_m - eta_m * exp_p   # coefficient of I
        coeff_M = exp_p - exp_m                     # coefficient of M

        # Handle degenerate case (diff ≈ 0) carefully
        safe_diff = np.where(np.abs(diff) < 1e-14, 1.0, diff)

        f0 = (coeff_I + coeff_M * (m11 + m12)) / safe_diff
        f1 = (coeff_I + coeff_M * (m21 + m22)) / safe_diff

        # For degenerate eigenvalues: exp(MT)·1 ≈ exp(η·T)·(I + (M - ηI)T)·1
        degen_mask = np.abs(diff) < 1e-14
        if np.any(degen_mask):
            eta_avg = (eta_p[degen_mask] + eta_m[degen_mask]) / 2
            e_eta = np.exp(eta_avg * T)
            f0[degen_mask] = e_eta * (1 + (m11[degen_mask] - eta_avg + m12) * T)
            f1[degen_mask] = e_eta * (1 + (m21 + m22[degen_mask] - eta_avg) * T)

        # Select the regime row
        f = f0 if regime == 0 else f1

        # Full CF:  φ_j(u) = exp(iu·x + iu·r·T) · f_j(u)
        result = np.exp(1j * u * x + 1j * u * r * T) * f

    return result if len(result) > 1 else result[0]


def _cf_discrete(u, x, r, T, sigmas, q12, q21, dt, regime):
    """Discrete-time CF via (B · P̃)^N — matches Monte Carlo directly."""
    N = int(T / dt)

    # Q-measure transition matrix
    P_tilde = np.array([
        [1 - q12, q12],
        [q21,     1 - q21]
    ])

    u = np.atleast_1d(np.asarray(u, dtype=complex))
    result = np.empty(len(u), dtype=complex)

    for i, ui in enumerate(u):
        alpha = -(1j * ui + ui**2) / 2.0

        # b_k = exp(a_k · Δt)
        b0 = np.exp(alpha * sigmas[0]**2 * dt)
        b1 = np.exp(alpha * sigmas[1]**2 * dt)
        B = np.array([[b0, 0], [0, b1]], dtype=complex)

        # (B · P̃)^N · 1
        BP = B @ P_tilde
        BP_N = np.linalg.matrix_power(BP, N)
        f = BP_N @ np.ones(2)

        result[i] = np.exp(1j * ui * x + 1j * ui * r * T) * f[regime]

    return result if len(result) > 1 else result[0]


# ════════════════════════════════════════════════════════════════════════
#  Closed-form 2×2 matrix exponential (eigenvalue method)
# ════════════════════════════════════════════════════════════════════════

def matrix_exp_2x2(M, T):
    """
    Compute exp(M·T) for a 2×2 matrix using the eigenvalue closed form.

    η± = (tr ± √(tr² - 4·det)) / 2

    exp(MT) = [(η₊·e^{η₋T} - η₋·e^{η₊T})·I + (e^{η₊T} - e^{η₋T})·M] / (η₊ - η₋)

    Falls back to scipy expm if eigenvalues are degenerate.
    """
    m11, m12 = M[0, 0], M[0, 1]
    m21, m22 = M[1, 0], M[1, 1]

    tr = m11 + m22
    det = m11 * m22 - m12 * m21
    disc = tr**2 - 4 * det

    sqrt_disc = np.sqrt(disc + 0j)  # ensure complex

    eta_p = (tr + sqrt_disc) / 2
    eta_m = (tr - sqrt_disc) / 2

    diff = eta_p - eta_m

    if abs(diff) < 1e-14:
        # Degenerate eigenvalues — fall back to scipy
        return expm(M * T)

    exp_p = np.exp(eta_p * T)
    exp_m = np.exp(eta_m * T)

    I = np.eye(2, dtype=complex)
    result = ((eta_p * exp_m - eta_m * exp_p) * I + (exp_p - exp_m) * M) / diff

    return result


# ════════════════════════════════════════════════════════════════════════
#  Convenience: CF using closed-form matrix exp (no scipy.expm needed)
# ════════════════════════════════════════════════════════════════════════

def characteristic_function_closed(
    u, S0, r, T, sigma_low, sigma_high, q12, q21,
    current_regime=0, dt=0.02,
):
    """Same as continuous-time CF but uses the analytical 2×2 matrix exponential."""
    x = np.log(S0)
    sigmas = np.array([sigma_low, sigma_high])

    lam12 = -np.log(1 - q12) / dt
    lam21 = -np.log(1 - q21) / dt

    u = np.atleast_1d(np.asarray(u, dtype=complex))
    result = np.empty(len(u), dtype=complex)

    for i, ui in enumerate(u):
        alpha = -(1j * ui + ui**2) / 2.0
        a0 = alpha * sigmas[0]**2
        a1 = alpha * sigmas[1]**2

        M = np.array([
            [-lam12 + a0, lam12],
            [lam21,       -lam21 + a1]
        ], dtype=complex)

        eM = matrix_exp_2x2(M, T)
        f = eM @ np.ones(2)
        result[i] = np.exp(1j * ui * x + 1j * ui * r * T) * f[current_regime]

    return result if len(result) > 1 else result[0]


# ════════════════════════════════════════════════════════════════════════
#  Verification: compare CF against Monte Carlo
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # ── Parameters (match simulate_gbm.py defaults) ─────────────────
    S0 = 36.0
    r = 0.06
    T = 1.0
    dt = 0.02
    sigma_low = 0.15
    sigma_high = 0.40
    q12 = 0.05
    q21 = 0.10
    regime = 0
    n_paths = 500_000

    print("=" * 70)
    print("  Characteristic Function Verification")
    print("=" * 70)
    print(f"  S0={S0}, r={r}, T={T}, σ_low={sigma_low}, σ_high={sigma_high}")
    print(f"  q12={q12}, q21={q21}, dt={dt}, starting regime={regime}")

    # ── 1. Monte Carlo: empirical CF ────────────────────────────────
    print(f"\n  Running Monte Carlo ({n_paths:,} paths)...")
    S, reg, params = simulate_regime_switching_gbm(
        S0=S0, r=r, T=T, dt=dt, n_paths=n_paths,
        sigma_low=sigma_low, sigma_high=sigma_high,
        q12=q12, q21=q21, measure="Q",
        seed=42, initial_regime=regime,
    )
    log_ST = np.log(S[-1, :])

    # ── 2. Compare at several u values ──────────────────────────────
    test_us = [0.5, 1.0, 2.0, 5.0, -1.0, 1.0 + 0.5j]

    print(f"\n  {'u':>10s}  |  {'φ_continuous':>28s}  |  {'φ_discrete':>28s}  |  {'φ_MC':>28s}  |  {'|Δ cont-MC|':>12s}  |  {'|Δ disc-MC|':>12s}")
    print("  " + "-" * 140)

    for u_val in test_us:
        # Analytical CFs
        phi_cont = characteristic_function(
            u_val, S0, r, T, sigma_low, sigma_high, q12, q21,
            current_regime=regime, dt=dt, method="continuous"
        )
        phi_disc = characteristic_function(
            u_val, S0, r, T, sigma_low, sigma_high, q12, q21,
            current_regime=regime, dt=dt, method="discrete"
        )

        # Monte Carlo CF: E[exp(iu·ln(S_T))]
        phi_mc = np.mean(np.exp(1j * u_val * log_ST))

        err_cont = abs(phi_cont - phi_mc)
        err_disc = abs(phi_disc - phi_mc)

        def fmt_complex(z):
            return f"{z.real:+.6f}{z.imag:+.6f}i"

        print(f"  {str(u_val):>10s}  |  {fmt_complex(phi_cont):>28s}  |  {fmt_complex(phi_disc):>28s}  |  {fmt_complex(phi_mc):>28s}  |  {err_cont:>12.6f}  |  {err_disc:>12.6f}")

    # ── 3. Moment checks via CF derivatives ─────────────────────────
    print("\n" + "=" * 70)
    print("  Moment Checks (CF derivatives vs. Monte Carlo)")
    print("=" * 70)

    # E[ln(S_T)] = -i · φ'(0) / φ(0)  ≈  -i · [φ(ε) - φ(-ε)] / (2ε·φ(0))
    eps = 1e-5
    phi_p = characteristic_function(eps, S0, r, T, sigma_low, sigma_high,
                                    q12, q21, current_regime=regime, dt=dt, method="discrete")
    phi_m = characteristic_function(-eps, S0, r, T, sigma_low, sigma_high,
                                    q12, q21, current_regime=regime, dt=dt, method="discrete")
    phi_0 = characteristic_function(0.0, S0, r, T, sigma_low, sigma_high,
                                    q12, q21, current_regime=regime, dt=dt, method="discrete")

    mean_cf = (-1j * (phi_p - phi_m) / (2 * eps * phi_0)).real
    mean_mc = np.mean(log_ST)
    print(f"\n  E[ln(S_T)]:")
    print(f"    CF (discrete)  = {mean_cf:.6f}")
    print(f"    Monte Carlo    = {mean_mc:.6f}")
    print(f"    Difference     = {abs(mean_cf - mean_mc):.6f}")

    # E[ln(S_T)²] - E[ln(S_T)]²  = Var[ln(S_T)]
    phi_pp = characteristic_function(eps, S0, r, T, sigma_low, sigma_high,
                                     q12, q21, current_regime=regime, dt=dt, method="discrete")
    deriv2 = (phi_p - 2*phi_0 + phi_m) / (eps**2 * phi_0)
    var_cf = (-deriv2 - (mean_cf * 1j)**2 ).real  # Var = -φ''/φ - (φ'/φ)²
    # Simpler: Var = -(φ''/φ(0)) - mean²
    var_cf2 = (-deriv2).real - mean_cf**2
    var_mc = np.var(log_ST)
    print(f"\n  Var[ln(S_T)]:")
    print(f"    CF (discrete)  = {var_cf2:.6f}")
    print(f"    Monte Carlo    = {var_mc:.6f}")
    print(f"    Difference     = {abs(var_cf2 - var_mc):.6f}")

    # ── 4. Sanity check: Black-Scholes limit ────────────────────────
    print("\n" + "=" * 70)
    print("  Sanity Check: σ_low = σ_high = 0.20  (should match Black-Scholes)")
    print("=" * 70)

    sigma_bs = 0.20
    u_test = 1.0

    # Regime-switching CF with equal vols
    phi_rs = characteristic_function(
        u_test, S0, r, T, sigma_bs, sigma_bs, q12, q21,
        current_regime=0, dt=dt, method="continuous"
    )

    # Black-Scholes CF:  exp(iu(x + rT) - σ²(iu + u²)T/2)
    x = np.log(S0)
    phi_bs = np.exp(1j * u_test * (x + r * T)
                    - sigma_bs**2 * (1j * u_test + u_test**2) * T / 2)

    print(f"\n  φ_RS(u={u_test})  = {phi_rs.real:+.10f}{phi_rs.imag:+.10f}i")
    print(f"  φ_BS(u={u_test})  = {phi_bs.real:+.10f}{phi_bs.imag:+.10f}i")
    print(f"  |Difference|    = {abs(phi_rs - phi_bs):.2e}")
    print(f"  {'✅ PASS' if abs(phi_rs - phi_bs) < 1e-8 else '❌ FAIL'}")

    print("\n" + "=" * 70)
