"""
Verification script for the regime-switching GBM simulation.

Three sanity checks:
  1. Visual: plot sample paths with regime-shaded backgrounds
  2. Regime occupancy vs. theoretical stationary distribution
  3. Annualized realized volatility per regime vs. target σ₁, σ₂
"""

import numpy as np
import matplotlib.pyplot as plt
from simulate_gbm import simulate_regime_switching_gbm

# ═══════════════════════════════════════════════════════════════════════
#  Run simulation (10k paths, start from stationary distribution)
# ═══════════════════════════════════════════════════════════════════════
S, regime, params = simulate_regime_switching_gbm(
    n_paths=10_000,
    initial_regime="stationary",
)

N = params["N"]
n_paths = params["n_paths"]
sigma_low = params["sigma_low"]
sigma_high = params["sigma_high"]
p12 = params["p12"]
p21 = params["p21"]


# ═══════════════════════════════════════════════════════════════════════
#  CHECK 1: Visual – sample paths with regime-shaded backgrounds
# ═══════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  CHECK 1: Visual inspection of sample paths with regime shading")
print("=" * 65)

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

for ax_idx in range(3):
    ax = axes[ax_idx]
    p_idx = ax_idx  # paths 0, 1, 2

    ax.plot(S[:, p_idx], color="black", linewidth=1.2)

    # Shade background by regime: green = Low Vol, red = High Vol
    for t in range(N):
        color = "#d4edda" if regime[t, p_idx] == 0 else "#f8d7da"
        ax.axvspan(t, t + 1, alpha=0.5, color=color, linewidth=0)

    ax.set_ylabel("Price")
    ax.set_title(f"Path {p_idx + 1}  (green = Low-Vol regime, red = High-Vol regime)",
                 fontsize=10)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Time Steps")
fig.suptitle("CHECK 1: Price Paths with Regime-Shaded Backgrounds", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("check1_regime_shading.png", dpi=150, bbox_inches="tight")
plt.show()


# ═══════════════════════════════════════════════════════════════════════
#  CHECK 2: Regime occupancy vs. theoretical stationary distribution
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  CHECK 2: Regime occupancy frequencies")
print("=" * 65)

# Theoretical stationary distribution:  π = [p21, p12] / (p12 + p21)
pi_theoretical = np.array([p21, p12]) / (p12 + p21)

# Empirical: fraction of all (time-step, path) pairs in each regime
regime_flat = regime[1:, :].flatten()
empirical_low = np.mean(regime_flat == 0)
empirical_high = np.mean(regime_flat == 1)
pi_empirical = np.array([empirical_low, empirical_high])

print(f"\n  Theoretical stationary distribution:")
print(f"    π(Low-Vol)  = p21 / (p12 + p21) = {pi_theoretical[0]:.4f}")
print(f"    π(High-Vol) = p12 / (p12 + p21) = {pi_theoretical[1]:.4f}")
print(f"\n  Empirical occupancy ({n_paths:,} paths × {N} steps = {n_paths * N:,} observations):")
print(f"    Fraction in Low-Vol  = {pi_empirical[0]:.4f}")
print(f"    Fraction in High-Vol = {pi_empirical[1]:.4f}")
print(f"\n  Absolute errors:")
print(f"    |Δ Low-Vol|  = {abs(pi_empirical[0] - pi_theoretical[0]):.4f}")
print(f"    |Δ High-Vol| = {abs(pi_empirical[1] - pi_theoretical[1]):.4f}")

match_ok = all(abs(pi_empirical - pi_theoretical) < 0.03)
print(f"\n  ✅ PASS" if match_ok else f"\n  ❌ FAIL (deviation > 0.03)")


# ═══════════════════════════════════════════════════════════════════════
#  CHECK 3: Annualized realized vol per regime vs. target σ₁, σ₂
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  CHECK 3: Realized volatility by regime")
print("=" * 65)

dt = params["dt"]
log_returns = np.log(S[1:] / S[:-1])          # shape (N, n_paths)
regime_at_step = regime[1:, :]                  # aligned with returns

returns_low = log_returns[regime_at_step == 0]
returns_high = log_returns[regime_at_step == 1]

realized_vol_low = np.std(returns_low) / np.sqrt(dt)
realized_vol_high = np.std(returns_high) / np.sqrt(dt)

print(f"\n  Target volatilities:")
print(f"    σ_low  = {sigma_low:.4f}")
print(f"    σ_high = {sigma_high:.4f}")
print(f"\n  Realized annualized volatilities:")
print(f"    σ_low  (realized) = {realized_vol_low:.4f}")
print(f"    σ_high (realized) = {realized_vol_high:.4f}")
print(f"\n  Relative errors:")
print(f"    |Δσ_low|  / σ_low  = {abs(realized_vol_low - sigma_low) / sigma_low:.2%}")
print(f"    |Δσ_high| / σ_high = {abs(realized_vol_high - sigma_high) / sigma_high:.2%}")

vol_ok = (abs(realized_vol_low - sigma_low) / sigma_low < 0.05 and
          abs(realized_vol_high - sigma_high) / sigma_high < 0.05)
print(f"\n  ✅ PASS" if vol_ok else f"\n  ❌ FAIL (relative error > 5%)")


# ═══════════════════════════════════════════════════════════════════════
#  Summary bar chart
# ═══════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

labels = ["Low-Vol (π₀)", "High-Vol (π₁)"]
x = np.arange(len(labels))
width = 0.3
ax1.bar(x - width / 2, pi_theoretical, width, label="Theoretical", color="steelblue")
ax1.bar(x + width / 2, pi_empirical, width, label="Empirical", color="coral")
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_ylabel("Fraction of time")
ax1.set_title("CHECK 2: Regime Occupancy")
ax1.legend()
ax1.set_ylim(0, 1)
ax1.grid(True, alpha=0.3, axis="y")

vol_labels = ["σ_low", "σ_high"]
target_vols = [sigma_low, sigma_high]
realized_vols = [realized_vol_low, realized_vol_high]
x2 = np.arange(len(vol_labels))
ax2.bar(x2 - width / 2, target_vols, width, label="Target", color="steelblue")
ax2.bar(x2 + width / 2, realized_vols, width, label="Realized", color="coral")
ax2.set_xticks(x2)
ax2.set_xticklabels(vol_labels)
ax2.set_ylabel("Annualized Volatility")
ax2.set_title("CHECK 3: Realized vs. Target Volatility")
ax2.legend()
ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("check2_3_summary.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n" + "=" * 65)
print("  All verification checks complete.")
print("=" * 65)
