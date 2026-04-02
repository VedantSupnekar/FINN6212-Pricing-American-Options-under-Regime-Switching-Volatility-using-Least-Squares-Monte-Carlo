import numpy as np
import matplotlib.pyplot as plt

# ── GBM Parameters ──────────────────────────────────────────────────
S0 = 36.0       # Initial stock price
r = 0.06        # Risk-free rate
T = 1.0         # Time to maturity (1 year)
dt = 0.02       # Time step (50 intervals per year)
N = int(T / dt) # Number of time steps
paths = 100    # Number of simulated paths

# ── Regime-Switching Volatility Parameters ──────────────────────────
sigma_low = 0.15    # σ₁  – "calm" regime volatility
sigma_high = 0.40   # σ₂  – "turbulent" regime volatility
sigmas = np.array([sigma_low, sigma_high])  # indexed 0 = Low, 1 = High

# Transition probability matrix (per time-step dt)
# P[i, j] = probability of moving from regime i to regime j
#   p12 = P(Low  → High)
#   p21 = P(High → Low)
p12 = 0.05   # probability of switching from Low Vol to High Vol
p21 = 0.10   # probability of switching from High Vol to Low Vol

transition_matrix = np.array([
    [1 - p12, p12],     # from Low:  stay Low  |  go High
    [p21,     1 - p21]  # from High: go Low    |  stay High
])

# ── Random seed for reproducibility ─────────────────────────────────
np.random.seed(42)

# ── Simulate regime paths via Markov chain ──────────────────────────
# regime[t, path] ∈ {0, 1}  (0 = Low Vol, 1 = High Vol)
regime = np.zeros((N + 1, paths), dtype=int)
regime[0] = 0  # start all paths in the Low-Vol regime

# At each step, draw a uniform random number; if it exceeds the
# "stay" probability, switch regimes.
for t in range(1, N + 1):
    u = np.random.uniform(size=paths)
    for p in range(paths):
        current = regime[t - 1, p]
        # cumulative probability of staying in current regime
        if u[p] < transition_matrix[current, 0]:
            regime[t, p] = 0
        else:
            regime[t, p] = 1

# Map regimes to volatilities: sigma_path[t, path] = σ for that step
sigma_path = sigmas[regime]

# ── Simulate stock-price paths (GBM with regime-switching σ) ───────
S = np.zeros((N + 1, paths))
S[0] = S0

for t in range(1, N + 1):
    Z = np.random.standard_normal(paths)
    sig = sigma_path[t, :]  # volatility determined by current regime
    S[t] = S[t - 1] * np.exp((r - 0.5 * sig**2) * dt + sig * np.sqrt(dt) * Z)

# ── Output ──────────────────────────────────────────────────────────
print("Final prices of the first 5 paths:", S[-1, :5])
print("\nRegime at final step for first 5 paths:",
      ["Low" if r == 0 else "High" for r in regime[-1, :5]])

# ── Plot: Stock Price Paths ─────────────────────────────────────────
highlight_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
n_highlight = min(5, paths)

fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

# Top panel – all paths in light grey, paths 1-5 highlighted
ax1 = axes[0]
for p_idx in range(paths):
    ax1.plot(S[:, p_idx], color="lightgrey", alpha=0.5, linewidth=0.5)
for p_idx in range(n_highlight):
    ax1.plot(S[:, p_idx], color=highlight_colors[p_idx],
             alpha=0.9, linewidth=1.4, label=f"Path {p_idx + 1}")
ax1.set_title("Regime-Switching GBM Paths  (paths 1-5 highlighted)")
ax1.set_ylabel("Asset Price")
ax1.legend(loc="upper left", fontsize=8)
ax1.grid(True, alpha=0.3)

# Bottom panel – regime indicator for paths 1-5 (same colours)
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
