import numpy as np
import matplotlib.pyplot as plt

# Parameters for the simulation (similar to the paper's examples)
S0 = 36.0      # Initial stock price
r = 0.06       # Risk-free rate
sigma = 0.2    # Volatility
T = 1.0        # Time to maturity (1 year)
dt = 0.02      # Time step (e.g., 50 intervals per year)
N = int(T / dt) # Number of time steps
paths = 100     # Number of simulated paths


np.random.seed(42)

# Initialize array for stock prices
# Rows = time steps, Columns = different simulated paths
S = np.zeros((N + 1, paths))
S[0] = S0

# Generate paths using Geometric Brownian Motion formula
for t in range(1, N + 1):
    # standard normal random numbers
    Z = np.random.standard_normal(paths)
    
    # GBM formula
    S[t] = S[t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

# Print the last prices to check them immediately
print("Final prices of the first 5 paths:", S[-1, :5])

# Plot the simulated paths to see if they look correct
plt.figure(figsize=(10, 6))
plt.plot(S)
plt.title('Geometric Brownian Motion (GBM) Paths')
plt.xlabel('Time Steps')
plt.ylabel('Asset Price')
plt.grid(True)
plt.show()
