import numpy as np
import matplotlib.pyplot as plt

# ——————————

# Inputs from the user

# ——————————

r1 = float(input(“Asset 1 Expected Return (%) [e.g., 10]: “)) / 100
sd1 = float(input(“Asset 1 Standard Deviation (%) [e.g., 15]: “)) / 100
esg1 = float(input(“Asset 1 ESG Score (0 to 1) [e.g., 0.8]: “))

r2 = float(input(“Asset 2 Expected Return (%) [e.g., 7]: “)) / 100
sd2 = float(input(“Asset 2 Standard Deviation (%) [e.g., 22]: “)) / 100
esg2 = float(input(“Asset 2 ESG Score (0 to 1) [e.g., 0.3]: “))

rho = float(input(“Correlation between Asset 1 and 2 [-1 to 1, e.g., 0.3]: “))
r_free = float(input(“Risk-Free Rate (%) [e.g., 2.5]: “)) / 100
gamma = float(input(“Risk Aversion (γ) [e.g., 3]: “))
lambda_ = float(input(“ESG Preference (λ) [0 = no preference, e.g., 1.5]: “))

# ——————————

# Functions

# ——————————

def portfolio_ret(w1, r1, r2):
return w1 * r1 + (1 - w1) * r2

def portfolio_sd(w1, sd1, sd2, rho):
return np.sqrt(w1**2 * sd1**2 + (1-w1)**2 * sd2**2 + 2 * w1 * (1-w1) * sd1 * sd2 * rho)

def portfolio_esg(w1, esg1, esg2):
return w1 * esg1 + (1 - w1) * esg2

# ——————————

# Sweep weights and compute metrics

# ——————————

weights = np.linspace(0, 1, 1000)
sharpe_ratios = []
utilities = []
esg_scores = []

for w in weights:
ret = portfolio_ret(w, r1, r2)
sd  = portfolio_sd(w, sd1, sd2, rho)
esg = portfolio_esg(w, esg1, esg2)
esg_scores.append(esg)

```
if sd > 0:
    sharpe = (ret - r_free) / sd
else:
    sharpe = -np.inf
sharpe_ratios.append(sharpe)

# ESG utility: U = E[Rp] - (γ/2)·σ²p + λ·s̄
utility = ret - (gamma / 2) * sd**2 + lambda_ * esg
utilities.append(utility)
```

# ——————————

# Max-Sharpe Portfolio (tangency)

# ——————————

max_sharpe_idx = np.argmax(sharpe_ratios)
w1_tangency = weights[max_sharpe_idx]
w2_tangency = 1 - w1_tangency
ret_tangency = portfolio_ret(w1_tangency, r1, r2)
sd_tangency  = portfolio_sd(w1_tangency, sd1, sd2, rho)

# ——————————

# Optimal Portfolio (max utility)

# ——————————

opt_idx = np.argmax(utilities)
w1_opt = weights[opt_idx]
w2_opt = 1 - w1_opt
ret_opt = portfolio_ret(w1_opt, r1, r2)
sd_opt  = portfolio_sd(w1_opt, sd1, sd2, rho)
esg_opt = portfolio_esg(w1_opt, esg1, esg2)

# ——————————

# Display results

# ——————————

print(”\n— Max-Sharpe (Tangency) Portfolio —”)
print(f”  Asset 1 weight: {w1_tangency*100:.2f}%”)
print(f”  Asset 2 weight: {w2_tangency*100:.2f}%”)
print(f”  Expected Return: {ret_tangency*100:.2f}%”)
print(f”  Std Dev: {sd_tangency*100:.2f}%”)
print(f”  Sharpe Ratio: {sharpe_ratios[max_sharpe_idx]:.3f}”)
print(f”  ESG Score: {esg_scores[max_sharpe_idx]:.3f}”)

print(”\n— Optimal Portfolio (Max Utility with ESG) —”)
print(f”  Asset 1 weight: {w1_opt*100:.2f}%”)
print(f”  Asset 2 weight: {w2_opt*100:.2f}%”)
print(f”  Expected Return: {ret_opt*100:.2f}%”)
print(f”  Std Dev: {sd_opt*100:.2f}%”)
print(f”  Sharpe Ratio: {sharpe_ratios[opt_idx]:.3f}”)
print(f”  ESG Score: {esg_opt:.3f}”)

esg_cost = sharpe_ratios[max_sharpe_idx] - sharpe_ratios[opt_idx]
print(f”\n  ESG cost (drop in Sharpe): {esg_cost:.3f}”)

# ——————————

# Plot

# ——————————

weights_plot = np.linspace(0, 1, 200)
returns_frontier = [portfolio_ret(w, r1, r2) for w in weights_plot]
sds_frontier     = [portfolio_sd(w, sd1, sd2, rho) for w in weights_plot]

fig, ax = plt.subplots(figsize=(8, 5))

# Efficient frontier

ax.plot(sds_frontier, returns_frontier, ‘b-’, linewidth=2, label=‘Efficient Frontier’)

# Capital Market Line

if sd_tangency > 0:
sd_max = max(sds_frontier) * 1.2
sd_cml = np.linspace(0, sd_max, 100)
ret_cml = r_free + (ret_tangency - r_free) / sd_tangency * sd_cml
ax.plot(sd_cml, ret_cml, ‘g–’, linewidth=2, label=‘Capital Market Line’)

# Key portfolios

ax.scatter(sd_tangency, ret_tangency, color=‘red’,    s=100, marker=’*’, zorder=5, label=‘Max-Sharpe Portfolio’)
ax.scatter(sd_opt,      ret_opt,      color=‘orange’, s=100, marker=’*’, zorder=5, label=f’Optimal Portfolio (ESG={esg_opt:.2f})’)
ax.scatter(0,           r_free,       color=‘green’,  s=80,  marker=‘s’, zorder=5, label=‘Risk-Free Asset’)

ax.set_xlabel(‘Risk (Standard Deviation)’)
ax.set_ylabel(‘Expected Return’)
ax.set_title(‘ESG Portfolio Optimisation’)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()