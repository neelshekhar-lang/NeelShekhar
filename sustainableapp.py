import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ——————————

# Inputs

# ——————————

print( "==== Sustainable Finance Portfolio App ====\n”)

r1    = float(input(“Asset 1 Expected Return (%) [e.g., 10]: “)) / 100
sd1   = float(input(“Asset 1 Standard Deviation (%) [e.g., 15]: “)) / 100
esg1  = float(input(“Asset 1 ESG Score (0 to 1) [e.g., 0.8]: “))

r2    = float(input(”\nAsset 2 Expected Return (%) [e.g., 7]: “)) / 100
sd2   = float(input(“Asset 2 Standard Deviation (%) [e.g., 22]: “)) / 100
esg2  = float(input(“Asset 2 ESG Score (0 to 1) [e.g., 0.3]: “))

rho    = float(input(”\nCorrelation between Asset 1 and 2 [-1 to 1, e.g., 0.3]: “))
r_free = float(input(“Risk-Free Rate (%) [e.g., 2.5]: “)) / 100
gamma  = float(input(”\nRisk Aversion (γ) [e.g., 3]: “))
lam    = float(input(“ESG Preference (λ) [e.g., 1.5]: “))

# ——————————

# Functions

# ——————————

def portfolio_ret(w1, r1, r2):
return w1 * r1 + (1 - w1) * r2

def portfolio_sd(w1, sd1, sd2, rho):
return np.sqrt(
w1**2 * sd1**2 +
(1 - w1)**2 * sd2**2 +
2 * rho * w1 * (1 - w1) * sd1 * sd2
)

def portfolio_esg(w1, esg1, esg2):
return w1 * esg1 + (1 - w1) * esg2

def utility(w1):
# Lecture 6 formula: U = E[Rp] - (γ/2)·σ²p + λ·s̄
ret = portfolio_ret(w1, r1, r2)
sd  = portfolio_sd(w1, sd1, sd2, rho)
esg = portfolio_esg(w1, esg1, esg2)
return ret - (gamma / 2) * sd**2 + lam * esg

# ——————————

# Build portfolios DataFrame

# ——————————

weights = np.linspace(0, 1, 1000)

rows = []
for w in weights:
ret = portfolio_ret(w, r1, r2)
sd  = portfolio_sd(w, sd1, sd2, rho)
esg = portfolio_esg(w, esg1, esg2)
u   = utility(w)
sharpe = (ret - r_free) / sd if sd > 0 else -np.inf
rows.append({
‘Weight Asset 1’: w,
‘Weight Asset 2’: 1 - w,
‘Return’:  ret,
‘Std Dev’: sd,
‘ESG Score’: esg,
‘Utility’: u,
‘Sharpe’:  sharpe
})

portfolios = pd.DataFrame(rows)

# ——————————

# Find optimal portfolios

# ——————————

# ESG-adjusted optimal (max utility) — Lecture 6

idx_opt = portfolios[‘Utility’].idxmax()
w1_opt  = portfolios.loc[idx_opt, ‘Weight Asset 1’]
ret_opt = portfolios.loc[idx_opt, ‘Return’]
sd_opt  = portfolios.loc[idx_opt, ‘Std Dev’]
esg_opt = portfolios.loc[idx_opt, ‘ESG Score’]
u_opt   = portfolios.loc[idx_opt, ‘Utility’]

# Tangency portfolio (max Sharpe, λ = 0 benchmark)

idx_tan = portfolios[‘Sharpe’].idxmax()
w1_tan  = portfolios.loc[idx_tan, ‘Weight Asset 1’]
ret_tan = portfolios.loc[idx_tan, ‘Return’]
sd_tan  = portfolios.loc[idx_tan, ‘Std Dev’]
esg_tan = portfolios.loc[idx_tan, ‘ESG Score’]
sharpe_max = portfolios.loc[idx_tan, ‘Sharpe’]
sharpe_opt = portfolios.loc[idx_opt, ‘Sharpe’]

# MV optimal (λ = 0, no ESG)

mv_utilities = portfolios[‘Return’] - (gamma / 2) * portfolios[‘Std Dev’]**2
idx_mv  = mv_utilities.idxmax()
w1_mv   = portfolios.loc[idx_mv, ‘Weight Asset 1’]
ret_mv  = portfolios.loc[idx_mv, ‘Return’]
sd_mv   = portfolios.loc[idx_mv, ‘Std Dev’]

# ESG cost

esg_cost = sharpe_max - sharpe_opt

# ——————————

# Output tables

# ——————————

print(”\n==== Portfolio Comparison ====\n”)

comparison = pd.DataFrame({
‘Portfolio’:          [‘MV Optimal (λ=0)’, ‘ESG Optimal’, ‘Tangency (Max Sharpe)’],
‘Weight Asset 1 (%)’: [round(w1_mv*100, 2),  round(w1_opt*100, 2),  round(w1_tan*100, 2)],
‘Weight Asset 2 (%)’: [round((1-w1_mv)*100,2),round((1-w1_opt)*100,2),round((1-w1_tan)*100,2)],
})
print(comparison.to_string(index=False))

print(”\n==== ESG Optimal Portfolio ====\n”)
details = pd.DataFrame({
‘Characteristic’: [‘Expected Return’, ‘Std Dev (Risk)’, ‘ESG Score’, ‘Utility’, ‘Sharpe Ratio’],
‘Value’: [
f”{ret_opt*100:.2f}%”,
f”{sd_opt*100:.2f}%”,
f”{esg_opt:.3f}”,
f”{u_opt:.4f}”,
f”{sharpe_opt:.3f}”,
]
})
print(details.to_string(index=False))
print(f”\nESG Cost (drop in Sharpe vs Tangency): {esg_cost:.3f}”)

# ——————————

# Plot

# ——————————

fig, ax = plt.subplots(figsize=(10, 6))

# Efficient frontier

ax.plot(portfolios[‘Std Dev’], portfolios[‘Return’],
‘b-’, linewidth=2, label=‘Efficient Frontier’)

# Capital Market Line

if sd_tan > 0:
sd_range = np.linspace(0, portfolios[‘Std Dev’].max() * 1.2, 200)
ret_cml  = r_free + (ret_tan - r_free) / sd_tan * sd_range
ax.plot(sd_range, ret_cml, ‘g–’, linewidth=1.5, label=‘Capital Market Line’)

# Key portfolios

ax.scatter(sd_mv,  ret_mv,  s=120, color=‘steelblue’, marker=‘D’, zorder=5, label=‘MV Optimal’)
ax.scatter(sd_tan, ret_tan, s=180, color=‘red’,       marker=’*’, zorder=5, label=‘Tangency Portfolio’)
ax.scatter(sd_opt, ret_opt, s=180, color=‘orange’,    marker=’*’, zorder=5, label=f’ESG Optimal (ESG={esg_opt:.2f})’)
ax.scatter(0,      r_free,  s=100, color=‘green’,     marker=‘s’, zorder=5, label=‘Risk-Free Asset’)

# MV indifference curve

U_mv_star    = ret_mv - (gamma / 2) * sd_mv**2
sigma_curve  = np.linspace(0, portfolios[‘Std Dev’].max() * 1.2, 200)
mu_mv_curve  = U_mv_star + (gamma / 2) * sigma_curve**2
ax.plot(sigma_curve, mu_mv_curve, ‘:’, linewidth=1.5,
color=‘steelblue’, label=‘MV Indifference Curve’)

# ESG indifference curve

# Derived from: U_opt = E[R] - (γ/2)σ² + λ·esg_opt  →  E[R] = U_opt - λ·esg_opt + (γ/2)σ²

mu_esg_curve = (u_opt - lam * esg_opt) + (gamma / 2) * sigma_curve**2
ax.plot(sigma_curve, mu_esg_curve, ‘-.’, linewidth=1.5,
color=‘orange’, label=‘ESG Indifference Curve’)

ax.set_xlabel(‘Risk (Standard Deviation)’)
ax.set_ylabel(‘Expected Return’)
ax.set_title(‘Sustainable Portfolio Optimisation\n(MV vs ESG Optimal with CML)’)
ax.legend(fontsize=8.5)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()