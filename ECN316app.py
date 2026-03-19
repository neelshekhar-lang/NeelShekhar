import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.title("Sustainable Finance Portfolio App")
st.markdown("**Utility function:** U = E[Rp] - (γ/2)·σ²p + λ·s̄")
st.divider()

# ------------------------------
# Inputs
# ------------------------------
st.subheader("Asset 1")
col1, col2, col3 = st.columns(3)
r1   = col1.number_input("Expected Return (%)", value=10.0, step=0.5, key="r1") / 100
sd1  = col2.number_input("Standard Deviation (%)", value=15.0, step=0.5, key="sd1") / 100
esg1 = col3.number_input("ESG Score (0 to 1)", value=0.8, step=0.01, min_value=0.0, max_value=1.0, key="esg1")

st.subheader("Asset 2")
col4, col5, col6 = st.columns(3)
r2   = col4.number_input("Expected Return (%)", value=7.0, step=0.5, key="r2") / 100
sd2  = col5.number_input("Standard Deviation (%)", value=22.0, step=0.5, key="sd2") / 100
esg2 = col6.number_input("ESG Score (0 to 1)", value=0.3, step=0.01, min_value=0.0, max_value=1.0, key="esg2")

st.subheader("Market & Preferences")
col7, col8, col9, col10 = st.columns(4)
rho    = col7.number_input("Correlation rho", value=0.3, step=0.05, min_value=-1.0, max_value=1.0)
r_free = col8.number_input("Risk-Free Rate (%)", value=2.5, step=0.25) / 100
gamma  = col9.number_input("Risk Aversion gamma", value=3.0, step=0.5, min_value=0.1)
lam    = col10.number_input("ESG Preference lambda", value=1.5, step=0.25, min_value=0.0)

st.divider()

# ------------------------------
# Functions
# ------------------------------
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
    ret = portfolio_ret(w1, r1, r2)
    sd  = portfolio_sd(w1, sd1, sd2, rho)
    esg = portfolio_esg(w1, esg1, esg2)
    return ret - (gamma / 2) * sd**2 + lam * esg

# ------------------------------
# Build portfolios DataFrame
# ------------------------------
weights = np.linspace(0, 1, 1000)

rows = []
for w in weights:
    ret = portfolio_ret(w, r1, r2)
    sd  = portfolio_sd(w, sd1, sd2, rho)
    esg = portfolio_esg(w, esg1, esg2)
    u   = utility(w)
    sharpe = (ret - r_free) / sd if sd > 0 else -np.inf
    rows.append({
        'Weight Asset 1': w,
        'Weight Asset 2': 1 - w,
        'Return':  ret,
        'Std Dev': sd,
        'ESG Score': esg,
        'Utility': u,
        'Sharpe':  sharpe
    })

portfolios = pd.DataFrame(rows)

# ------------------------------
# Find optimal portfolios
# ------------------------------

idx_opt = portfolios['Utility'].idxmax()
w1_opt  = portfolios.loc[idx_opt, 'Weight Asset 1']
ret_opt = portfolios.loc[idx_opt, 'Return']
sd_opt  = portfolios.loc[idx_opt, 'Std Dev']
esg_opt = portfolios.loc[idx_opt, 'ESG Score']
u_opt   = portfolios.loc[idx_opt, 'Utility']
sharpe_opt = portfolios.loc[idx_opt, 'Sharpe']

idx_tan = portfolios['Sharpe'].idxmax()
w1_tan  = portfolios.loc[idx_tan, 'Weight Asset 1']
ret_tan = portfolios.loc[idx_tan, 'Return']
sd_tan  = portfolios.loc[idx_tan, 'Std Dev']
sharpe_max = portfolios.loc[idx_tan, 'Sharpe']

mv_utilities = portfolios['Return'] - (gamma / 2) * portfolios['Std Dev']**2
idx_mv  = mv_utilities.idxmax()
w1_mv   = portfolios.loc[idx_mv, 'Weight Asset 1']
ret_mv  = portfolios.loc[idx_mv, 'Return']
sd_mv   = portfolios.loc[idx_mv, 'Std Dev']

esg_cost = sharpe_max - sharpe_opt

# ------------------------------
# Output tables
# ------------------------------
st.subheader("Portfolio Comparison")

comparison = pd.DataFrame({
    'Portfolio':           ['MV Optimal (lambda=0)', 'ESG Optimal', 'Tangency (Max Sharpe)'],
    'Weight Asset 1 (%)':  [round(w1_mv*100, 2),   round(w1_opt*100, 2),   round(w1_tan*100, 2)],
    'Weight Asset 2 (%)':  [round((1-w1_mv)*100,2), round((1-w1_opt)*100,2),round((1-w1_tan)*100,2)],
})
st.dataframe(comparison, hide_index=True, use_container_width=True)

st.subheader("ESG Optimal Portfolio")

col_a, col_b, col_c, col_d, col_e = st.columns(5)
col_a.metric("Expected Return",   f"{ret_opt*100:.2f}%")
col_b.metric("Std Dev (Risk)",    f"{sd_opt*100:.2f}%")
col_c.metric("ESG Score",         f"{esg_opt:.3f}")
col_d.metric("Sharpe Ratio",      f"{sharpe_opt:.3f}")
col_e.metric("ESG Cost (Sharpe)", f"{esg_cost:.3f}")

if lam > 0 and esg_cost > 0.001:
    st.info(f"Your ESG preference (lambda = {lam}) shifts the portfolio to a greener choice, "
            f"at a cost of {esg_cost:.3f} in Sharpe ratio vs the pure max-Sharpe portfolio.")

st.divider()

# ------------------------------
# Plot
# ------------------------------
st.subheader("Chart")

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(portfolios['Std Dev'], portfolios['Return'],
        'b-', linewidth=2, label='Efficient Frontier')

if sd_tan > 0:
    sd_range = np.linspace(0, portfolios['Std Dev'].max() * 1.2, 200)
    ret_cml  = r_free + (ret_tan - r_free) / sd_tan * sd_range
    ax.plot(sd_range, ret_cml, 'g--', linewidth=1.5, label='Capital Market Line')

ax.scatter(sd_mv,  ret_mv,  s=120, color='steelblue', marker='D', zorder=5, label='MV Optimal')
ax.scatter(sd_tan, ret_tan, s=180, color='red',       marker='*', zorder=5, label='Tangency Portfolio')
ax.scatter(sd_opt, ret_opt, s=180, color='orange',    marker='*', zorder=5, label=f'ESG Optimal (ESG={esg_opt:.2f})')
ax.scatter(0,      r_free,  s=100, color='green',     marker='s', zorder=5, label='Risk-Free Asset')

U_mv_star   = ret_mv - (gamma / 2) * sd_mv**2
sigma_curve = np.linspace(0, portfolios['Std Dev'].max() * 1.2, 200)
mu_mv_curve = U_mv_star + (gamma / 2) * sigma_curve**2
ax.plot(sigma_curve, mu_mv_curve, ':', linewidth=1.5,
        color='steelblue', label='MV Indifference Curve')

mu_esg_curve = (u_opt - lam * esg_opt) + (gamma / 2) * sigma_curve**2
ax.plot(sigma_curve, mu_esg_curve, '-.', linewidth=1.5,
        color='orange', label='ESG Indifference Curve')

ax.set_xlabel('Risk (Standard Deviation)')
ax.set_ylabel('Expected Return')
ax.set_title('Sustainable Portfolio Optimisation\n(MV vs ESG Optimal with CML)')
ax.legend(fontsize=8.5)
ax.grid(True, alpha=0.3)
plt.tight_layout()

st.pyplot(fig)
plt.close(fig)

st.caption("ECN316 Sustainable Finance")