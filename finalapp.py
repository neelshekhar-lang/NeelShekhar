import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(
page_title=“GreenPort — Sustainable Finance App”,
page_icon=“🌿”,
layout=“wide”,
initial_sidebar_state=“expanded”,
)

st.markdown(”””

<style>
    /* Sidebar green background */
    [data-testid="stSidebar"] {
        background-color: #1B4332;
    }
    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }

    /* Main background off-white */
    .stApp {
        background-color: #F8FAF9;
    }

    /* Green headings */
    h1, h2, h3 {
        color: #1B4332 !important;
    }

    /* Green divider */
    hr {
        border-color: #2D6A4F;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #D8F3DC;
        border-left: 4px solid #2D6A4F;
        border-radius: 6px;
        padding: 8px 12px;
    }

    /* Dataframe header */
    [data-testid="stDataFrame"] thead {
        background-color: #2D6A4F;
        color: white;
    }

    footer {visibility: hidden;}
</style>

“””, unsafe_allow_html=True)

# ——————————

# Header

# ——————————

st.markdown(”# 🌿 GreenPort”)
st.markdown(”**Sustainable Finance Portfolio Optimiser** — ECN316 Group Project”)
st.markdown(”*Utility function: U = E[Rp] - (γ/2)·σ²p + λ·s̄*”)
st.divider()

# ——————————

# Sidebar inputs

# ——————————

with st.sidebar:
st.markdown(”## ⚙️ Inputs”)

```
st.markdown("### Asset 1")
r1   = st.number_input("Expected Return (%)", value=10.0, step=0.5, key="r1") / 100
sd1  = st.number_input("Standard Deviation (%)", value=15.0, step=0.5, key="sd1") / 100
esg1 = st.number_input("ESG Score (0 to 1)", value=0.8, step=0.01, min_value=0.0, max_value=1.0, key="esg1")

st.markdown("### Asset 2")
r2   = st.number_input("Expected Return (%)", value=7.0, step=0.5, key="r2") / 100
sd2  = st.number_input("Standard Deviation (%)", value=22.0, step=0.5, key="sd2") / 100
esg2 = st.number_input("ESG Score (0 to 1)", value=0.3, step=0.01, min_value=0.0, max_value=1.0, key="esg2")

st.markdown("### Market & Preferences")
rho    = st.number_input("Correlation rho", value=0.3, step=0.05, min_value=-1.0, max_value=1.0)
r_free = st.number_input("Risk-Free Rate (%)", value=2.5, step=0.25) / 100
gamma  = st.number_input("Risk Aversion gamma", value=3.0, step=0.5, min_value=0.1)
lam    = st.number_input("ESG Preference lambda", value=1.5, step=0.25, min_value=0.0)
```

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

idx_opt = portfolios[‘Utility’].idxmax()
w1_opt  = portfolios.loc[idx_opt, ‘Weight Asset 1’]
ret_opt = portfolios.loc[idx_opt, ‘Return’]
sd_opt  = portfolios.loc[idx_opt, ‘Std Dev’]
esg_opt = portfolios.loc[idx_opt, ‘ESG Score’]
u_opt   = portfolios.loc[idx_opt, ‘Utility’]
sharpe_opt = portfolios.loc[idx_opt, ‘Sharpe’]

idx_tan = portfolios[‘Sharpe’].idxmax()
w1_tan  = portfolios.loc[idx_tan, ‘Weight Asset 1’]
ret_tan = portfolios.loc[idx_tan, ‘Return’]
sd_tan  = portfolios.loc[idx_tan, ‘Std Dev’]
sharpe_max = portfolios.loc[idx_tan, ‘Sharpe’]

mv_utilities = portfolios[‘Return’] - (gamma / 2) * portfolios[‘Std Dev’]**2
idx_mv  = mv_utilities.idxmax()
w1_mv   = portfolios.loc[idx_mv, ‘Weight Asset 1’]
ret_mv  = portfolios.loc[idx_mv, ‘Return’]
sd_mv   = portfolios.loc[idx_mv, ‘Std Dev’]

esg_cost = sharpe_max - sharpe_opt

# ——————————

# Output tables

# ——————————

st.subheader(“📊 Portfolio Comparison”)

comparison = pd.DataFrame({
‘Portfolio’:           [‘MV Optimal (lambda=0)’, ‘ESG Optimal’, ‘Tangency (Max Sharpe)’],
‘Weight Asset 1 (%)’:  [round(w1_mv*100, 2),   round(w1_opt*100, 2),   round(w1_tan*100, 2)],
‘Weight Asset 2 (%)’:  [round((1-w1_mv)*100,2), round((1-w1_opt)*100,2),round((1-w1_tan)*100,2)],
})
st.dataframe(comparison, hide_index=True, use_container_width=True)

st.subheader(“🏆 ESG Optimal Portfolio”)

col_a, col_b, col_c, col_d, col_e = st.columns(5)
col_a.metric(“Expected Return”,   f”{ret_opt*100:.2f}%”)
col_b.metric(“Std Dev (Risk)”,    f”{sd_opt*100:.2f}%”)
col_c.metric(“ESG Score”,         f”{esg_opt:.3f}”)
col_d.metric(“Sharpe Ratio”,      f”{sharpe_opt:.3f}”)
col_e.metric(“ESG Cost (Sharpe)”, f”{esg_cost:.3f}”)

if lam > 0 and esg_cost > 0.001:
st.info(f”🌱 Your ESG preference (lambda = {lam}) shifts the portfolio to a greener choice, “
f”at a cost of {esg_cost:.3f} in Sharpe ratio vs the pure max-Sharpe portfolio.”)

st.divider()

# ——————————

# Plot

# ——————————

st.subheader(“📈 Efficient Frontier”)

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(’#F8FAF9’)
ax.set_facecolor(’#F0F7F4’)

ax.plot(portfolios[‘Std Dev’], portfolios[‘Return’],
color=’#2D6A4F’, linewidth=2.5, label=‘Efficient Frontier’)

if sd_tan > 0:
sd_range = np.linspace(0, portfolios[‘Std Dev’].max() * 1.2, 200)
ret_cml  = r_free + (ret_tan - r_free) / sd_tan * sd_range
ax.plot(sd_range, ret_cml, ‘–’, color=’#74C69D’, linewidth=1.5, label=‘Capital Market Line’)

ax.scatter(sd_mv,  ret_mv,  s=120, color=’#52B788’, marker=‘D’, zorder=5, label=‘MV Optimal’)
ax.scatter(sd_tan, ret_tan, s=180, color=’#B7E4C7’, marker=’*’, zorder=5, edgecolors=’#1B4332’, linewidths=0.8, label=‘Tangency Portfolio’)
ax.scatter(sd_opt, ret_opt, s=200, color=’#F9C74F’, marker=’*’, zorder=6, edgecolors=’#1B4332’, linewidths=0.8, label=f’ESG Optimal (ESG={esg_opt:.2f})’)
ax.scatter(0,      r_free,  s=100, color=’#1B4332’, marker=‘s’, zorder=5, label=‘Risk-Free Asset’)

U_mv_star   = ret_mv - (gamma / 2) * sd_mv**2
sigma_curve = np.linspace(0, portfolios[‘Std Dev’].max() * 1.2, 200)
mu_mv_curve = U_mv_star + (gamma / 2) * sigma_curve**2
ax.plot(sigma_curve, mu_mv_curve, ‘:’, linewidth=1.5, color=’#52B788’, label=‘MV Indifference Curve’)

mu_esg_curve = (u_opt - lam * esg_opt) + (gamma / 2) * sigma_curve**2
ax.plot(sigma_curve, mu_esg_curve, ‘-.’, linewidth=1.5, color=’#F9C74F’, label=‘ESG Indifference Curve’)

ax.set_xlabel(‘Risk (Standard Deviation)’, fontsize=11)
ax.set_ylabel(‘Expected Return’, fontsize=11)
ax.set_title(‘Sustainable Portfolio Optimisation\n(MV vs ESG Optimal with CML)’, fontsize=13, fontweight=‘bold’, color=’#1B4332’)
ax.legend(fontsize=8.5, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle=’–’)
ax.spines[[‘top’, ‘right’]].set_visible(False)
plt.tight_layout()

st.pyplot(fig)
plt.close(fig)

st.divider()
st.caption(“Green Compass — ECN316 Sustainable Finance”)