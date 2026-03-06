# “””
GreenPort — ESG Portfolio Optimiser

Sustainable Finance App (Phase 1 + Phase 2)
Based on Lecture 6: ESG Portfolio Management

Theory:
Utility: U = E[Rp] - (γ/2)·σ²p + λ·s̄
where s̄ = weighted average ESG score (risky assets only)
ESG-efficient frontier: max Sharpe ratio for each ESG constraint level
“””

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────

# PAGE CONFIG

# ─────────────────────────────────────────────

st.set_page_config(
page_title=“GreenPort — ESG Portfolio Optimiser”,
page_icon=“🌿”,
layout=“wide”,
initial_sidebar_state=“expanded”,
)

# ─────────────────────────────────────────────

# CUSTOM STYLING

# ─────────────────────────────────────────────

st.markdown(”””

<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        color: #2E7D52;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #555;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #2E7D52;
        border-bottom: 2px solid #2E7D52;
        padding-bottom: 4px;
        margin-top: 1.5rem;
    }
    .metric-box {
        background: #f0f8f4;
        border-left: 4px solid #2E7D52;
        padding: 0.6rem 1rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
    }
    .stMetric label { font-size: 0.75rem !important; }
    .formula-box {
        background: #f5f5f5;
        border-left: 4px solid #888;
        padding: 0.6rem 1rem;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.9rem;
        margin: 0.5rem 0;
    }
    footer {visibility: hidden;}
</style>

“””, unsafe_allow_html=True)

# ─────────────────────────────────────────────

# HEADER

# ─────────────────────────────────────────────

col_logo, col_title = st.columns([1, 8])
with col_logo:
st.markdown(”# 🌿”)
with col_title:
st.markdown(’<p class="main-header">GreenPort</p>’, unsafe_allow_html=True)
st.markdown(’<p class="sub-header">ESG Portfolio Optimiser — build a personalised portfolio balancing financial returns and sustainability</p>’, unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────

# SIDEBAR — INVESTOR PREFERENCES

# ─────────────────────────────────────────────

with st.sidebar:
st.image(“https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Financial_leaves.svg/120px-Financial_leaves.svg.png”,
width=60, use_container_width=False)
st.markdown(”## 🎯 Your Preferences”)

```
st.markdown("**Risk Aversion (γ)**")
gamma = st.slider(
    "γ — how much do you dislike risk?",
    min_value=0.5, max_value=10.0, value=3.0, step=0.5,
    help="γ = 1: risk tolerant | γ = 5: moderate | γ = 10: very risk-averse"
)
risk_label = "🟢 Low" if gamma <= 2 else ("🟡 Moderate" if gamma <= 5 else "🔴 High")
st.caption(f"Risk aversion level: **{risk_label}** (γ = {gamma})")

st.markdown("---")
st.markdown("**ESG Preference (λ)**")
lambda_ = st.slider(
    "λ — how much do you value sustainability?",
    min_value=0.0, max_value=5.0, value=1.5, step=0.25,
    help="λ = 0: standard investor (ESG irrelevant) | λ > 0: willing to sacrifice return for greener portfolio"
)
esg_label = "⚪ Standard" if lambda_ == 0 else ("🟡 ESG-aware" if lambda_ <= 2 else "🟢 ESG-motivated")
st.caption(f"ESG investor type: **{esg_label}** (λ = {lambda_})")

st.markdown("---")
st.markdown("**Risk-Free Rate**")
rf = st.slider(
    "rf (%)",
    min_value=0.0, max_value=8.0, value=2.5, step=0.25,
    help="Current risk-free rate (e.g. government bond yield)"
) / 100

st.markdown("---")
st.markdown("**Correlation between Assets**")
rho = st.slider(
    "ρ (correlation)",
    min_value=-1.0, max_value=1.0, value=0.3, step=0.05,
    help="How correlated are the two assets? Lower = more diversification benefit."
)

st.markdown("---")
st.markdown(
    '<div class="formula-box">U = E[Rp] − (γ/2)·σ²p + λ·s̄</div>',
    unsafe_allow_html=True
)
st.caption("Utility function from Lecture 6")
```

# ─────────────────────────────────────────────

# ASSET INPUTS — TWO COLUMNS

# ─────────────────────────────────────────────

st.markdown(’<p class="section-title">📋 Asset Characteristics</p>’, unsafe_allow_html=True)

col1, col2 = st.columns(2, gap=“large”)

with col1:
st.markdown(”### 🔵 Asset 1”)
name1 = st.text_input(“Asset name”, value=“Green Tech ETF”, key=“name1”)
er1   = st.slider(“Expected Return E[R₁] (%)”, 0.0, 30.0, 10.0, 0.5, key=“er1”) / 100
sd1   = st.slider(“Std Deviation σ₁ (%)”,       1.0, 50.0, 15.0, 0.5, key=“sd1”) / 100
esg1  = st.slider(“ESG Score (0 = worst, 1 = best)”, 0.0, 1.0, 0.78, 0.01, key=“esg1”,
help=“Normalised ESG score for this asset”)

with col2:
st.markdown(”### 🟠 Asset 2”)
name2 = st.text_input(“Asset name”, value=“Energy Value Fund”, key=“name2”)
er2   = st.slider(“Expected Return E[R₂] (%)”, 0.0, 30.0, 7.5, 0.5, key=“er2”) / 100
sd2   = st.slider(“Std Deviation σ₂ (%)”,       1.0, 50.0, 22.0, 0.5, key=“sd2”) / 100
esg2  = st.slider(“ESG Score (0 = worst, 1 = best)”, 0.0, 1.0, 0.32, 0.01, key=“esg2”,
help=“Normalised ESG score for this asset”)

# ─────────────────────────────────────────────

# CORE COMPUTATIONS

# ─────────────────────────────────────────────

N = 2000
weights = np.linspace(0, 1, N)   # w = fraction invested in Asset 1

# Portfolio metrics across all possible weights

er_p   = weights * er1 + (1 - weights) * er2
var_p  = (weights**2 * sd1**2
+ (1 - weights)**2 * sd2**2
+ 2 * weights * (1 - weights) * sd1 * sd2 * rho)
var_p  = np.maximum(var_p, 1e-12)   # numerical safety
sd_p   = np.sqrt(var_p)
esg_p  = weights * esg1 + (1 - weights) * esg2
sharpe_p = (er_p - rf) / sd_p

# Utility function: U = E[Rp] - (γ/2)·σ²p + λ·s̄

utility_p = er_p - (gamma / 2) * var_p + lambda_ * esg_p

# Optimal portfolio = max utility

opt_idx   = int(np.argmax(utility_p))
w_opt     = weights[opt_idx]
er_opt    = er_p[opt_idx]
sd_opt    = sd_p[opt_idx]
sharpe_opt = sharpe_p[opt_idx]
esg_opt   = esg_p[opt_idx]
utility_opt = utility_p[opt_idx]

# Max-Sharpe portfolio (λ = 0 benchmark)

sharpe_idx  = int(np.argmax(sharpe_p))
w_sharpe    = weights[sharpe_idx]
sharpe_max  = sharpe_p[sharpe_idx]
esg_sharpe  = esg_p[sharpe_idx]

# ESG-Efficient Frontier: for each ESG target, find max achievable Sharpe

esg_min = min(esg1, esg2)
esg_max = max(esg1, esg2)
esg_targets = np.linspace(esg_min, esg_max, 400)
frontier_sharpe = []
for s_t in esg_targets:
mask = esg_p >= s_t
if mask.any():
frontier_sharpe.append(float(sharpe_p[mask].max()))
else:
frontier_sharpe.append(np.nan)
frontier_sharpe = np.array(frontier_sharpe)

# ESG cost = drop in Sharpe when moving from max-Sharpe to optimal

esg_cost = sharpe_max - sharpe_opt

# ─────────────────────────────────────────────

# RESULTS SECTION

# ─────────────────────────────────────────────

st.markdown(’<p class="section-title">🏆 Optimal Portfolio</p>’, unsafe_allow_html=True)

# Key metrics row

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric(f”Weight: {name1}”, f”{w_opt:.1%}”,
delta=f”{w_opt - 0.5:.1%} vs 50/50”)
m2.metric(f”Weight: {name2}”, f”{1 - w_opt:.1%}”,
delta=f”{(1-w_opt) - 0.5:.1%} vs 50/50”)
m3.metric(“Expected Return”, f”{er_opt:.2%}”,
delta=f”{er_opt - rf:.2%} excess return”)
m4.metric(“Std Dev (Risk)”, f”{sd_opt:.2%}”)
m5.metric(“Sharpe Ratio”, f”{sharpe_opt:.3f}”,
delta=f”{sharpe_opt - sharpe_max:.3f} vs max Sharpe”)
m6.metric(“ESG Score”, f”{esg_opt:.3f}”,
delta=f”{esg_opt - esg_sharpe:.3f} vs max Sharpe ptf”)

# ESG cost callout

if lambda_ > 0 and esg_cost > 0.001:
st.info(
f”🌱 **Sustainability–Performance Trade-off:** “
f”Your ESG preference (λ = {lambda_}) leads to a portfolio with ESG score {esg_opt:.2f} “
f”vs {esg_sharpe:.2f} for the pure max-Sharpe portfolio. “
f”The cost is a **{esg_cost:.3f} drop in Sharpe ratio** — “
f”the price of a greener portfolio.”
)
elif lambda_ == 0:
st.info(“ℹ️ With λ = 0, the optimal portfolio maximises financial utility only (no ESG preference).”)

st.markdown(”—”)

# Summary Table

st.markdown(’<p class="section-title">📊 Portfolio Summary Table</p>’, unsafe_allow_html=True)

sharpe1 = (er1 - rf) / sd1 if sd1 > 0 else 0
sharpe2 = (er2 - rf) / sd2 if sd2 > 0 else 0

df_summary = pd.DataFrame({
“Asset / Portfolio”: [name1, name2, “⭐ Optimal Portfolio”, “📈 Max-Sharpe Portfolio”],
“Weight in Asset 1”: [f”100%”, “0%”, f”{w_opt:.1%}”, f”{w_sharpe:.1%}”],
“Weight in Asset 2”: [f”0%”, “100%”, f”{1-w_opt:.1%}”, f”{1-w_sharpe:.1%}”],
“E[R]”:   [f”{er1:.2%}”, f”{er2:.2%}”, f”{er_opt:.2%}”, f”{er_p[sharpe_idx]:.2%}”],
“Std Dev σ”: [f”{sd1:.2%}”, f”{sd2:.2%}”, f”{sd_opt:.2%}”, f”{sd_p[sharpe_idx]:.2%}”],
“Sharpe Ratio”: [f”{sharpe1:.3f}”, f”{sharpe2:.3f}”, f”{sharpe_opt:.3f}”, f”{sharpe_max:.3f}”],
“ESG Score”: [f”{esg1:.3f}”, f”{esg2:.3f}”, f”{esg_opt:.3f}”, f”{esg_sharpe:.3f}”],
“Utility U”: [”—”, “—”, f”{utility_opt:.4f}”, f”{utility_p[sharpe_idx]:.4f}”],
})
st.dataframe(df_summary, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────

# CHARTS

# ─────────────────────────────────────────────

st.markdown(’<p class="section-title">📈 Visualisations</p>’, unsafe_allow_html=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(’#FAFAFA’)

DARK_GREEN  = “#2E7D52”
BLUE_ASSET  = “#1565C0”
ORG_ASSET   = “#E64A19”
GOLD_OPT    = “#F9A825”
GREY_BENCH  = “#757575”

# ── Chart 1: Mean-Variance Frontier ──────────────────────────────────────────

ax = axes[0]
ax.set_facecolor(’#F5F5F5’)
ax.plot(sd_p * 100, er_p * 100,
color=DARK_GREEN, lw=2.5, label=“Mean-Variance Frontier”, zorder=2)

# Capital Market Line from rf to tangency

if sharpe_max > 0:
cml_x = np.linspace(0, max(sd_p) * 1.1, 200)
cml_y = rf * 100 + sharpe_max * cml_x
ax.plot(cml_x, cml_y, ‘–’, color=GREY_BENCH, lw=1.5, alpha=0.7,
label=“Capital Market Line”)

# Individual assets

ax.scatter([sd1 * 100], [er1 * 100], color=BLUE_ASSET, s=120, zorder=5,
label=name1, marker=‘D’)
ax.scatter([sd2 * 100], [er2 * 100], color=ORG_ASSET,  s=120, zorder=5,
label=name2, marker=‘D’)

# Max-Sharpe portfolio

ax.scatter([sd_p[sharpe_idx]*100], [er_p[sharpe_idx]*100],
color=GREY_BENCH, s=110, zorder=5, marker=‘s’,
label=“Max-Sharpe Portfolio”)

# Optimal portfolio (star)

ax.scatter([sd_opt * 100], [er_opt * 100],
color=GOLD_OPT, s=220, zorder=6, marker=’*’,
edgecolors=‘black’, linewidths=0.5,
label=“★ Optimal Portfolio (Your Choice)”)

ax.set_xlabel(“Portfolio Std Dev, σp (%)”, fontsize=11)
ax.set_ylabel(“Expected Return, E[Rp] (%)”, fontsize=11)
ax.set_title(“Mean-Variance Frontier”, fontsize=13, fontweight=‘bold’)
ax.legend(fontsize=8.5, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle=’–’)
ax.spines[[‘top’, ‘right’]].set_visible(False)

# ── Chart 2: ESG-Efficient Frontier ──────────────────────────────────────────

ax2 = axes[1]
ax2.set_facecolor(’#F5F5F5’)

# Frontier curve

valid = ~np.isnan(frontier_sharpe)
ax2.plot(esg_targets[valid], frontier_sharpe[valid],
color=DARK_GREEN, lw=2.5,
label=“ESG-Efficient Frontier\n(max Sharpe at each ESG level)”, zorder=2)

# Individual assets

ax2.scatter([esg1], [sharpe1], color=BLUE_ASSET, s=120, zorder=5,
label=name1, marker=‘D’)
ax2.scatter([esg2], [sharpe2], color=ORG_ASSET,  s=120, zorder=5,
label=name2, marker=‘D’)

# Max-Sharpe portfolio

ax2.scatter([esg_sharpe], [sharpe_max],
color=GREY_BENCH, s=110, zorder=5, marker=‘s’,
label=“Max-Sharpe Portfolio”)

# Optimal portfolio

ax2.scatter([esg_opt], [sharpe_opt],
color=GOLD_OPT, s=220, zorder=6, marker=’*’,
edgecolors=‘black’, linewidths=0.5,
label=“★ Optimal Portfolio (Your Choice)”)

# Annotate ESG cost

if esg_cost > 0.002 and lambda_ > 0:
ax2.annotate(””,
xy=(esg_opt, sharpe_opt),
xytext=(esg_sharpe, sharpe_max),
arrowprops=dict(arrowstyle=”<->”, color=“red”, lw=1.5))
mid_esg = (esg_opt + esg_sharpe) / 2
mid_sr  = (sharpe_opt + sharpe_max) / 2
ax2.text(mid_esg + 0.01, mid_sr,
f”ESG cost\n−{esg_cost:.3f} SR”,
fontsize=8, color=“red”)

ax2.set_xlabel(“Portfolio ESG Score (s̄)”, fontsize=11)
ax2.set_ylabel(“Sharpe Ratio”, fontsize=11)
ax2.set_title(“ESG-Efficient Frontier”, fontsize=13, fontweight=‘bold’)
ax2.legend(fontsize=8.5, framealpha=0.9, loc=‘lower left’)
ax2.grid(True, alpha=0.3, linestyle=’–’)
ax2.spines[[‘top’, ‘right’]].set_visible(False)

plt.tight_layout(pad=2.0)
st.pyplot(fig, use_container_width=True)
plt.close(fig)

# ─────────────────────────────────────────────

# THEORY EXPLAINER (collapsible)

# ─────────────────────────────────────────────

with st.expander(“📚 How it works — Theory (Lecture 6)”):
st.markdown(”””
**Utility Function with ESG Preferences**

```
We extend the standard mean-variance utility with an ESG term:

> **U = E[Rp] − (γ/2)·σ²p + λ·s̄**

- **E[Rp]** = expected portfolio return
- **σ²p** = portfolio variance (risk)
- **γ** = absolute risk-aversion (higher = more risk-averse)
- **s̄** = weighted average ESG score of the portfolio
- **λ** = ESG preference intensity (λ = 0: standard investor; λ > 0: values sustainability)

**ESG-Efficient Frontier**

For each target ESG level *s*, we find the portfolio that maximises the Sharpe ratio
subject to the constraint that the portfolio ESG score ≥ *s*.
This traces out the frontier in (ESG score, Sharpe ratio) space.

**Sustainability–Performance Trade-off**

The cost of ESG preferences = drop in Sharpe ratio when choosing a greener portfolio
than the one that maximises Sharpe (the tangency portfolio).
""")
```

# ─────────────────────────────────────────────

# FOOTER

# ─────────────────────────────────────────────

st.markdown(”—”)
st.caption(“🌿 **GreenPort** — ECN316 Sustainable Finance | Built with Streamlit & Python”)
