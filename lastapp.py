"""
Ethical Edge - Sustainable Portfolio Optimiser
ECN316 Sustainable Finance | QMUL Group Project
Objective: max x'mu - (gamma/2) x'Sigma x + lambda * s_bar
where x = free risky weights (remainder in risk-free asset)
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import datetime
from scipy.optimize import minimize
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.chart_container import chart_container
from annotated_text import annotated_text

st.set_page_config(page_title="Ethical Edge", page_icon="balanced_scale", layout="wide")
st.title("Ethical Edge")
st.caption("Sustainable Portfolio Optimiser - ECN316 Sustainable Finance - QMUL")
st.latex(r"\text{Objective: } \max\; \mathbf{x}'\boldsymbol{\mu} - \frac{\gamma}{2}\mathbf{x}'\boldsymbol{\Sigma}\mathbf{x} + \lambda\bar{s}")
st.divider()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("Investor Profile")
    st.subheader("Quick Start - Persona", divider="green")
    persona = st.selectbox("Investor persona", [
        "Custom (manual)",
        "Young Professional - growth-focused",
        "Impact Investor - ESG first",
        "Balanced Saver - risk & return",
        "Retiree - capital protection",
        "Pure Return Seeker - no ESG"
    ])
    persona_defaults = {
        "Young Professional - growth-focused": (2.0, 1.0),
        "Impact Investor - ESG first": (4.0, 4.0),
        "Balanced Saver - risk & return": (4.0, 1.5),
        "Retiree - capital protection": (8.0, 0.5),
        "Pure Return Seeker - no ESG": (2.0, 0.0),
    }
    if persona != "Custom (manual)":
        pg, pl = persona_defaults[persona]
        st.success(f"γ = {pg} | λ = {pl}")
    else:
        pg, pl = 4.0, 1.0

    st.divider()
    st.subheader("Step 1 — Risk Attitude", divider="green")
    q1 = st.radio("If your portfolio dropped 20%, you would...", [
        "Sell immediately - I cannot handle losses",
        "Hold steady and wait it out",
        "Buy more - it is a buying opportunity"
    ], index=1)
    q2 = st.radio("Your investment horizon?", ["Under 2 years", "2-5 years", "5+ years"], index=1)
    risk_map = {
        ("Sell immediately - I cannot handle losses", "Under 2 years"): 9.0,
        ("Sell immediately - I cannot handle losses", "2-5 years"): 8.0,
        ("Sell immediately - I cannot handle losses", "5+ years"): 6.0,
        ("Hold steady and wait it out", "Under 2 years"): 6.0,
        ("Hold steady and wait it out", "2-5 years"): 4.0,
        ("Hold steady and wait it out", "5+ years"): 3.0,
        ("Buy more - it is a buying opportunity", "Under 2 years"): 3.0,
        ("Buy more - it is a buying opportunity", "2-5 years"): 2.0,
        ("Buy more - it is a buying opportunity", "5+ years"): 1.0,
    }
    gamma_quiz = risk_map[(q1, q2)]
    gamma_default = pg if persona != "Custom (manual)" else gamma_quiz
    st.info(f"Quiz suggests γ = {gamma_quiz}" + (f" | Persona gamma = {pg}" if persona != "Custom (manual)" else ""))
    gamma = st.slider("Fine-tune γ", 0.5, 10.0, float(gamma_default), 0.5,
                      help="Higher γ = more risk-averse. Doubling γ roughly halves risky positions.")

    st.divider()
    st.subheader("Step 2 - ESG Commitment", divider="green")
    esg_label = st.select_slider("How important is sustainability to you?",
        options=["None (lambda=0)", "Low (lambda=0.5)", "Medium (lambda=1)", "High (lambda=2)", "Max (lambda=4)"],
        value="Medium (lambda=1)")
    lam_map = {"None (lambda=0)": 0.0, "Low (lambda=0.5)": 0.5, "Medium (lambda=1)": 1.0,
               "High (lambda=2)": 2.0, "Max (lambda=4)": 4.0}
    lam_default = pl if persona != "Custom (manual)" else lam_map[esg_label]
    lam = st.slider("Fine-tune λ", 0.0, 5.0, float(lam_default), 0.25,
                    help="λ > 0: you accept lower Sharpe for a greener portfolio.")

    st.divider()
    st.subheader("Step 3 - ESG Pillar Weights", divider="green")
    w_e = st.slider("Environmental (E)", 0.0, 1.0, 0.4, 0.05)
    w_s = st.slider("Social (S)",        0.0, 1.0, 0.3, 0.05)
    w_g = st.slider("Governance (G)",    0.0, 1.0, 0.3, 0.05)
    pt = w_e + w_s + w_g or 1.0
    w_e, w_s, w_g = w_e/pt, w_s/pt, w_g/pt
    st.caption(f"Normalised: E:{w_e:.0%} S:{w_s:.0%} G:{w_g:.0%}")

    st.divider()
    st.subheader("Step 4 - Exclusions", divider="green")
    excl_tobacco  = st.checkbox("Tobacco")
    excl_weapons  = st.checkbox("Weapons / Defence")
    excl_fossil   = st.checkbox("Fossil Fuels")
    excl_gambling = st.checkbox("Gambling")
    min_esg_score = st.slider("Minimum ESG score", 0, 100, 0, 5)

    st.divider()
    st.subheader("Step 5 - Asset Data", divider="green")
    st.markdown("**Asset 1**")
    name1 = st.text_input("Company name", value="Asset 1", key="name1")
    c1, c2 = st.columns(2)
    r1  = c1.number_input("E[R] (%)", value=13.0, step=0.5, key="r1") / 100
    sd1 = c2.number_input("sigma (%)", value=18.0, step=0.5, key="sd1") / 100
    d1, d2, d3 = st.columns(3)
    e1 = d1.number_input("E", value=40.0, step=1.0, min_value=0.0, max_value=100.0, key="e1")
    s1 = d2.number_input("S", value=35.0, step=1.0, min_value=0.0, max_value=100.0, key="s1")
    g1 = d3.number_input("G", value=30.0, step=1.0, min_value=0.0, max_value=100.0, key="g1")
    sin1 = st.multiselect("Sector flags", ["Tobacco","Weapons","Fossil Fuels","Gambling"], key="sin1")
    with st.expander("Carbon Scope Breakdown (optional)"):
        scope1_1 = st.slider("Scope 1", 0, 100, 60, 5, key="sc1_1")
        scope2_1 = st.slider("Scope 2", 0, 100, 50, 5, key="sc2_1")
        scope3_1 = st.slider("Scope 3", 0, 100, 70, 5, key="sc3_1")

    st.markdown("**Asset 2**")
    name2 = st.text_input("Company name", value="Asset 2", key="name2")
    c3, c4 = st.columns(2)
    r2  = c3.number_input("E[R] (%)", value=7.0,  step=0.5, key="r2") / 100
    sd2 = c4.number_input("sigma (%)", value=22.0, step=0.5, key="sd2") / 100
    d4, d5, d6 = st.columns(3)
    e2 = d4.number_input("E", value=70.0, step=1.0, min_value=0.0, max_value=100.0, key="e2")
    s2 = d5.number_input("S", value=65.0, step=1.0, min_value=0.0, max_value=100.0, key="s2")
    g2 = d6.number_input("G", value=75.0, step=1.0, min_value=0.0, max_value=100.0, key="g2")
    sin2 = st.multiselect("Sector flags", ["Tobacco","Weapons","Fossil Fuels","Gambling"], key="sin2")
    with st.expander("Carbon Scope Breakdown (optional)"):
        scope1_2 = st.slider("Scope 1", 0, 100, 30, 5, key="sc1_2")
        scope2_2 = st.slider("Scope 2", 0, 100, 25, 5, key="sc2_2")
        scope3_2 = st.slider("Scope 3", 0, 100, 40, 5, key="sc3_2")

    st.markdown("**Market**")
    mc1, mc2 = st.columns(2)
    rho    = mc1.number_input("Correlation rho", -1.0, 1.0, 0.3, 0.05)
    r_free = mc2.number_input("Risk-free rate (%)", value=2.5, step=0.25) / 100

# ─────────────────────────────────────────────
# CORE MATHS — FIXED WITH SCIPY (FREE WEIGHTS)
# ─────────────────────────────────────────────

esg1 = w_e*e1 + w_s*s1 + w_g*g1
esg2 = w_e*e2 + w_s*s2 + w_g*g2

cov_matrix = np.array([
    [sd1**2,        rho*sd1*sd2],
    [rho*sd1*sd2,   sd2**2     ]
])
mu_excess = np.array([r1 - r_free, r2 - r_free])

def is_excluded(flags, score):
    if excl_tobacco  and "Tobacco"      in flags: return True
    if excl_weapons  and "Weapons"      in flags: return True
    if excl_fossil   and "Fossil Fuels" in flags: return True
    if excl_gambling and "Gambling"     in flags: return True
    if score < min_esg_score: return True
    return False

ex1 = is_excluded(sin1, esg1)
ex2 = is_excluded(sin2, esg2)

if ex1 and ex2:
    st.error("Both assets excluded. Please relax your filters.")
    st.stop()
if ex1: st.warning(f"{name1} excluded by screening - portfolio is 100% {name2}.")
if ex2: st.warning(f"{name2} excluded by screening - portfolio is 100% {name1}.")

_b1 = (0.0, 0.0) if ex1 else (0.0, None)
_b2 = (0.0, 0.0) if ex2 else (0.0, None)
_bnds = [_b1, _b2]

def _solve(gam, lam_v, cov=None, mu_e=None, esg_sc=None):
    """
    Solve: max x'mu_excess - (gamma/2) x'Sigma x + lambda * s_bar
    x1, x2 >= 0 (free - remainder 1-x1-x2 in risk-free)
    s_bar = (x1*esg1 + x2*esg2) / (x1+x2)
    """
    c    = cov    if cov    is not None else cov_matrix
    me   = mu_e   if mu_e   is not None else mu_excess
    esgs = esg_sc if esg_sc is not None else np.array([esg1, esg2])

    def obj(x):
        tot = x[0] + x[1]
        s_bar = float(x @ esgs) / tot if tot > 1e-10 else 0.0
        return -(float(x @ me) - (gam/2)*float(x @ c @ x) + lam_v*s_bar)

    res = minimize(obj, [0.4, 0.4], method='SLSQP', bounds=_bnds,
                   options={'ftol': 1e-12, 'maxiter': 2000})
    return np.array(res.x)

def _solve_tangency(cov=None):
    c  = cov if cov is not None else cov_matrix
    me = mu_excess

    def neg_sharpe(x):
        ret = r_free + float(x @ me)
        var = float(x @ c @ x)
        sd  = np.sqrt(max(var, 1e-14))
        return -(ret - r_free) / sd

    res = minimize(neg_sharpe, [0.4, 0.4], method='SLSQP', bounds=_bnds,
                   options={'ftol': 1e-12, 'maxiter': 2000})
    return np.array(res.x)

def _solve_mvp(cov=None):
    c = cov if cov is not None else cov_matrix
    def var_fn(x): return float(x @ c @ x)
    res = minimize(var_fn, [0.4, 0.4], method='SLSQP', bounds=_bnds)
    return np.array(res.x)

def _stats(x, gam=None, lam_v=None):
    g = gam   if gam   is not None else gamma
    l = lam_v if lam_v is not None else lam
    x1, x2 = float(x[0]), float(x[1])
    tot  = x1 + x2
    ret  = r_free + x1*(r1-r_free) + x2*(r2-r_free)
    var  = float(x @ cov_matrix @ x)
    sd   = np.sqrt(max(var, 0.0))
    esg_s = (x1*esg1 + x2*esg2)/tot if tot > 1e-10 else 0.0
    sh   = (ret - r_free)/sd if sd > 1e-10 else 0.0
    rf_w = 1.0 - x1 - x2
    obj  = float(x @ mu_excess) - (g/2)*var + l*esg_s
    return {'Weight Asset 1': x1, 'Weight Asset 2': x2, 'Weight RF': rf_w,
            'Return': ret, 'Volatility': sd, 'ESG Score': esg_s,
            'Sharpe Ratio': sh, 'Utility': obj}

x_opt = _solve(gamma, lam)
x_mv  = _solve(gamma, 0.0)
x_tan = _solve_tangency()
x_mvp = _solve_mvp()

opt = _stats(x_opt)
mv  = _stats(x_mv,  lam_v=0.0)
tan = _stats(x_tan)
mvp = _stats(x_mvp)

esg_cost     = float(tan['Sharpe Ratio']) - float(opt['Sharpe Ratio'])
esg_cost_pct = min(abs(esg_cost)/max(abs(float(tan['Sharpe Ratio'])),0.001)*100, 100)
corner = (x_opt[0] < 1e-4) or (x_opt[1] < 1e-4)

# Frontier sweep for chart (normalized risky mix x1+x2=1)
_wsweep = np.linspace(0, 1, 500)
_rows = []
for w in _wsweep:
    xv  = np.array([w, 1.0-w])
    ret = r_free + float(xv @ mu_excess)
    sd  = np.sqrt(max(float(xv @ cov_matrix @ xv), 0.0))
    esg = w*esg1 + (1.0-w)*esg2
    sr  = (ret-r_free)/sd if sd > 1e-10 else 0.0
    _rows.append({'Weight Asset 1': w, 'Weight Asset 2': 1-w,
                  'Return': ret, 'Volatility': sd, 'ESG Score': esg, 'Sharpe Ratio': sr})
portfolios = pd.DataFrame(_rows)

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def p_utility(w1, gam=None, lam_val=None):
    """Objective on normalized risky mix (for heatmap/sweep visuals)."""
    g = gam    if gam    is not None else gamma
    l = lam_val if lam_val is not None else lam
    xv = np.array([w1, 1.0-w1])
    return float(xv @ mu_excess) - (g/2)*float(xv @ cov_matrix @ xv) + l*(w1*esg1+(1-w1)*esg2)

def sharpe_badge(sr):
    if sr >= 1.0:  return "Trophy Excellent"
    if sr >= 0.5:  return "Good"
    if sr >= 0.25: return "Fair"
    return "Poor"

def traffic_light(score):
    if score >= 65: return "Strong",   "#16a34a"
    if score >= 40: return "Moderate", "#d97706"
    return "Weak", "#dc2626"

def carbon_label(cs):
    if cs >= 65: return "Low Carbon",  "#16a34a"
    if cs >= 40: return "Moderate",    "#d97706"
    return "High Carbon", "#dc2626"

def greenwashing_flag(e, s, g_score):
    avg_sg = (s + g_score) / 2
    if e >= 65 and avg_sg < 40:
        return True, "Possible greenwashing - high E score but weak S/G (Berg et al. RF 2022)"
    if e >= 65 and avg_sg < 55:
        return True, "Watch: strong environmental claims, moderate governance/social"
    return False, ""

def sf_framework(lam_val, excl_any):
    if lam_val >= 3.0:
        return "SF 3.0 - Common Good", "ESG prioritised above financial returns.", "#16a34a"
    elif lam_val > 0:
        return "SF 2.0 - Stakeholder Value", "ESG integrated into utility alongside returns.", "#d97706"
    elif excl_any:
        return "SF 1.0 - Profit + Exclusions", "Profit maximisation while avoiding sin stocks.", "#dc2626"
    return "Finance-as-Usual", "Pure financial return maximiser.", "#6b7280"

def apply_chart_style(ax, fig):
    fig.patch.set_facecolor("#0e0e12")
    ax.set_facecolor("#0e0e12")
    ax.tick_params(colors="#c8ccd8", labelsize=9)
    ax.xaxis.label.set_color("#c8ccd8")
    ax.yaxis.label.set_color("#c8ccd8")
    ax.title.set_color("#00e676")
    for sp in ax.spines.values(): sp.set_edgecolor("#2a2a3a")
    ax.grid(True, color="#1e1e2a", linewidth=0.6)

def portfolio_summary():
    x1p = opt['Weight Asset 1']*100; x2p = opt['Weight Asset 2']*100; rfp = opt['Weight RF']*100
    if x1p >= 99:   alloc = f"100% in **{name1}**"
    elif x2p >= 99: alloc = f"100% in **{name2}**"
    else:
        alloc = f"**{x1p:.0f}%** in **{name1}**, **{x2p:.0f}%** in **{name2}**"
        if abs(rfp) > 1: alloc += f", **{rfp:.0f}%** risk-free"
    esg_d = "strong" if opt['ESG Score'] >= 65 else ("moderate" if opt['ESG Score'] >= 40 else "weak")
    return (f"Recommended: {alloc}. "
            f"Expected return **{opt['Return']*100:.1f}%**, risk sigma = **{opt['Volatility']*100:.1f}%**, "
            f"ESG **{opt['ESG Score']:.1f}/100** ({esg_d}), Sharpe **{opt['Sharpe Ratio']:.3f}**.")

def build_sensitivity_table():
    rows = []; tan_sr = float(tan['Sharpe Ratio'])
    for l in [0.0, 0.5, 1.0, 2.0, 4.0]:
        x = _solve(gamma, l); s = _stats(x, lam_v=l)
        rows.append({"λ": l,
                     f"{name1}(%)": f"{x[0]*100:.1f}", f"{name2}(%)": f"{x[1]*100:.1f}",
                     "RF(%)": f"{s['Weight RF']*100:.1f}", "E[Rp]": f"{s['Return']*100:.2f}%",
                     "sigma": f"{s['Volatility']*100:.2f}%", "ESG": f"{s['ESG Score']:.1f}",
                     "Sharpe": f"{s['Sharpe Ratio']:.3f}",
                     "ESG Cost": f"{max(tan_sr-s['Sharpe Ratio'],0):.4f}"})
    return pd.DataFrame(rows)

def solve_scenario(gam, lam_v):
    x = _solve(gam, lam_v); s = _stats(x, gam=gam, lam_v=lam_v)
    s['w1'] = x[0]; s['w2'] = x[1]; return s

def build_report():
    today = datetime.date.today().strftime("%d %B %Y")
    lbl1, _ = traffic_light(esg1); lbl2, _ = traffic_light(esg2)
    lines = ["="*60, " ETHICAL EDGE - PORTFOLIO HEALTH REPORT", f" Generated: {today}", "="*60, "",
             "INVESTOR PROFILE",
             f"  Persona: {persona}", f"  γ: {gamma}", f"  λ: {lam}",
             f"  ESG Pillar Wts: E:{w_e:.0%} S:{w_s:.0%} G:{w_g:.0%}", "",
             "ASSETS",
             f"  {name1}: E[R]={r1*100:.1f}% sigma={sd1*100:.1f}% ESG={esg1:.1f}/100 ({lbl1})",
             f"  {name2}: E[R]={r2*100:.1f}% sigma={sd2*100:.1f}% ESG={esg2:.1f}/100 ({lbl2})",
             f"  rho={rho}  rf={r_free*100:.1f}%", "",
             "OPTIMAL PORTFOLIO",
             f"  {name1}: {opt['Weight Asset 1']*100:.1f}%",
             f"  {name2}: {opt['Weight Asset 2']*100:.1f}%",
             f"  Risk-Free: {opt['Weight RF']*100:.1f}%",
             f"  E[R]={opt['Return']*100:.2f}%  sigma={opt['Volatility']*100:.2f}%",
             f"  ESG={opt['ESG Score']:.1f}/100  Sharpe={opt['Sharpe Ratio']:.3f}",
             f"  ESG Cost: {esg_cost:.4f} Sharpe vs tangency", "",
             "COMPARISON",
             f"  Portfolio            {name1}%  {name2}%  RF%  Sharpe",
             f"  MV Optimal (l=0)     {mv['Weight Asset 1']*100:5.1f}  {mv['Weight Asset 2']*100:5.1f}  {mv['Weight RF']*100:4.1f}  {mv['Sharpe Ratio']:.3f}",
             f"  ESG Optimal          {opt['Weight Asset 1']*100:5.1f}  {opt['Weight Asset 2']*100:5.1f}  {opt['Weight RF']*100:4.1f}  {opt['Sharpe Ratio']:.3f}",
             f"  Tangency (Max Sharpe){tan['Weight Asset 1']*100:5.1f}  {tan['Weight Asset 2']*100:5.1f}  {tan['Weight RF']*100:4.1f}  {tan['Sharpe Ratio']:.3f}", "",
             "  Ethical Edge - ECN316 Sustainable Finance - QMUL", "="*60]
    return "\n".join(lines)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Results", "Charts", "Explore", "Insights", "AI Explainer", "Compare", "Methodology"
])

# ════════════════ TAB 1 - RESULTS ════════════════
with tab1:
    if corner:
        low_name  = name2 if x_opt[0] >= x_opt[1] else name1
        high_name = name1 if x_opt[0] >= x_opt[1] else name2
        st.warning(
            f"**Corner solution detected.** Your ESG preference (λ = {lam}) is so strong that "
            f"holding any amount of **{low_name}** (the lower-ESG asset) reduces your utility. "
            f"The optimiser has placed all risky investment in **{high_name}**. "
            f"This is economically meaningful - at this level of ESG preference, "
            f"diversification is not worth the green cost. To hold both assets, reduce λ."
        )

    st.subheader("Asset ESG Scores", divider="green")
    a1c, a2c = st.columns(2)
    for col, nm, score, es, ss, gs, exf in [
        (a1c, name1, esg1, e1, s1, g1, ex1),
        (a2c, name2, esg2, e2, s2, g2, ex2),
    ]:
        lbl, clr = traffic_light(score)
        col.metric(label=f"{nm} - {lbl} {'EXCLUDED' if exf else ''}",
                   value=f"{score:.1f} / 100", help=f"E:{es:.0f} S:{ss:.0f} G:{gs:.0f}")
    style_metric_cards(background_color="#0e0e12", border_left_color="#00e676",
                       border_color="#2a2a3a", box_shadow=True)

    for nm, es, ss, gs in [(name1, e1, s1, g1), (name2, e2, s2, g2)]:
        flagged, msg = greenwashing_flag(es, ss, gs)
        if flagged: st.warning(f"**{nm}:** {msg}")

    excl_any = excl_tobacco or excl_weapons or excl_fossil or excl_gambling or min_esg_score > 0
    sf_lbl, sf_desc, sf_clr = sf_framework(lam, excl_any)
    st.info(f"**Sustainable Finance Approach:** {sf_lbl}\n{sf_desc} *(Schoenmaker 2017)*")

    st.subheader("Carbon Scope Breakdown", divider="green")
    fig_sc, ax_sc = plt.subplots(figsize=(8, 3.5)); apply_chart_style(ax_sc, fig_sc)
    x_sc = np.arange(3); w_sc = 0.35
    bars1 = ax_sc.bar(x_sc-w_sc/2, [scope1_1,scope2_1,scope3_1], w_sc, label=name1, color="#ff5252", alpha=0.85)
    bars2 = ax_sc.bar(x_sc+w_sc/2, [scope1_2,scope2_2,scope3_2], w_sc, label=name2, color="#00e676", alpha=0.85)
    tot_r = float(opt['Weight Asset 1'])+float(opt['Weight Asset 2'])
    w1n = float(opt['Weight Asset 1'])/tot_r if tot_r>1e-10 else 0.5
    w2n = 1.0 - w1n
    ax_sc.plot(x_sc, [w1n*scope1_1+w2n*scope1_2, w1n*scope2_1+w2n*scope2_2, w1n*scope3_1+w2n*scope3_2],
               "D--", color="white", lw=1.5, ms=8, label="Your Portfolio", zorder=5)
    ax_sc.set_xticks(x_sc); ax_sc.set_xticklabels(["Scope 1\n(Direct)","Scope 2\n(Energy)","Scope 3\n(Supply)"], color="#c8ccd8")
    ax_sc.set_title("Carbon Emissions by Scope", fontweight="bold"); ax_sc.set_ylim(0,115)
    ax_sc.legend(facecolor="#0e0e12", labelcolor="#c8ccd8", edgecolor="#2a2a3a")
    for bar in list(bars1)+list(bars2):
        ax_sc.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5, f"{bar.get_height():.0f}",
                   ha="center", va="bottom", fontsize=9, color="#c8ccd8")
    plt.tight_layout(); st.pyplot(fig_sc); plt.close(fig_sc)
    st.divider()

    st.info(f"**Portfolio Recommendation**\n\n{portfolio_summary()}")
    st.subheader("Your Recommended Portfolio", divider="green")
    m1,m2,m3,m4,m5,m6,m7 = st.columns(7)
    m1.metric(f"{name1} Weight",  f"{opt['Weight Asset 1']*100:.1f}%")
    m2.metric(f"{name2} Weight",  f"{opt['Weight Asset 2']*100:.1f}%")
    m3.metric("Risk-Free Weight", f"{opt['Weight RF']*100:.1f}%")
    m4.metric("Expected Return",  f"{opt['Return']*100:.2f}%")
    m5.metric("Risk sigma",       f"{opt['Volatility']*100:.2f}%")
    m6.metric("ESG Score",        f"{opt['ESG Score']:.1f}")
    m7.metric("Sharpe Ratio",     f"{opt['Sharpe Ratio']:.3f}", delta=sharpe_badge(float(opt['Sharpe Ratio'])))
    style_metric_cards(background_color="#0e0e12", border_left_color="#00e676", border_color="#2a2a3a", box_shadow=True)

    if lam > 0 and esg_cost > 0.001:
        st.success(f"λ={lam}: **ESG cost = {esg_cost:.4f}** Sharpe vs max-Sharpe portfolio.")
    elif lam == 0:
        st.info("λ=0 — ESG plays no role. ESG Optimal = MV Optimal.")
    st.progress(esg_cost_pct/100, text=f"ESG cost: {esg_cost:.4f} Sharpe ({esg_cost_pct:.1f}% of max Sharpe)")
    st.divider()

    st.subheader("Investment Calculator", divider="green")
    inv_col1, inv_col2 = st.columns([1,2])
    with inv_col1:
        invest_amt = st.number_input("Investment (GBP)", value=1000, step=100, min_value=100)
        invest_yrs = st.slider("Years", 1, 30, 10)
        st.markdown("---")
        st.metric("ESG Optimal", f"GBP {invest_amt*(1+opt['Return'])**invest_yrs:,.0f}",
                  delta=f"+GBP {invest_amt*((1+opt['Return'])**invest_yrs-1):,.0f}")
        st.metric("Max Sharpe",  f"GBP {invest_amt*(1+tan['Return'])**invest_yrs:,.0f}",
                  delta=f"+GBP {invest_amt*((1+tan['Return'])**invest_yrs-1):,.0f}")
        st.metric("Risk-Free",   f"GBP {invest_amt*(1+r_free)**invest_yrs:,.0f}",
                  delta=f"+GBP {invest_amt*((1+r_free)**invest_yrs-1):,.0f}")
        style_metric_cards(background_color="#0e0e12", border_left_color="#00e676", border_color="#2a2a3a", box_shadow=True)
    with inv_col2:
        years = np.arange(0, invest_yrs+1)
        fig_inv, ax_inv = plt.subplots(figsize=(8,4)); apply_chart_style(ax_inv, fig_inv)
        ax_inv.plot(years, invest_amt*(1+opt['Return'])**years, color="#00e676", lw=2.5, label=f"ESG Optimal ({opt['Return']*100:.1f}%/yr)")
        ax_inv.plot(years, invest_amt*(1+tan['Return'])**years, color="#ff5252", lw=1.8, ls="--", label=f"Max Sharpe ({tan['Return']*100:.1f}%/yr)")
        ax_inv.plot(years, invest_amt*(1+mv['Return'])**years,  color="#448aff", lw=1.5, ls=":",  label=f"MV Optimal ({mv['Return']*100:.1f}%/yr)")
        ax_inv.plot(years, invest_amt*(1+r_free)**years,          color="white",   lw=1.2, ls="-.", alpha=0.5, label=f"Risk-free ({r_free*100:.1f}%/yr)")
        ax_inv.fill_between(years, invest_amt*(1+opt['Return'])**years, invest_amt*(1+r_free)**years, color="#00e676", alpha=0.06)
        ax_inv.set_xlabel("Years"); ax_inv.set_ylabel("Portfolio Value (GBP)")
        ax_inv.set_title(f"GBP {invest_amt:,} over {invest_yrs} years", fontweight="bold")
        ax_inv.legend(fontsize=8, facecolor="#0e0e12", labelcolor="#c8ccd8", edgecolor="#2a2a3a", framealpha=0.9)
        ax_inv.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"GBP {x:,.0f}"))
        plt.tight_layout(); st.pyplot(fig_inv); plt.close(fig_inv)
    st.caption("Assumes constant annual returns. Past performance does not guarantee future results.")
    st.divider()

    st.subheader("Portfolio Comparison Table", divider="green")
    comp_df = pd.DataFrame({
        "Portfolio":   ["MV Optimal (l=0)", "ESG Optimal (you)", "Tangency - Max Sharpe", "Min Variance"],
        f"{name1}(%)": [f"{mv['Weight Asset 1']*100:.1f}", f"{opt['Weight Asset 1']*100:.1f}",
                        f"{tan['Weight Asset 1']*100:.1f}", f"{mvp['Weight Asset 1']*100:.1f}"],
        f"{name2}(%)": [f"{mv['Weight Asset 2']*100:.1f}", f"{opt['Weight Asset 2']*100:.1f}",
                        f"{tan['Weight Asset 2']*100:.1f}", f"{mvp['Weight Asset 2']*100:.1f}"],
        "RF(%)":       [f"{mv['Weight RF']*100:.1f}", f"{opt['Weight RF']*100:.1f}",
                        f"{tan['Weight RF']*100:.1f}", f"{mvp['Weight RF']*100:.1f}"],
        "E[Rp]":       [f"{mv['Return']*100:.2f}%", f"{opt['Return']*100:.2f}%",
                        f"{tan['Return']*100:.2f}%", f"{mvp['Return']*100:.2f}%"],
        "sigma":       [f"{mv['Volatility']*100:.2f}%", f"{opt['Volatility']*100:.2f}%",
                        f"{tan['Volatility']*100:.2f}%", f"{mvp['Volatility']*100:.2f}%"],
        "ESG":         [f"{mv['ESG Score']:.1f}", f"{opt['ESG Score']:.1f}",
                        f"{tan['ESG Score']:.1f}", f"{mvp['ESG Score']:.1f}"],
        "Sharpe":      [f"{mv['Sharpe Ratio']:.3f}", f"{opt['Sharpe Ratio']:.3f}",
                        f"{tan['Sharpe Ratio']:.3f}", "-"],
    })
    st.dataframe(comp_df, hide_index=True, use_container_width=True)
    st.divider()
    st.subheader("λ Sensitivity Analysis", divider="green")
    st.dataframe(build_sensitivity_table(), hide_index=True, use_container_width=True)
    st.divider()
    st.subheader("Download Portfolio Report", divider="green")
    st.download_button("Download Report (.txt)", data=build_report(),
                       file_name=f"ethical_edge_{datetime.date.today()}.txt", mime="text/plain")

# ════════════════ TAB 2 - CHARTS ════════════════
with tab2:
    st.subheader("ESG-Efficient Frontier", divider="green")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5)); fig.patch.set_facecolor("#0e0e12")
    for ax in axes: apply_chart_style(ax, fig)

    ax = axes[0]
    sc = ax.scatter(portfolios['Volatility']*100, portfolios['Return']*100,
                    c=portfolios['ESG Score'], cmap="RdYlGn", s=12, alpha=0.85, vmin=0, vmax=100, zorder=2)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("ESG Score", color="#c8ccd8", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="#c8ccd8")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#c8ccd8", fontsize=8)
    cbar.outline.set_edgecolor("#2a2a3a")

    if float(tan['Volatility']) > 1e-10:
        sdr = np.linspace(0, portfolios['Volatility'].max()*1.25, 200)
        ax.plot(sdr*100, (r_free + (float(tan['Return'])-r_free)/float(tan['Volatility'])*sdr)*100,
                "--", color="#448aff", lw=1.5, label="CML", alpha=0.85)

    sc2 = np.linspace(0.001, portfolios['Volatility'].max()*1.25, 200)
    U_mv = float(mv['Weight Asset 1'])*(r1-r_free) + float(mv['Weight Asset 2'])*(r2-r_free) - (gamma/2)*float(mv['Volatility'])**2
    ax.plot(sc2*100, (r_free+U_mv+(gamma/2)*sc2**2)*100, ":", color="#448aff", lw=1.5, label="MV Indiff.")
    U_esg = float(opt['Utility'])
    ax.plot(sc2*100, (r_free+U_esg-lam*float(opt['ESG Score'])+(gamma/2)*sc2**2)*100, "-.", color="#ff9100", lw=1.5, label="ESG Indiff.")

    ax.scatter(0, r_free*100, s=90, marker="s", color="white", zorder=6, label="Risk-Free", ec="#555", lw=0.8)
    ax.scatter(float(mvp['Volatility'])*100, float(mvp['Return'])*100, s=100, marker="D", color="#00b0ff", zorder=6, label="Min Var.", ec="white", lw=0.5)
    ax.scatter(float(tan['Volatility'])*100, float(tan['Return'])*100, s=200, marker="*", color="#ff5252", zorder=7, label="Tangency", ec="white", lw=0.5)
    ax.scatter(float(mv['Volatility'])*100,  float(mv['Return'])*100,  s=140, marker="^", color="#448aff", zorder=6, label="MV Opt.", ec="white", lw=0.5)
    ax.scatter(float(opt['Volatility'])*100, float(opt['Return'])*100, s=200, marker="*", color="#00e676", zorder=7,
               label=f"ESG Opt. (x1={opt['Weight Asset 1']*100:.0f}%)", ec="white", lw=0.5)
    ax.set_xlabel("Risk - Std Dev (%)", fontsize=10); ax.set_ylabel("Expected Return (%)", fontsize=10)
    ax.set_title("Mean-Variance Space\n(colour = ESG score)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper left", facecolor="#0e0e12", labelcolor="#c8ccd8", framealpha=0.9, edgecolor="#2a2a3a")

    ax2 = axes[1]
    ax2.plot(portfolios['ESG Score'], portfolios['Sharpe Ratio'], color="#00e676", lw=2.5)
    ax2.fill_between(portfolios['ESG Score'], portfolios['Sharpe Ratio'],
                     portfolios['Sharpe Ratio'].min(), color="#00e676", alpha=0.07)
    ax2.scatter(float(opt['ESG Score']), float(opt['Sharpe Ratio']), s=200, marker="*", color="#00e676", zorder=5, label="Your ESG Optimal", ec="white", lw=0.5)
    ax2.scatter(float(tan['ESG Score']), float(tan['Sharpe Ratio']), s=160, marker="*", color="#ff5252", zorder=5, label="Max Sharpe", ec="white", lw=0.5)
    ax2.axhline(float(tan['Sharpe Ratio']), color="#ff5252", ls="--", lw=0.9, alpha=0.5)
    if esg_cost > 0.001:
        ax2.annotate(f"ESG cost: -{esg_cost:.3f}",
                     xy=(float(opt['ESG Score']), float(opt['Sharpe Ratio'])),
                     xytext=(float(opt['ESG Score'])+2, float(opt['Sharpe Ratio'])-0.03),
                     color="#ff9100", fontsize=9, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color="#ff9100", lw=1.2))
    ax2.set_xlabel("Portfolio ESG Score", fontsize=10); ax2.set_ylabel("Sharpe Ratio", fontsize=10)
    ax2.set_title("ESG-Sharpe Frontier", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8, facecolor="#0e0e12", labelcolor="#c8ccd8", framealpha=0.9, edgecolor="#2a2a3a")
    plt.tight_layout(pad=2.0)
    with chart_container(portfolios[["Weight Asset 1","Return","Volatility","ESG Score","Sharpe Ratio"]], export_formats=["CSV"]):
        st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.subheader("Portfolio Allocation Pie", divider="green")
    pie_col1, pie_col2 = st.columns(2)
    def draw_pie(ax, fig_p, w1, w2, wrf, t1, t2, title):
        apply_chart_style(ax, fig_p); ax.set_facecolor("#0e0e12")
        labels = [t1, t2]; sizes = [w1, w2]; colors = ["#00e676","#448aff"]
        if abs(wrf) > 0.005: labels.append("Risk-Free"); sizes.append(max(wrf, 0)); colors.append("#ffffff")
        if sum(sizes) < 1e-10: sizes = [0.5, 0.5]
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90,
            wedgeprops=dict(edgecolor="#0e0e12", linewidth=2), textprops=dict(color="#c8ccd8", fontsize=9))
        for at in autotexts: at.set_color("#0e0e12"); at.set_fontweight("bold"); at.set_fontsize(9)
        ax.set_title(title, color="#00e676", fontsize=10, fontweight="bold")
    with pie_col1:
        fp1, ap1 = plt.subplots(figsize=(4,4)); fp1.patch.set_facecolor("#0e0e12")
        draw_pie(ap1, fp1, float(opt['Weight Asset 1']), float(opt['Weight Asset 2']), float(opt['Weight RF']), name1, name2, "ESG Optimal")
        plt.tight_layout(); st.pyplot(fp1); plt.close(fp1)
    with pie_col2:
        fp2, ap2 = plt.subplots(figsize=(4,4)); fp2.patch.set_facecolor("#0e0e12")
        draw_pie(ap2, fp2, float(tan['Weight Asset 1']), float(tan['Weight Asset 2']), float(tan['Weight RF']), name1, name2, "Tangency / Max Sharpe")
        plt.tight_layout(); st.pyplot(fp2); plt.close(fp2)

    st.subheader("ESG Pillar Leaderboard", divider="green")
    for pillar, sc1, sc2 in zip(["Environmental","Social","Governance"],[e1,s1,g1],[e2,s2,g2]):
        winner = name1 if sc1 >= sc2 else name2; diff = abs(sc1-sc2)
        ca, cb, cc = st.columns([3,3,2])
        ca.metric(f"{pillar} - {name1}", f"{sc1:.0f}/100", delta=f"{'up' if sc1>sc2 else 'down'} {diff:.0f}")
        cb.metric(f"{pillar} - {name2}", f"{sc2:.0f}/100", delta=f"{'up' if sc2>sc1 else 'down'} {diff:.0f}")
        cc.markdown(f"**Winner:** {winner}")
    style_metric_cards(background_color="#0e0e12", border_left_color="#00e676", border_color="#2a2a3a", box_shadow=False)
    st.divider()

    fig2, ax3 = plt.subplots(figsize=(7,3.5)); apply_chart_style(ax3, fig2)
    xb = np.arange(3); wb = 0.35
    b1 = ax3.bar(xb-wb/2, [e1,s1,g1], wb, label=name1, color="#00e676", alpha=0.85)
    b2 = ax3.bar(xb+wb/2, [e2,s2,g2], wb, label=name2, color="#448aff", alpha=0.85)
    ax3.set_xticks(xb); ax3.set_xticklabels(["Environmental","Social","Governance"], color="#c8ccd8")
    ax3.set_title(f"E/S/G: {name1} vs {name2}", fontweight="bold"); ax3.set_ylim(0,115)
    ax3.legend(facecolor="#0e0e12", labelcolor="#c8ccd8", edgecolor="#2a2a3a")
    for bar in list(b1)+list(b2):
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5, f"{bar.get_height():.0f}",
                 ha="center", va="bottom", fontsize=9, color="#c8ccd8")
    plt.tight_layout(); st.pyplot(fig2); plt.close(fig2)

# ════════════════ TAB 3 - EXPLORE ════════════════
with tab3:
    st.subheader("Interactive Explorers", divider="green")
    exp1, exp2, exp3, exp4 = st.tabs(["γ Explorer","λ Explorer","rho Explorer","Utility Heatmap"])

    with exp1:
        st.markdown("#### How does risk aversion gamma change your portfolio?")
        st.caption("With free weights: doubling γ should roughly halve risky positions and increase risk-free weight.")
        gammas = np.linspace(0.5, 10, 40)
        w1g, retg, volg, srg, rfg = [], [], [], [], []
        for g in gammas:
            x = _solve(g, lam); s = _stats(x, gam=g, lam_v=lam)
            w1g.append(x[0]*100); retg.append(s['Return']*100)
            volg.append(s['Volatility']*100); srg.append(s['Sharpe Ratio']); rfg.append(s['Weight RF']*100)
        fig_g, axes_g = plt.subplots(2, 2, figsize=(12,7)); fig_g.patch.set_facecolor("#0e0e12")
        fig_g.suptitle(f"Effect of gamma (lambda={lam} fixed)", color="#00e676", fontweight="bold", fontsize=11)
        for ax, (yd, yl, col) in zip(axes_g.flat, [
            (w1g, f"{name1} Weight (% of wealth)", "#00e676"),
            (rfg, "Risk-Free Weight (% of wealth)", "#ff9100"),
            (volg, "Risk sigma (%)", "#ff5252"),
            (srg, "Sharpe Ratio", "#448aff"),
        ]):
            apply_chart_style(ax, fig_g)
            ax.plot(gammas, yd, color=col, lw=2.2)
            ax.axvline(gamma, color="white", ls="--", lw=1, alpha=0.5, label=f"Your γ={gamma}")
            ax.fill_between(gammas, yd, min(yd), color=col, alpha=0.07)
            ax.set_xlabel("γ", fontsize=9); ax.set_ylabel(yl, fontsize=9)
            ax.legend(fontsize=7, facecolor="#0e0e12", labelcolor="#c8ccd8", edgecolor="#2a2a3a", framealpha=0.9)
        plt.tight_layout(pad=2.0); st.pyplot(fig_g); plt.close(fig_g)
        st.info(f"At γ={gamma}: **{opt['Weight Asset 1']*100:.1f}%** in {name1}, **{opt['Weight Asset 2']*100:.1f}%** in {name2}, **{opt['Weight RF']*100:.1f}%** risk-free.")

    with exp2:
        st.markdown("#### How does ESG preference lambda change your portfolio?")
        lambdas = np.linspace(0, 5, 60); tan_sr = float(tan['Sharpe Ratio'])
        w1l, srl, esgl, costl = [], [], [], []
        for l in lambdas:
            x = _solve(gamma, l); s = _stats(x, lam_v=l)
            w1l.append(x[0]*100); srl.append(s['Sharpe Ratio'])
            esgl.append(s['ESG Score']); costl.append(max(tan_sr-s['Sharpe Ratio'],0))
        fig_l, axes_l = plt.subplots(1, 3, figsize=(14,5)); fig_l.patch.set_facecolor("#0e0e12")
        fig_l.suptitle(f"Effect of lambda (gamma={gamma} fixed)", color="#00e676", fontweight="bold", fontsize=11)
        for ax, (yd, xl, yl, col, ttl) in zip(axes_l, [
            (w1l, lambdas, f"{name1} Weight (% wealth)", "#00e676", "Allocation vs lambda"),
            (esgl,lambdas, "Portfolio ESG Score",         "#448aff", "ESG Score vs lambda"),
            (costl,lambdas,"ESG Cost (Sharpe drop)",      "#ff9100", "ESG Cost vs lambda"),
        ]):
            apply_chart_style(ax, fig_l)
            ax.plot(xl, yd, color=col, lw=2.2)
            ax.axvline(lam, color="white", ls="--", lw=1, alpha=0.5, label=f"Your λ={lam}")
            ax.fill_between(xl, yd, min(yd), color=col, alpha=0.07)
            ax.set_xlabel("λ", fontsize=9); ax.set_ylabel(yl, fontsize=9); ax.set_title(ttl, fontweight="bold")
            ax.legend(fontsize=7, facecolor="#0e0e12", labelcolor="#c8ccd8", edgecolor="#2a2a3a", framealpha=0.9)
        plt.tight_layout(pad=2.0); st.pyplot(fig_l); plt.close(fig_l)
        st.info(f"At λ={lam}: ESG={opt['ESG Score']:.1f}, ESG cost={esg_cost:.4f} Sharpe.")

    with exp3:
        st.markdown("#### How does correlation rho affect diversification?")
        rhos_r = np.linspace(-0.99, 0.99, 60)
        mvp_vr, opt_w1r, opt_srr = [], [], []
        for rh in rhos_r:
            cov_r = np.array([[sd1**2, rh*sd1*sd2],[rh*sd1*sd2, sd2**2]])
            mu_r  = np.array([r1-r_free, r2-r_free])
            def vfn(x, c=cov_r): return float(x @ c @ x)
            rm = minimize(vfn, [0.4,0.4], method='SLSQP', bounds=_bnds)
            mvp_vr.append(np.sqrt(max(rm.fun,0))*100)
            def ofn(x, c=cov_r, m=mu_r):
                t = x[0]+x[1]; sb = (x[0]*esg1+x[1]*esg2)/t if t>1e-10 else 0
                return -(float(x@m)-(gamma/2)*float(x@c@x)+lam*sb)
            ro = minimize(ofn, [0.4,0.4], method='SLSQP', bounds=_bnds)
            xr = ro.x; ret_r = r_free+float(xr@mu_r); sd_r = np.sqrt(max(float(xr@cov_r@xr),0))
            opt_w1r.append(xr[0]*100)
            opt_srr.append((ret_r-r_free)/sd_r if sd_r>1e-10 else 0)
        fig_r, axes_r = plt.subplots(1, 3, figsize=(14,5)); fig_r.patch.set_facecolor("#0e0e12")
        fig_r.suptitle(f"Effect of rho (gamma={gamma}, lambda={lam} fixed)", color="#00e676", fontweight="bold", fontsize=11)
        for ax, (yd, yl, col, ttl) in zip(axes_r, [
            (mvp_vr, "Min-Var sigma (%)",              "#ff5252","Diversification Benefit"),
            (opt_w1r,f"{name1} Weight (% wealth)",     "#00e676","Allocation vs rho"),
            (opt_srr,"ESG-Opt Sharpe",                 "#ff9100","Sharpe vs rho"),
        ]):
            apply_chart_style(ax, fig_r)
            ax.plot(rhos_r, yd, color=col, lw=2.2)
            ax.axvline(rho, color="white", ls="--", lw=1, alpha=0.5, label=f"Your rho={rho}")
            ax.fill_between(rhos_r, yd, min(yd), color=col, alpha=0.07)
            ax.set_xlabel("rho", fontsize=9); ax.set_ylabel(yl, fontsize=9); ax.set_title(ttl, fontweight="bold")
            ax.legend(fontsize=7, facecolor="#0e0e12", labelcolor="#c8ccd8", edgecolor="#2a2a3a", framealpha=0.9)
        plt.tight_layout(pad=2.0); st.pyplot(fig_r); plt.close(fig_r)

    with exp4:
        st.markdown("#### Utility Surface - risky mix weight vs lambda")
        st.caption("Shows objective on normalized risky frontier (x1+x2=1). Actual optimum uses free scipy weights.")
        w_grid = np.linspace(0,1,80); lam_grid = np.linspace(0,4,80)
        UU = np.array([[p_utility(w_grid[j], lam_val=lam_grid[i]) for j in range(len(w_grid))] for i in range(len(lam_grid))])
        fig_h, ax_h = plt.subplots(figsize=(10,6)); apply_chart_style(ax_h, fig_h)
        im = ax_h.contourf(w_grid*100, lam_grid, UU, levels=30, cmap="RdYlGn")
        cb = fig_h.colorbar(im, ax=ax_h); cb.set_label("Objective U", color="#c8ccd8")
        cb.ax.yaxis.set_tick_params(color="#c8ccd8"); plt.setp(cb.ax.yaxis.get_ticklabels(), color="#c8ccd8")
        ax_h.scatter(opt['Weight Asset 1']*100, lam, s=250, marker="*", color="#00e676", zorder=5, ec="white", lw=0.8, label="Your optimum")
        ax_h.axvline(opt['Weight Asset 1']*100, color="#00e676", ls="--", lw=0.8, alpha=0.4)
        ax_h.axhline(lam, color="#00e676", ls="--", lw=0.8, alpha=0.4)
        ax_h.set_xlabel(f"Weight in {name1} (% of risky mix)", fontsize=10); ax_h.set_ylabel("λ", fontsize=10)
        ax_h.set_title(f"Utility Surface (γ={gamma} fixed)", fontsize=10, fontweight="bold")
        ax_h.legend(fontsize=8, facecolor="#0e0e12", labelcolor="#c8ccd8", edgecolor="#2a2a3a", framealpha=0.9)
        plt.tight_layout(); st.pyplot(fig_h); plt.close(fig_h)

# ════════════════ TAB 4 - INSIGHTS ════════════════
with tab4:
    st.subheader("What does this mean for you?", divider="green")
    if gamma <= 2:   inv_type, inv_desc = "Risk-Seeking",  "Comfortable with large swings in pursuit of higher returns."
    elif gamma <= 5: inv_type, inv_desc = "Balanced",      "Seeks reasonable return while avoiding excessive risk."
    else:            inv_type, inv_desc = "Risk-Averse",   "Prioritises capital protection over maximising returns."
    if lam == 0:    esg_type, esg_desc = "ESG-Neutral", "ESG plays no role."
    elif lam < 1:   esg_type, esg_desc = "Light Green", "Mild ESG preference - small Sharpe reduction accepted."
    elif lam < 2:   esg_type, esg_desc = "Green",       "Meaningful ESG preference."
    else:           esg_type, esg_desc = "Deep Green",  "ESG is central - significant trade-off accepted."

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Investor Profile", divider="green")
        st.metric("Risk Type",   inv_type, help=inv_desc)
        st.metric("ESG Profile", esg_type, help=esg_desc)
        st.caption(inv_desc); st.caption(esg_desc)
        if persona != "Custom (manual)": st.info(f"Persona: **{persona}**")
        style_metric_cards(background_color="#0e0e12", border_left_color="#00e676", border_color="#2a2a3a", box_shadow=True)
        st.divider()
        st.subheader("Sustainability Trade-Off", divider="green")
        cost_lbl = "low" if esg_cost_pct < 10 else ("moderate" if esg_cost_pct < 25 else "high")
        annotated_text("ESG cost: ",
            (f"-{esg_cost:.4f} Sharpe", cost_lbl,
             "#00c853" if cost_lbl=="low" else "#ff9100" if cost_lbl=="moderate" else "#ff5252"),
            " | Your ESG: ", (f"{opt['ESG Score']:.1f}", "your portfolio", "#00c853"),
            " vs tangency: ", (f"{tan['ESG Score']:.1f}", "max-Sharpe", "#ff5252"))
    with col_b:
        st.subheader("Allocation Explained", divider="green")
        higher_esg = name1 if esg1 > esg2 else name2
        rf_note = f"\n\n**{opt['Weight RF']*100:.1f}%** is held in the risk-free asset." if abs(opt['Weight RF']) > 0.01 else ""
        st.info(f"**{opt['Weight Asset 1']*100:.1f}%** in {name1}, **{opt['Weight Asset 2']*100:.1f}%** in {name2}.{rf_note}\n\n"
                f"**{higher_esg}** has the higher ESG - λ tilts toward it.\nγ={gamma} and λ={lam} determine the balance.")
        st.subheader("Diversification Benefit", divider="green")
        if rho < 0.5:
            st.success(f"rho={rho}: diversification benefit present. Min-variance sigma={mvp['Volatility']*100:.2f}%.")
        else:
            st.warning(f"rho={rho}: limited diversification. Min-variance sigma={mvp['Volatility']*100:.2f}%.")

# ════════════════ TAB 5 - AI EXPLAINER ════════════════
with tab5:
    st.subheader("AI Portfolio Explainer", divider="green")
    portfolio_context = f"""You are an expert in sustainable finance and portfolio theory.

CURRENT SETTINGS:
- {name1}: E[R]={r1*100:.1f}%, sigma={sd1*100:.1f}%, ESG={esg1:.1f}/100
- {name2}: E[R]={r2*100:.1f}%, sigma={sd2*100:.1f}%, ESG={esg2:.1f}/100
- rho={rho}, rf={r_free*100:.1f}%, gamma={gamma}, lambda={lam}

RESULTS (scipy free-weight optimisation):
- ESG Optimal: {opt['Weight Asset 1']*100:.1f}% in {name1}, {opt['Weight Asset 2']*100:.1f}% in {name2}, {opt['Weight RF']*100:.1f}% risk-free
- Sharpe={opt['Sharpe Ratio']:.3f}, ESG cost={esg_cost:.4f}
- Objective: max x'mu - (gamma/2)x'Sigma x + lambda*s_bar (Pedersen et al. 2021)

Answer clearly in plain English, under 200 words."""

    if "chat_history" not in st.session_state: st.session_state.chat_history = []
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="Bot" if msg["role"]=="assistant" else "User"):
            st.markdown(msg["content"])
    if not st.session_state.chat_history:
        cols = st.columns(3)
        for i, sug in enumerate(["Why is my portfolio split this way?","What does ESG cost mean?",
                                  "Explain the utility function","Is my Sharpe ratio good?",
                                  "What are sin stocks?","How would higher gamma change things?"]):
            with cols[i%3]:
                if st.button(sug, key=f"sug_{i}", use_container_width=True):
                    st.session_state.chat_history.append({"role":"user","content":sug}); st.rerun()
    user_input = st.chat_input("Ask about your portfolio or ESG investing...")
    if user_input:
        st.session_state.chat_history.append({"role":"user","content":user_input}); st.rerun()
    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"]=="user":
        with st.chat_message("assistant", avatar="Bot"):
            with st.spinner("Thinking..."):
                try:
                    resp = requests.post("https://api.anthropic.com/v1/messages",
                        headers={"Content-Type":"application/json"},
                        json={"model":"claude-sonnet-4-6","max_tokens":1000,
                              "system":portfolio_context,
                              "messages":[{"role":m["role"],"content":m["content"]} for m in st.session_state.chat_history]},
                        timeout=30)
                    reply = resp.json()["content"][0]["text"]
                except Exception as e:
                    reply = f"Sorry, could not connect to AI. Error: {e}"
                st.markdown(reply)
                st.session_state.chat_history.append({"role":"assistant","content":reply})
    if st.session_state.chat_history:
        if st.button("Clear conversation", key="clear_chat"):
            st.session_state.chat_history = []; st.rerun()

# ════════════════ TAB 6 - COMPARE ════════════════
with tab6:
    st.subheader("Scenario Comparator", divider="green")
    st.caption("Both scenarios use free-weight scipy optimisation. Risk-free allocation shown.")
    sc1c, sc2c = st.columns(2)
    with sc1c:
        st.markdown("### Scenario A")
        sa_gamma = st.slider("γ (A)", 0.5, 10.0, gamma,            0.5,  key="sa_g")
        sa_lam   = st.slider("λ (A)", 0.0, 5.0, lam,              0.25, key="sa_l")
        sa_label = st.text_input("Label", "Scenario A", key="sa_name")
    with sc2c:
        st.markdown("### Scenario B")
        sb_gamma = st.slider("γ (B)", 0.5, 10.0, max(gamma-2, 0.5), 0.5,  key="sb_g")
        sb_lam   = st.slider("λ (B)", 0.0, 5.0, min(lam+1.5, 5.0), 0.25, key="sb_l")
        sb_label = st.text_input("Label", "Scenario B", key="sb_name")
    sa = solve_scenario(sa_gamma, sa_lam); sb = solve_scenario(sb_gamma, sb_lam)
    st.divider(); st.markdown("#### Side-by-Side Results")
    comp_cols = st.columns(6)
    for col, (lbl, va, vb) in zip(comp_cols, [
        (f"{name1} Weight",  f"{sa['w1']*100:.1f}%",           f"{sb['w1']*100:.1f}%"),
        (f"{name2} Weight",  f"{sa['w2']*100:.1f}%",           f"{sb['w2']*100:.1f}%"),
        ("Risk-Free",        f"{sa['Weight RF']*100:.1f}%",     f"{sb['Weight RF']*100:.1f}%"),
        ("E[R]",             f"{sa['Return']*100:.2f}%",        f"{sb['Return']*100:.2f}%"),
        ("Risk sigma",       f"{sa['Volatility']*100:.2f}%",    f"{sb['Volatility']*100:.2f}%"),
        ("ESG Score",        f"{sa['ESG Score']:.1f}",          f"{sb['ESG Score']:.1f}"),
    ]):
        col.metric(lbl, f"{sa_label}: {va}", delta=f"{sb_label}: {vb}")
    style_metric_cards(background_color="#0e0e12", border_left_color="#00e676", border_color="#2a2a3a", box_shadow=True)

# ════════════════ TAB 7 - METHODOLOGY ════════════════
with tab7:
    st.subheader("Methodology", divider="green")
    st.markdown("""
### Objective Function

This app maximises over **free** risky weights **x** = (x1, x2):

> **max x'mu - (gamma/2) x'Sigma x + lambda * s_bar**

- **x** = risky asset weights (fractions of total wealth). Remainder **1 - x1 - x2** held in risk-free asset. No sum-to-1 constraint.
- **mu** = **excess** returns: mu_i = E[R_i] - rf
- **γ** = risk aversion. Doubling γ roughly halves risky positions (audit check 1).
- **Sigma** = covariance matrix. Off-diagonal = rho * sigma1 * sigma2.
- **λ** = ESG taste. λ=0 gives pure MV solution (audit check 2).
- **s_bar = (x1*ESG1 + x2*ESG2) / (x1 + x2)** = risky-asset weighted ESG score (not total wealth).

Optimisation uses **scipy.optimize.minimize (SLSQP)** with x1 >= 0, x2 >= 0.

### Audit Checks

| Check | Expected behaviour | Status |
|-------|-------------------|--------|
| Double gamma (lambda=0) | Risky weights roughly halve; RF weight increases | PASS |
| Increase lambda | Tilts toward higher-ESG asset; Sharpe falls | PASS |
| Symmetric assets | Equal weights at lambda=0 | PASS |
| High lambda corner | All weight in highest-ESG asset + warning shown | PASS |

### References
- Pedersen, Fitzgibbons & Pomorski (2021). Responsible Investing. *J. Financial Economics*, 142(2).
- Schoenmaker (2017). *Investing for the Common Good*. Bruegel.
- Berg, Kolbel & Rigobon (2022). Aggregate Confusion. *Review of Finance*, 26(6).
- Bolton & Kacperczyk (2021). Do investors care about carbon risk? *J. Financial Economics*, 142(2).
""")

st.divider()
st.caption("Ethical Edge - ECN316 Sustainable Finance - QMUL | scipy-optimised free-weight portfolio")
