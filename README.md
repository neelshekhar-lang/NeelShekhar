# 🌿 GreenPort — ESG Portfolio Optimiser

**ECN316 Sustainable Finance | Group Project**

A Streamlit web app that helps retail investors build personalised portfolios balancing financial returns and ESG (sustainability) preferences.

-----

## Theory (Lecture 6)

**Utility function:**

```
U = E[Rp] − (γ/2)·σ²p + λ·s̄
```

- **γ** — risk aversion parameter
- **λ** — ESG preference intensity
- **s̄** — weighted average ESG score of the portfolio

The app constructs the **ESG-efficient frontier** (max Sharpe ratio at each ESG constraint level) and finds the optimal portfolio via the utility function.

-----

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

-----

## Deploying to Streamlit Cloud (Phase 2)

1. Create a free GitHub repository
1. Upload `app.py` and `requirements.txt`
1. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
1. Select your repo, branch: `main`, file: `app.py`
1. Click **Deploy**
1. Generate a QR code from your live URL at [qrtrac.com](https://qrtrac.com/free-qr-code-generator/)

-----

## App Features

|Feature      |Details                                                         |
|-------------|----------------------------------------------------------------|
|**Inputs**   |Sliders for γ, λ, rf, ρ, E[R], σ, ESG score per asset           |
|**Outputs**  |Portfolio weights table, E[R], σ, Sharpe ratio, ESG score       |
|**Chart 1**  |Mean-variance frontier with optimal point marked                |
|**Chart 2**  |ESG-efficient frontier (Sharpe vs ESG score)                    |
|**Trade-off**|Annotated ESG cost (drop in Sharpe from pursuing sustainability)|

-----

## Files

```
app.py              ← Main Streamlit app (Phase 1 + 2)
requirements.txt    ← Python dependencies
README.md           ← This file
```
