# Used-Car Residual Value Model

A production-grade probabilistic model for estimating the residual value (RV) of used vehicles over time. Built for credit risk applications — collateral valuation, LGD estimation, and LTV monitoring under IFRS 9 and Basel frameworks.

---

## Why This Problem Is Hard

Residual value estimation looks deceptively simple: given a car bought at price $P$, predict its market value at future time $t$. In practice, three compounding difficulties make naive regression inadequate.

**Non-linear time decay.** Vehicle depreciation is not linear. A new car loses roughly 15–20% of its value in the first year, then the rate slows. Any model that does not account for this convex decay shape will systematically over-estimate value for young vehicles and under-estimate for older ones.

**Sparse, heterogeneous segments.** A portfolio contains hundreds of model–fuel–condition combinations. Some combinations have thousands of historical transactions; others have fewer than thirty. A model that treats each segment independently will overfit sparse ones. A model that pools everything loses the real differences between, say, a diesel pickup and a luxury sedan.

**Uncertainty quantification is not optional.** In credit risk, a point estimate of collateral value is insufficient. A lender needs to know the *worst credible case* — not just the expected sale price, but the price at which collateral will cover the loan under stress. This requires a full predictive distribution, not a single number.

The two-stage architecture in this repository is designed to solve all three problems simultaneously.

---

## Architecture Overview

```
Stage 1 — Hierarchical Bayesian Weibull
          Learn the depreciation curve shape per segment
          with partial pooling across brand tiers
                    ↓
          theoretical_rvr  (posterior mean)
          curve_uncertainty (posterior p90 − p10)
          rvr_residual      (actual − theoretical)
                    ↓
Stage 2 — NGBoost
          Learn what the curve cannot explain
          (condition, color, market timing, km usage)
          Output: residual distribution → p10 / p50 / p90
                    ↓
          final_rvr = theoretical_rvr + residual_quantile
          final_price = buy_price × final_rvr
```

The key insight is that depreciation has two separable components. The first is a *structural* component — every vehicle in a given segment follows a similar time-decay shape governed by its age. The second is an *idiosyncratic* component — the deviation of an individual vehicle from that curve, driven by its specific condition, color, usage history, and the market environment at the time of sale. Mixing these two components into a single model makes both harder to estimate well. Separating them lets each stage focus on what it can explain best.

---

## Stage 1 — Hierarchical Bayesian Weibull

### The Weibull Depreciation Curve

The Residual Value Ratio (RVR) is defined as:

$$\text{RVR}(t) = \frac{\text{sold\_price}}{\text{buy\_price}} \in (0, 1)$$

The structural depreciation is modelled as a Weibull survival function:

$$\text{RVR}(t) = \exp\!\left(-\left(\frac{t}{\lambda}\right)^k\right)$$

where $t$ is vehicle age in months, $\lambda$ is the scale parameter (controls how quickly value decays), and $k$ is the shape parameter (controls the acceleration of decay).

This form has two important properties. First, at $t = 0$, RVR = 1 exactly — a vehicle is worth its full purchase price the moment it is bought. Second, the shape parameter $k$ captures the convexity of depreciation: when $k > 1$, depreciation accelerates over time (fast early loss slowing down), which matches empirical used-car data. When $k = 1$, the model reduces to simple exponential decay.

### Why Hierarchical?

Fitting independent Weibull curves per segment causes two failure modes. Segments with abundant data overfit to noise. Segments with sparse data produce unreliable estimates — or fail to converge entirely. The solution is *partial pooling* via a hierarchical prior structure.

```
brand_tier level
  μ_log_λ_brand ~ Normal(log(48), 0.5)   ← hyperprior: λ centred at ~4 year half-life
  μ_log_k_brand ~ Normal(log(1.2), 0.3)  ← hyperprior: slight acceleration

        ↓ partial pooling

segment level (brand_tier × fuel)
  log_λ_seg ~ Normal(μ_log_λ_brand[tier], σ_λ)
  log_k_seg ~ Normal(μ_log_k_brand[tier], σ_k)
```

A sparse segment (e.g. luxury × EV with 25 observations) borrows statistical strength from its parent brand tier. A dense segment (e.g. pickup × Diesel with 3,000 observations) is largely self-determined. The degree of pooling is learned automatically from the data through the variance parameters $\sigma_\lambda$ and $\sigma_k$.

### Why `brand_tier × fuel` as the Segment Definition

The segment is defined as `brand_tier × fuel` rather than individual car models for two reasons. First, the depreciation *curve shape* (captured by $\lambda$ and $k$) is primarily driven by vehicle class and powertrain type, not by specific model names. Second, using car models directly produces 20+ segments many of which are sparse, while `brand_tier × fuel` yields approximately six well-populated groups with meaningfully different depreciation profiles.

Features like `condition`, `color`, `km`, and `car_model` are not included in Stage 1 because they affect the *level* of an individual observation relative to the curve, not the *shape* of the curve itself. These are handled by Stage 2.

### Beta Likelihood

The observed RVR is bounded in $(0, 1)$, which makes a Gaussian likelihood inappropriate. The model uses a Beta likelihood:

$$\text{RVR}_{\text{obs}} \sim \text{Beta}(\mu \cdot \kappa,\ (1 - \mu) \cdot \kappa)$$

where $\mu$ is the theoretical RVR from the Weibull curve and $\kappa$ is a precision parameter that controls how tightly observations cluster around the curve.

### Stage 1 Output

After sampling, the posterior over $(\lambda, k)$ is used to compute four features per observation that are passed to Stage 2:

| Feature | Description |
|---|---|
| `theoretical_rvr` | Posterior mean of RVR$(t)$ |
| `theoretical_rvr_std` | Posterior standard deviation — how uncertain the curve is |
| `curve_uncertainty` | Posterior p90 − p10 width — credible interval of the curve |
| `rvr_residual` | `actual_rvr` − `theoretical_rvr` — the Stage 2 target |

The `curve_uncertainty` feature is particularly important: it tells Stage 2 how much of the total prediction uncertainty has already been accounted for by Stage 1, allowing the NGBoost model to calibrate its own uncertainty accordingly.

---

## Stage 2 — NGBoost on Residuals

### What NGBoost Does Differently

Standard gradient boosting (LightGBM, XGBoost) predicts a single number. For uncertainty quantification, the common workaround is to train three separate quantile regression models — one each for p10, p50, p90. This approach has two drawbacks: the three models are trained independently and may produce crossing quantiles (p10 > p50 for some observations), and the resulting intervals are not derived from a coherent probability distribution.

NGBoost trains a single model whose output is the parameters of a probability distribution rather than a point estimate. The model learns $\mu$ and $\sigma$ of a chosen distribution family (Normal, LogNormal, or Laplace) for each observation. Any quantile can then be computed analytically from the distribution.

### Natural Gradient

The "natural" in NGBoost refers to the use of the *natural gradient* — the ordinary gradient pre-multiplied by the inverse Fisher Information Matrix — when updating the boosting ensemble. For distribution parameter estimation, the Fisher Information captures the curvature of the likelihood surface with respect to those parameters. This makes the update steps scale-invariant across parameters with very different magnitudes (e.g. a mean of 0.05 and a standard deviation of 0.02), leading to better-calibrated distributions and faster convergence than ordinary gradient descent.

### Distribution Selection

Before training, the distribution of `rvr_residual` is diagnosed automatically:

```
Shapiro-Wilk p > 0.05          → Normal
Skewness > 0.5 (right-skewed)  → LogNormal
Skewness < -0.5 (left-skewed)  → Laplace
Excess kurtosis > 3            → Laplace
```

This matters because residuals from used-car transactions are rarely Gaussian. Salvage vehicles and flood-damaged cars create a heavy left tail. Rare models in strong demand create a right tail. Using the wrong distribution family produces mis-calibrated prediction intervals even if the point estimate is accurate.

### Features

Stage 2 receives all Stage 1 outputs as features alongside the vehicle-level characteristics that Stage 1 cannot model:

```
From Stage 1:
  theoretical_rvr, theoretical_rvr_std,
  theoretical_rvr_p10, theoretical_rvr_p90, curve_uncertainty

Vehicle features:
  age_months, age_band, km_per_month, model_age,
  engine, door, down_percent

Categorical (encoded):
  car_model       → LeaveOneOut encoding (prevents target leakage)
  color, condition, wheel, transmission, fuel, age_band
                  → Ordinal encoding

Market timing:
  sell_year, sell_month
```

### Quantile Crossing Fix

Even with a coherent distribution, numerical sampling can occasionally produce crossing quantiles in edge cases. All output quantiles are post-processed with isotonic regression to enforce strict monotonicity: p10 < p50 < p90 for every observation.

---

## Outputs

### Per-Vehicle Depreciation Schedule

For any vehicle in the portfolio, the model generates a full depreciation schedule across a user-defined time grid (yearly, quarterly, or monthly):

```
age_months  year   rvr_q10   rvr_q50   rvr_q90   price_q10   price_q50   price_q90   depreciation_pct
        12  1.00    0.7821    0.8312    0.8734      782,100     831,200     873,400             16.9
        24  2.00    0.6543    0.7104    0.7631      654,300     710,400     763,100             29.0
        36  3.00    0.5412    0.5983    0.6541      541,200     598,300     654,100             40.2
        48  4.00    0.4401    0.4987    0.5512      440,100     498,700     551,200             50.1
        60  5.00    0.3612    0.4143    0.4701      361,200     414,300     470,100             58.6
```

The `price_q10` column is the stress-case collateral value — the floor that should be used for LGD and LTV calculations under adverse scenarios.

### Portfolio Collateral Aggregation

Across all active contracts, the model aggregates collateral at each time horizon:

```
year   collateral_stress   collateral_base   collateral_upside   ltv_base   ltv_stress
 1.0      1,842,000,000     2,103,000,000       2,341,000,000       0.714       0.816
 2.0      1,534,000,000     1,791,000,000       2,012,000,000       0.837       0.980
 3.0      1,241,000,000     1,482,000,000       1,712,000,000       1.009       1.212
```

When `ltv_stress > 1.0`, the portfolio is underwater on a stress basis at that horizon — the collateral cannot cover the original outstanding balance under adverse market conditions.

### Collateral Grading

Each vehicle receives a grade based on its stress-case RVR at a 36-month horizon:

| Grade | p10 RVR at 36m | Interpretation |
|---|---|---|
| A | ≥ 0.65 | Very strong — minimal haircut required |
| B | ≥ 0.50 | Adequate — standard haircut applies |
| C | ≥ 0.35 | Moderate — enhanced monitoring required |
| D | < 0.35 | Weak — additional collateral or haircut required |

The haircut percentage is defined as `1 − p10_RVR`, representing the minimum buffer needed to cover the gap between expected and stress collateral value.

---

## Backtesting

Model performance is evaluated across five segment dimensions:

| Dimension | Why It Matters |
|---|---|
| `car_model` | Different models depreciate at structurally different rates |
| `condition` | Fair vs Poor vs Salvage have very different RVR distributions |
| `fuel` | Diesel vs Petrol demand shifts over time affect relative values |
| `age_band` | Model accuracy may degrade at extremes (very new or very old) |
| `brand_tier` | Luxury vs standard vs pickup have different curve shapes |

Three flags trigger a recalibration recommendation:

```
MAE      > 5%  → Stage 2 recalibration for this segment
|bias|   > 3%  → Stage 1 Weibull curve shape review
coverage < 70% → NGBoost distribution choice review
```

---

## Notebooks

| Notebook | Purpose |
|---|---|
| `NB01_EDA_DataPrep` | Load data, engineer features, exploratory analysis |
| `NB02_Stage1_BayesianWeibull` | Fit hierarchical Weibull, extract posterior features |
| `NB03_Stage2_NGBoost_Schedule` | Train NGBoost, generate schedules, portfolio analysis, collateral grading |
| `NB04_SHAP_Analysis` | Feature importance, beeswarm, dependence, waterfall plots |
| `NB05_Backtesting_by_Segment` | Segment-level performance evaluation and flag report |

---

## Dependencies

```bash
pip install pymc arviz ngboost optuna shap category_encoders \
            scikit-learn lightgbm pandas numpy matplotlib seaborn pyarrow
```

---

## Data

The model was developed on a Thai used-car transaction dataset (`car_data.csv`) containing 26,157 transactions from 2017–2020 across vehicle models including C-Cab, Yaris, Vios, Altis, Fortuner, and Camry, with features covering purchase price, sale price, age, mileage, condition, fuel type, transmission, and body type.

---

## Credit Risk Integration

| Model Output | Credit Application |
|---|---|
| `price_q50` at sold date | LGD base case collateral value |
| `price_q10` at sold date | LGD stress case / collateral haircut |
| Full schedule p10 | LTV monitoring trigger at each reporting date |
| Portfolio `ltv_stress` | ECL Stage 2 / Stage 3 allocation signal |
| Grade A–D | Collateral quality classification for risk-weighted assets |
