
import numpy as np
import pandas as pd
import warnings

from scipy.stats import shapiro, kstest, skew, kurtosis

warnings.simplefilter(action = "ignore", category = pd.errors.PerformanceWarning)

# Clean data
def clean_and_engineer(
    df: pd.DataFrame
) -> pd.DataFrame:

    """
    Clean raw car data and engineer features for RV Modelling.

    Description:
        Performs date parsing, RVR target construction, age calculation,
        imputation of missing values, outlier clipping, and segment
        construction. Produces the modelling-ready DataFrame.

    Args:
        df (pd.DataFrame) : Raw DataFrame from load_car_data().

    Returns:
        pd.DataFrame: Cleaned DataFrame with additional engineered columns:
            {
                "rvr"            : float  — sold_price / car_price,
                "age_months"     : float  — months from contract to sold date,
                "age_band"       : str    — 5 discrete age buckets,
                "km_per_month"   : float  — odometer / age_months,
                "sell_year"      : int    — year of sold_date,
                "sell_month"     : int    — month of sold_date,
                "model_age"      : int    — sold year minus model_year,
                "brand_tier"     : str    — luxury / standard / pickup,
                "segment"        : str    — brand_tier + "_" + fuel
            }
    """

    df = df.copy()

    # Parse dates
    df["contract_date"] = pd.to_datetime(df["contract_date"])
    df["sold_date"] = pd.to_datetime(df["sold_date"])

    # Target: Residual Value Ratio
    df["rvr"] = df["sold_price"] / df["car_price"]

    # Age in months
    df["age_months"] = (
        (
            (df["sold_date"].dt.year - df["contract_date"].dt.year) * 12
            + (df["sold_date"].dt.month - df["contract_date"].dt.month)
            - (df["sold_date"].dt.day < df["contract_date"].dt.day).astype(int)
        )
        .clip(lower = 0)
    )

    # Age band
    df["age_band"] = pd.cut(
        df["age_months"],
        bins = [0, 12, 24, 36, 48, np.inf],
        labels = ["0-12m", "13-24m", "25-36m", "37-48m", "48m+"]
    )

    # km: clean and impute
    df["km"] = pd.to_numeric(df["km"], errors = "coerce")
    km_median = df.groupby("car_model")["km"].transform("median")
    df["km"] = df["km"].fillna(km_median).fillna(df["km"].median())

    # km per month
    df["km_per_month"] = (df["km"] / df["age_months"].clip(1)).round(1)

    # Sell date features
    df["sell_year"] = df["sold_date"].dt.year
    df["sell_month"] = df["sold_date"].dt.month

    # Model age
    df["model_age"] = df["sell_year"] - df["model_year"]

    # Color imputation: most common per car_model
    df["color"] = df["color"].fillna("Unknown")

    # Brand tier mapping
    luxury_models = ["Camry", "Fortuner"]
    pickup_models = ["C-Cab", "B-Cab", "D-Cab"]

    def assign_tier(model: str) -> str:
        if model in luxury_models:
            return "luxury"
        elif model in pickup_models:
            return "pickup"
        else:
            return "standard"

    df["brand_tier"] = df["car_model"].apply(assign_tier)

    # Segment
    df["segment"] = df["brand_tier"] + "_" + df["fuel"]

    # Clip extreme RVR (outlier: salvage or data error)
    df = df[(df["rvr"] >= 0.05) & (df["rvr"] <= 1.05)].copy()

    # Clip extreme age
    df = df[df["age_months"] <= 120].copy()

    return df.reset_index(drop = True)

# EAD Summary
def print_data_summary(
    df: pd.DataFrame
) -> dict:

    """
    Print and return key data quality and distribution statistics.

    Description:
        Computes shape, null rates, RVR statistics, skewness, kurtosis,
        and Shapiro-Wilk normality test on RVR to guide Stage 2 dist choice.

    Args:
        df (pd.DataFrame) : Cleaned DataFrame.

    Returns:
        dict: {
            "n_rows"       : int   — number of rows after cleaning,
            "rvr_mean"     : float — mean RVR,
            "rvr_std"      : float — std of RVR,
            "skewness"     : float — skewness of RVR,
            "excess_kurt"  : float — excess kurtosis of RVR,
            "p_shapiro"    : float — Shapiro-Wilk p-value (sample),
            "recommended_dist" : str — suggested NGBoost distribution
        }
    """

    rvr = df["rvr"]
    sample = rvr.sample(min(500, len(rvr)), random_state = 42)
    _, p_sw = shapiro(sample)
    sk = float(skew(rvr))
    kt = float(kurtosis(rvr))

    if p_sw > 0.05:
        dist = "Normal"
    elif abs(sk) > 0.5 and sk > 0:
        dist = "LogNormal"
    elif abs(sk) > 0.5 and sk < 0:
        dist = "Laplace"
    elif kt > 3:
        dist = "Laplace"
    else:
        dist = "LogNormal"

    result = {
        "n_rows": int(len(df)),
        "rvr_mean": float(rvr.mean()),
        "rvr_std": float(rvr.std()),
        "skewness": round(sk, 4),
        "excess_kurt": round(kt, 4),
        "p_shapiro": round(p_sw, 4),
        "recommended_dist": dist
    }
    l = 30
    print("=" * 60)
    print("Data summary")
    print(f"{"Rows after cleaning":<{l}}: {result['n_rows']:,}")
    print(f"{"RVR Mean":<{l}}: {result['rvr_mean']:.4f}")
    print(f"{"RVR STD":<{l}}: {result['rvr_std']:.4f}")
    print(f"{"Skewness":<{l}}: {result['skewness']}")
    print(f"{"Excess kurtosis":<{l}}: {result['excess_kurt']}")
    print(f"{"Shapiro-Wilk":<{l}}: {result['p_shapiro']}")
    print(f"{"Recommended dist":<{l}}: {result['recommended_dist']}")
    print("=" * 60)

    return result