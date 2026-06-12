
import numpy as np
import pandas as pd
import pymc as pm
import warnings

warnings.simplefilter(action = "ignore", category = pd.errors.PerformanceWarning)

# Segment indicators
def build_segment_indices(
    df: pd.DataFrame,
    segment_col: str,
    brand_col: str
) -> dict:

    """
    Build integer index arrays for hierarchical segment-brand structure.

    Description:
        Encodes segment and brand_tier columns to integer indices,
        and constructs a mapping from each segment to its parent brand.
        Required as inputs to PyMC hierarchical model.

    Args:
        df (pd.DataFrame) : Training DataFrame.
        segment_col (str)  : Column name for the segment variable.
        brand_col (str)    : Column name for the brand tier variable.

    Returns:
        dict: {
            "seg_idx"      : np.ndarray - integer segment index per row,
            "brand_idx"    : np.ndarray - integer brand index per row,
            "seg_names"    : list       - ordered list of segment names,
            "brand_names"  : list       - ordered list of brand names,
            "seg_to_brand" : np.ndarray - brand index for each segment,
            "n_seg"        : int        - number of unique segments,
            "n_brand"      : int        - number of unique brands
        }
    """

    segments = df[segment_col].astype("category")
    brands = df[brand_col].astype("category")

    seg_idx = segments.cat.codes.values
    brand_idx = brands.cat.codes.values
    seg_names = segments.cat.categories.tolist()
    brand_names = brands.cat.categories.tolist()

    seg_to_brand = (
        df.groupby(segment_col)[brand_col]
        .first()
        .astype("category")
        .cat.set_categories(brand_names)
        .cat.codes
        .values
    )

    return {
        "seg_idx": seg_idx,
        "brand_idx": brand_idx,
        "seg_names": seg_names,
        "brand_names": brand_names,
        "seg_to_brand": seg_to_brand,
        "n_seg": len(seg_names),
        "n_brand": len(brand_names)
    }

# Fitting Hierarchical Bayesian Weibull
def fit_hierarchical_weibull(
    df: pd.DataFrame,
    idx: dict,
    age_col: str = "age_months",
    rvr_col: str = "rvr",
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 2
) -> dict:

    """
    Fit Hierarchical Bayesian Weibull depreciation curve via PyMC.

    Description:
        Models RVR(t) = exp(-(t/lambda)^k) with:
        - Brand-level hyperpriors for log(lambda) and log(k)
        - Segment-level parameters with partial pooling toward brand mean
        - Beta likelihood for bounded RVR in (0, 1)
        Partial pooling allows sparse segments to borrow strength from
        their parent brand tier, avoiding fallback to global defaults.

    Args:
        df (pd.DataFrame) : Training DataFrame.
        idx (dict)         : Output of build_segment_indices().
        age_col (str)      : Column name for age in months (Default: "age_months").
        rvr_col (str)      : Column name for RVR target (Default: "rvr").
        draws (int)        : Posterior draws per chain (Default: 1000).
        tune (int)         : Tuning steps per chain (Default: 1000).
        chains (int)       : Number of MCMC chains (Default: 2).

    Returns:
        dict: {
            "model"        : pm.Model - compiled PyMC model,
            "trace"        : az.InferenceData - posterior samples,
            "seg_names"    : list - segment name list,
            "brand_names"  : list - brand name list,
            "seg_to_brand" : np.ndarray - segment-to-brand mapping
        }
    """

    t   = df[age_col].values.astype(float)
    rvr = df[rvr_col].values.clip(0.01, 0.99)

    with pm.Model() as model:

        # Brand-level hyperpriors for lambda (scale)
        mu_log_lam = pm.Normal(
            "mu_log_lam",
            mu = np.log(48),
            sigma = 0.5,
            shape = idx["n_brand"]
        )
        sigma_log_lam = pm.HalfNormal("sigma_log_lam", sigma = 0.3)

        # Brand-level hyperpriors for k (shape)
        mu_log_k = pm.Normal(
            "mu_log_k",
            mu = np.log(1.2),
            sigma = 0.3,
            shape = idx["n_brand"]
        )
        sigma_log_k = pm.HalfNormal("sigma_log_k", sigma = 0.2)

        # Segment-level parameters with partial pooling
        log_lam_seg = pm.Normal(
            "log_lam_seg",
            mu = mu_log_lam[idx["seg_to_brand"]],
            sigma = sigma_log_lam,
            shape = idx["n_seg"]
        )
        log_k_seg = pm.Normal(
            "log_k_seg",
            mu = mu_log_k[idx["seg_to_brand"]],
            sigma = sigma_log_k,
            shape = idx["n_seg"]
        )

        lam = pm.Deterministic("lam", pm.math.exp(log_lam_seg))
        k = pm.Deterministic("k",   pm.math.exp(log_k_seg))

        # Weibull depreciation curve
        theo_rvr = pm.Deterministic(
            "theoretical_rvr",
            pm.math.exp(-((t / lam[idx["seg_idx"]]) ** k[idx["seg_idx"]]))
        )

        # Beta likelihood
        kappa = pm.HalfNormal("kappa", sigma = 20)
        mu_obs = theo_rvr.clip(1e-6, 1 - 1e-6)

        pm.Beta(
            "rvr_obs",
            alpha = mu_obs * kappa,
            beta = (1 - mu_obs) * kappa,
            observed = rvr
        )

        trace = pm.sample(
            draws = draws,
            tune = tune,
            chains = chains,
            target_accept = 0.9,
            return_inferencedata = True,
            progressbar = True
        )

    return {
        "model": model,
        "trace": trace,
        "seg_names": idx["seg_names"],
        "brand_names": idx["brand_names"],
        "seg_to_brand": idx["seg_to_brand"]
    }

# Compute posterior
def extract_posterior_features(
    df: pd.DataFrame,
    bayes_result: dict,
    segment_col: str = "segment",
    age_col: str = "age_months",
    rvr_col: str = "rvr",
    n_samples: int = 500
) -> pd.DataFrame:

    """
    Compute posterior mean, std, and credible interval of RVR(t) per row.

    Description:
        Samples n_samples draws from the posterior distributions of
        lambda and k per segment, vectorises Weibull computation over
        all draws and observations, then summarises into per-row features.
        The residual (actual - posterior mean) becomes the Stage 2 target.

    Args:
        df (pd.DataFrame)  : DataFrame (train or test).
        bayes_result (dict) : Output of fit_hierarchical_weibull().
        segment_col (str)   : Segment column name (Default: "segment").
        age_col (str)       : Age column name (Default: "age_months").
        rvr_col (str)       : RVR column name (Default: "rvr").
        n_samples (int)     : Posterior draws to sample (Default: 500).

    Returns:
        pd.DataFrame: Input df with additional columns:
            {
                "theoretical_rvr"     : float — posterior mean curve value,
                "theoretical_rvr_std" : float — posterior std (uncertainty),
                "theoretical_rvr_p10" : float — 10th percentile,
                "theoretical_rvr_p90" : float — 90th percentile,
                "curve_uncertainty"   : float — p90 - p10 width,
                "rvr_residual"        : float — actual_rvr - theoretical_rvr
            }
    """

    df = df.copy()
    trace = bayes_result["trace"]
    seg_names = bayes_result["seg_names"]

    seg_codes = (
        df[segment_col]
        .astype("category")
        .cat.set_categories(seg_names)
        .cat.codes
        .values
    )
    seg_codes = np.where(seg_codes < 0, 0, seg_codes) #Unknown segment --> fallback

    t = df[age_col].values.astype(float)

    lam_post = trace.posterior["lam"].values.reshape(-1, len(seg_names))
    k_post = trace.posterior["k"].values.reshape(-1, len(seg_names))

    rng = np.random.default_rng(42)
    idx = rng.choice(lam_post.shape[0], n_samples, replace = False)
    lam_s = lam_post[idx] #(n_samples, n_seg)
    k_s = k_post[idx]

    # Vectorised: (n_samples, n_obs)
    lam_obs = lam_s[:, seg_codes]
    k_obs = k_s[:, seg_codes]
    rvr_samp = np.exp(-((t[None, :] / lam_obs) ** k_obs))

    df["theoretical_rvr"] = rvr_samp.mean(axis = 0)
    df["theoretical_rvr_std"] = rvr_samp.std(axis = 0)
    df["theoretical_rvr_p10"] = np.percentile(rvr_samp, 10, axis = 0)
    df["theoretical_rvr_p90"] = np.percentile(rvr_samp, 90, axis = 0)
    df["curve_uncertainty"] = df["theoretical_rvr_p90"] - df["theoretical_rvr_p10"]
    df["rvr_residual"] = df[rvr_col] - df["theoretical_rvr"]

    return df