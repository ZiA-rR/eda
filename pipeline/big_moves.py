"""
big_moves.py
------------
Identifying "big move days" in a price series.

Two methods, both volatility-relative so that assets with very different
normal volatility (crypto vs a major FX pair) are treated comparably.

METHOD 1 - standardised return (simple, practical for daily data)
    r_t     = log(P_t / P_{t-1})
    sigma_t = volatility estimated from a trailing window
    z_t     = r_t / sigma_t
    flag if |z_t| > threshold

METHOD 2 - Lee & Mykland (2008) jump test (the literature standard)
    Same idea, but two refinements that matter:

    (a) volatility is estimated with BIPOWER VARIATION, which uses products
        of adjacent absolute returns. A single large return contaminates a
        plain standard deviation (the jump inflates the very yardstick you
        are measuring it against). Bipower variation is robust to that.

            sigma_hat(t)^2 = 1/(K-2) * sum |r_j| * |r_{j-1}|

    (b) the threshold is not chosen by hand. Under the null of no jump the
        maximum of the standardised statistic follows a Gumbel distribution,
        so the cutoff comes from the chosen significance level.

    Reference: Lee, S.S. and Mykland, P.A. (2008), "Jumps in Financial
    Markets: A New Nonparametric Test and Jump Dynamics", Review of
    Financial Studies 21(6).

    Note: the test was designed for intraday data and loses power at daily
    frequency. It is included because it is the citable standard; method 1
    is the practical fallback.
"""

import numpy as np
import pandas as pd

C_CONST = np.sqrt(2.0 / np.pi)   # E|N(0,1)|, appears in the Gumbel scaling


def log_returns(prices: pd.Series) -> pd.Series:
    """Daily log returns."""
    return np.log(prices / prices.shift(1))


# ---------------------------------------------------------------- method 1
def standardised_returns(prices: pd.Series, window: int = 90,
                         min_periods: int = 30) -> pd.DataFrame:
    """
    Flag days where the return is large relative to that asset's own
    recent volatility.

    window : trailing days used to estimate volatility. 90 is a reasonable
             default: long enough to be stable, short enough to adapt to
             changing volatility regimes.
    """
    r = log_returns(prices)
    # shift(1) so volatility uses only information available BEFORE day t.
    # Without this the day's own move contaminates its own yardstick.
    sigma = r.rolling(window, min_periods=min_periods).std().shift(1)
    z = r / sigma
    return pd.DataFrame({"return": r, "sigma": sigma, "z": z})


# ---------------------------------------------------------------- method 2
def bipower_sigma(r: pd.Series, K: int = 16) -> pd.Series:
    """
    Lee-Mykland local volatility estimate.

        sigma_hat(t)^2 = 1/(K-2) * sum_{j=t-K+2}^{t-1} |r_j| * |r_{j-1}|

    Uses only returns strictly before t, so day t's own move cannot inflate
    its own volatility estimate.
    """
    abs_prod = (r.abs() * r.abs().shift(1))
    # trailing sum over the K-2 products ending at t-1
    s = abs_prod.rolling(K - 2).sum().shift(1)
    return np.sqrt(s / (K - 2))


def lee_mykland(prices: pd.Series, K: int = 16, alpha: float = 0.01) -> pd.DataFrame:
    """
    Lee-Mykland jump test.

    K     : window for the local volatility estimate. LM give guidance by
            sampling frequency; 16 is their suggested value for daily data.
    alpha : significance level for the jump test.
    """
    r = log_returns(prices)
    sigma = bipower_sigma(r, K)
    L = r / sigma

    n = int(L.notna().sum())
    if n < 2:
        raise ValueError("not enough observations for the test")

    # Gumbel normalising constants
    sqrt2logn = np.sqrt(2.0 * np.log(n))
    C_n = sqrt2logn / C_CONST - (np.log(np.pi) + np.log(np.log(n))) / (
        2.0 * C_CONST * sqrt2logn
    )
    S_n = 1.0 / (C_CONST * sqrt2logn)

    # critical value from the Gumbel distribution
    beta_star = -np.log(-np.log(1.0 - alpha))
    threshold = C_n + S_n * beta_star

    is_jump = L.abs() > threshold

    return pd.DataFrame({
        "return": r,
        "sigma": sigma,
        "L": L,
        "is_jump": is_jump,
    }).assign(threshold=threshold, n_obs=n)


# ---------------------------------------------------------------- selection
def select_big_move_days(prices: pd.Series,
                         method: str = "zscore",
                         z_threshold: float = 3.0,
                         window: int = 90,
                         K: int = 16,
                         alpha: float = 0.01) -> pd.DataFrame:
    """
    Return only the flagged days, sorted by how extreme they are.

    method : "zscore" or "lee_mykland"
    """
    if method == "zscore":
        out = standardised_returns(prices, window=window)
        flagged = out[out["z"].abs() > z_threshold].copy()
        flagged["score"] = flagged["z"].abs()
    elif method == "lee_mykland":
        out = lee_mykland(prices, K=K, alpha=alpha)
        flagged = out[out["is_jump"]].copy()
        flagged["score"] = flagged["L"].abs()
    else:
        raise ValueError("method must be 'zscore' or 'lee_mykland'")

    flagged["direction"] = np.where(flagged["return"] > 0, "up", "down")
    flagged["pct_move"] = (np.exp(flagged["return"]) - 1) * 100
    return flagged.sort_values("score", ascending=False)
